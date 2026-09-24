# coding=utf-8

import torch
import taichi as ti
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time, os, json
import imageio
import torchvision
from tqdm import tqdm, trange
import cv2
import matplotlib.pyplot as plt
from train_gs import training_MO, training
from argparse import ArgumentParser
from gaussian_renderer import render
from scene import Scene, DeformModel
from utils.general_utils import safe_state
from gaussian_renderer import GaussianModel
from simulator import MPMSimulator, Estimator
# from simulator.estimator_multi_beta import Estimator
from simulator.estimator_multi import Estimator
from train_gs_fixed_pcd import train_gs_with_fixed_pcd, assign_gs_to_pcd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from utils.system_utils import check_gs_model, draw_curve, write_particles
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args

import open3d as o3d
from pathlib import Path
import trimesh



# Try to import wandb for real-time tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with 'pip install wandb' for real-time tracking")


image_scale = 1.0


def print_torch_memory(tag=""):
    if not torch.cuda.is_available():
        print(f"[{tag}] CUDA not available")
        return
    device = torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    print(f"[{tag}] Torch CUDA mem: allocated = {alloc:.1f} MB, reserved = {reserved:.1f} MB")








def load_multiobject_gt_pcs(root_dir, n_frames=30, obj_id_dir_map=None, device="cuda"):
    """
    root_dir: e.g. 'data/03'
    n_frames: 使用多少帧 GT 点云（前几帧）
    obj_id_dir_map: dict，比如 {1: 0, 2: 1}
        - key = physics 里的 object_id (1,2,...)
        - value = point_clouds 目录下的子目录编号 (0,1,...)

    返回：
        gt_pcs_per_object: {obj_id: [T tensors], 每个 [Ni,3]}
    """
    if obj_id_dir_map is None:
        # 默认假设目录编号 = object_id - 1 （你的 config 是 object_id:1,2；目录是0,1）
        obj_id_dir_map ={1: 0, 2: 1}

    gt_pcs_per_object = {}

    for obj_id, dir_idx in obj_id_dir_map.items():
        pcs_frames = []
        for t in range(n_frames):
            ply_path = os.path.join(root_dir, "point_clouds", str(dir_idx), f"{t}.ply")
            if not os.path.exists(ply_path):
                raise FileNotFoundError(f"Missing GT point cloud: {ply_path}")
            mesh = trimesh.load(ply_path, process=False)
            pts = torch.from_numpy(mesh.vertices).float().to(device)  # [N,3]
            pcs_frames.append(pts)
        gt_pcs_per_object[obj_id] = pcs_frames

    return gt_pcs_per_object


def fix_misclassified_particles_connected_components(gaussians, distance_threshold='auto', min_cluster_size_ratio=0.1):
    """
    Fix misclassified particles using connected components clustering.
    Identifies isolated clusters and reassigns small clusters to the majority class.

    Args:
        gaussians: GaussianModel with object probabilities
        distance_threshold: Maximum distance to consider points connected (in world units)
                          'auto' to estimate from data, or a float value
        min_cluster_size_ratio: Clusters smaller than this ratio of the majority class get reassigned

    Returns:
        Corrected object assignments
    """
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
    from scipy import stats
    import numpy as np

    with torch.no_grad():
        device = gaussians._xyz.device
        positions = gaussians._xyz.cpu().numpy()
        object_probs = gaussians.get_object_probs.cpu()  # [N, 3]
        object_assignments = torch.argmax(object_probs, dim=1).numpy()  # [N] with values 0,1,2

        # Auto-estimate distance threshold if needed
        if distance_threshold == 'auto':
            # Estimate from average nearest neighbor distance
            nbrs = NearestNeighbors(n_neighbors=10).fit(positions)
            distances, _ = nbrs.kneighbors(positions)
            avg_nn_dist = distances[:, 1:].mean()  # Exclude self (distance 0)
            distance_threshold = avg_nn_dist * 3  # 3x average nearest neighbor distance
            print(f"\nAuto-estimated distance threshold: {distance_threshold:.4f} (avg NN dist: {avg_nn_dist:.4f})")

        print(f"\nFixing misclassified particles using connected components")
        print(f"Distance threshold: {distance_threshold:.4f} world units")
        print(f"Initial distribution - Background: {(object_assignments==0).sum()}, "
              f"Obj1: {(object_assignments==1).sum()}, Obj2: {(object_assignments==2).sum()}")

        corrected_assignments = object_assignments.copy()

        # Process each object class separately
        for obj_class in [1, 2]:
            # Get points belonging to this class
            class_mask = (object_assignments == obj_class)
            if class_mask.sum() == 0:
                continue

            class_positions = positions[class_mask]

            # Cluster using DBSCAN to find connected components
            clustering = DBSCAN(eps=distance_threshold, min_samples=5).fit(class_positions)
            labels = clustering.labels_

            # Find the size of each cluster
            unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)

            if len(unique_labels) == 0:
                continue

            # Find the largest cluster (main body)
            main_cluster_size = counts.max()
            min_size = int(main_cluster_size * min_cluster_size_ratio)

            print(f"  Object {obj_class}: Found {len(unique_labels)} clusters, main cluster: {main_cluster_size} points")

            # Mark small clusters for reassignment
            small_clusters = unique_labels[counts < min_size]

            if len(small_clusters) > 0:
                # Get indices of points in small clusters
                class_indices = np.where(class_mask)[0]
                for cluster_id in small_clusters:
                    cluster_points = class_indices[labels == cluster_id]

                    # Find nearest points from other class
                    other_class = 2 if obj_class == 1 else 1
                    other_mask = (object_assignments == other_class)

                    if other_mask.sum() > 0:
                        other_positions = positions[other_mask]

                        # For each point in small cluster, check distance to other class
                        for point_idx in cluster_points:
                            point_pos = positions[point_idx:point_idx+1]
                            distances = np.linalg.norm(other_positions - point_pos, axis=1)

                            # If close enough to other class, reassign
                            if distances.min() < distance_threshold * 2:
                                corrected_assignments[point_idx] = other_class

                print(f"    Reassigned {len(small_clusters)} small clusters")

        print(f"Final distribution - Background: {(corrected_assignments==0).sum()}, "
              f"Obj1: {(corrected_assignments==1).sum()}, Obj2: {(corrected_assignments==2).sum()}")

        return torch.tensor(corrected_assignments, dtype=torch.long, device=device)


def fix_misclassified_particles(gaussians, k_neighbors=50, min_ratio=0.55, iterations=5, use_connected_components=True):
    """
    Fix misclassified particles using either KNN or connected components.
    """
    if use_connected_components:
        return fix_misclassified_particles_connected_components(gaussians)

    # Original KNN implementation
    from sklearn.neighbors import NearestNeighbors

    with torch.no_grad():
        device = gaussians._xyz.device
        positions = gaussians._xyz.cpu().numpy()
        object_probs = gaussians.get_object_probs.cpu()  # [N, 3]
        object_assignments = torch.argmax(object_probs, dim=1).numpy()  # [N] with values 0,1,2

        print(f"\nFixing misclassified particles using KNN with k={k_neighbors}")
        print(f"Initial distribution - Background: {(object_assignments==0).sum()}, "
              f"Obj1: {(object_assignments==1).sum()}, Obj2: {(object_assignments==2).sum()}")

        # Build KNN model
        nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, len(positions)), algorithm='auto')
        nbrs.fit(positions)

        corrected_assignments = object_assignments.copy()

        for iteration in range(iterations):
            changes = 0
            new_assignments = corrected_assignments.copy()

            # Find neighbors for all points
            distances, indices = nbrs.kneighbors(positions)

            for i in range(len(positions)):
                if corrected_assignments[i] == 0:  # Skip background
                    continue

                # Get neighbor labels (excluding self)
                neighbor_labels = corrected_assignments[indices[i, 1:]]

                # Count non-background neighbors
                non_bg_neighbors = neighbor_labels[neighbor_labels > 0]

                if len(non_bg_neighbors) == 0:
                    continue

                # Count occurrences of each object label
                obj1_count = (non_bg_neighbors == 1).sum()
                obj2_count = (non_bg_neighbors == 2).sum()
                total_count = len(non_bg_neighbors)

                # Determine majority label
                if obj1_count > obj2_count:
                    majority_label = 1
                    majority_ratio = obj1_count / total_count
                else:
                    majority_label = 2
                    majority_ratio = obj2_count / total_count

                # Change label if strong majority and different from current
                if majority_ratio >= min_ratio and corrected_assignments[i] != majority_label:
                    new_assignments[i] = majority_label
                    changes += 1

            corrected_assignments = new_assignments
            print(f"  Iteration {iteration+1}: {changes} particles corrected")

            if changes == 0:
                break

        print(f"Final distribution - Background: {(corrected_assignments==0).sum()}, "
              f"Obj1: {(corrected_assignments==1).sum()}, Obj2: {(corrected_assignments==2).sum()}")

        # Convert back to torch tensor on original device
        return torch.tensor(corrected_assignments, dtype=torch.long, device=device)


def filter_gaussians_by_object(gaussians, object_id, use_clustering_fix=False):
    """
    Filter Gaussians by object assignment using argmax.
    Returns a new GaussianModel containing only Gaussians assigned to object_id.
    object_id: 1 for obj1, 2 for obj2
    """
    with torch.no_grad():
        if use_clustering_fix and hasattr(gaussians, '_corrected_assignments'):
            # Use pre-computed corrected assignments if available
            object_assignments = gaussians._corrected_assignments
        elif use_clustering_fix:
            # Compute corrected assignments if not already done
            object_assignments = fix_misclassified_particles(gaussians)
        else:
            # Use original argmax assignments
            object_probs = gaussians.get_object_probs  # [N, 3] softmax probabilities
            object_assignments = torch.argmax(object_probs, dim=1)  # [N] with values 0,1,2

        mask = (object_assignments == object_id)  # Boolean mask for this object

        # Ensure mask is on the same device as the data
        device = gaussians._xyz.device
        mask = mask.to(device)
        print(f"Object {object_id}: {mask.sum().item()} / {len(mask)} Gaussians")

        # Create new filtered Gaussian model
        filtered = GaussianModel(gaussians.active_sh_degree)

        # Filter all attributes
        filtered._xyz = gaussians._xyz[mask]
        filtered._features_dc = gaussians._features_dc[mask]
        filtered._features_rest = gaussians._features_rest[mask]
        filtered._scaling = gaussians._scaling[mask]
        filtered._rotation = gaussians._rotation[mask]
        filtered._opacity = gaussians._opacity[mask]
        filtered._object_logits = gaussians._object_logits[mask]

        # Copy scalar attributes
        filtered.active_sh_degree = gaussians.active_sh_degree
        filtered.max_sh_degree = gaussians.max_sh_degree

        return filtered



def prepare_gt_multi(dataset: ModelParams, iteration: int, pipeline: PipelineParams, phys_args, use_clustering_fix=True):
    """
    Prepare ground truth for multi-object scene by processing each object separately.
    """
    import copy

    print("Preparing ground truth for multi-object scene...")

    # Load full Gaussians
    gaussians_full = GaussianModel(dataset.sh_degree)
    scene_full = Scene(dataset, gaussians_full, load_iteration=iteration, shuffle=False, resolution_scales=[image_scale])

    # Store original gaussians
    original_gaussians = scene_full.gaussians

    # Apply clustering fix if enabled
    if use_clustering_fix:
        print("\n=== Applying clustering-based particle correction ===")

        # Choose method: 'connected_components' or 'knn'
        method = getattr(phys_args, 'cluster_method', 'connected_components')

        if method == 'connected_components':
            # Connected components parameters
            distance_threshold = getattr(phys_args, 'cluster_distance_threshold', 'auto')
            min_cluster_size_ratio = getattr(phys_args, 'cluster_min_size_ratio', 0.1)

            corrected_assignments = fix_misclassified_particles_connected_components(
                original_gaussians,
                distance_threshold=distance_threshold,
                min_cluster_size_ratio=min_cluster_size_ratio
            )
        else:
            # KNN parameters (fallback)
            k_neighbors = getattr(phys_args, 'cluster_k_neighbors', 500)
            min_ratio = getattr(phys_args, 'cluster_min_ratio', 0.6)
            iterations = getattr(phys_args, 'cluster_iterations', 10)

            corrected_assignments = fix_misclassified_particles(
                original_gaussians,
                k_neighbors=k_neighbors,
                min_ratio=min_ratio,
                iterations=iterations,
                use_connected_components=False
            )
        # Store corrected assignments for use in filter function
        original_gaussians._corrected_assignments = corrected_assignments

    # Process Object 1
    print("\n=== Processing Object 1 ===")
    gaussians_obj1 = filter_gaussians_by_object(original_gaussians, object_id=1, use_clustering_fix=use_clustering_fix)
    scene_full.gaussians = gaussians_obj1  # Update scene's gaussians
    output1 = prepare_gt(dataset, iteration, pipeline, phys_args, gaussians=gaussians_obj1, scene=scene_full, object_id=1)
    print(f"Object 1 ({phys_args.sub_objects[0]['name']}) after filling: {len(output1[1]):,} particles")

    # Process Object 2
    print("\n=== Processing Object 2 ===")
    gaussians_obj2 = filter_gaussians_by_object(original_gaussians, object_id=2, use_clustering_fix=use_clustering_fix)
    scene_full.gaussians = gaussians_obj2  # Update scene's gaussians
    output2 = prepare_gt(dataset, iteration, pipeline, phys_args, gaussians=gaussians_obj2, scene=scene_full, object_id=2)
    print(f"Object 2 ({phys_args.sub_objects[1]['name']}) after filling: {len(output2[1]):,} particles")
    
    # Restore original gaussians
    scene_full.gaussians = original_gaussians
    
    # Unpack outputs
    gts1, vol1, vol_densities1, grid_size1, volume_surface1, cam_info1 = output1
    gts2, vol2, vol_densities2, grid_size2, volume_surface2, cam_info2 = output2
    
    print("\n=== Combining Results ===")
    
    # Simple copies (same for both)
    grid_size = grid_size1
    cam_info = cam_info1

    # Combine volumes and densities using torch
    if len(vol1) > 0 and len(vol2) > 0:
        vol_combined = torch.cat([vol1, vol2], dim=0)
        vol_densities_combined = torch.cat([vol_densities1, vol_densities2], dim=0)
        
        # Fix surface indices for second object (add offset)
        volume_surface2_offset = volume_surface2 + len(vol1)
        volume_surface_combined = torch.cat([volume_surface1, volume_surface2_offset])
        
        # Create object labels for physics
        object_labels = torch.cat([
            torch.ones(len(vol1), dtype=torch.int32, device=vol1.device),
            torch.ones(len(vol2), dtype=torch.int32, device=vol2.device) * 2
        ])
        
        print(f"Combined: {len(vol_combined)} particles (Obj1: {len(vol1)}, Obj2: {len(vol2)})")
    elif len(vol1) > 0:
        vol_combined = vol1
        vol_densities_combined = vol_densities1
        volume_surface_combined = volume_surface1
        object_labels = np.ones(len(vol1), dtype=np.int32)
        print(f"Only Object 1: {len(vol1)} particles")
    elif len(vol2) > 0:
        vol_combined = vol2
        vol_densities_combined = vol_densities2
        volume_surface_combined = volume_surface2
        object_labels = np.ones(len(vol2), dtype=np.int32) * 2
        print(f"Only Object 2: {len(vol2)} particles")
    else:
        vol_combined = np.array([])
        vol_densities_combined = np.array([])
        volume_surface_combined = np.array([])
        object_labels = np.array([])
        print("Warning: No particles found!")
    
    # Combine gts frames using torch
    gts_combined = []
    max_frames = max(len(gts1) if gts1 else 0, len(gts2) if gts2 else 0)
    for i in range(max_frames):
        frame_points = []
        if i < len(gts1):
            frame_points.append(gts1[i])
        if i < len(gts2):
            frame_points.append(gts2[i])
        if frame_points:
            gts_combined.append(torch.cat(frame_points, dim=0))
    
    # Store object labels in cam_info
    cam_info["object_labels"] = object_labels

    # Save GT surface points for all frames
    if gts_combined:
        for frame_idx, frame_gts in enumerate(gts_combined):
            # Create colors based on which object each point came from
            # Estimate based on position in concatenated tensor
            n_pts = len(frame_gts)
            colors = torch.zeros((n_pts, 3), device=frame_gts.device)
            
            # Rough estimate: first part is obj1, second part is obj2
            if frame_idx < len(gts1) and frame_idx < len(gts2):
                n_obj1 = len(gts1[frame_idx])
                colors[:n_obj1] = torch.tensor([1.0, 0.784, 0.157], device=frame_gts.device)  # Object 1: yellow/gold
                colors[n_obj1:] = torch.tensor([0.004, 0.267, 0.129], device=frame_gts.device)  # Object 2: green
            elif frame_idx < len(gts1):
                colors[:] = torch.tensor([1.0, 0.784, 0.157], device=frame_gts.device)  # All obj1
            else:
                colors[:] = torch.tensor([0.004, 0.267, 0.129], device=frame_gts.device)  # All obj2
            
            colors_np = (colors * 255).cpu().numpy().astype(np.uint8)
            write_particles(frame_gts, frame_idx, dataset.model_path, 'gt_surface', vertex_colors=colors_np)
        
        print(f"Saved GT surface points for all {len(gts_combined)} frames to {dataset.model_path}/mpm/gt_surface_*.ply")
        print(f"Points in frame 0: {len(gts_combined[0]) if gts_combined else 0}")
    
    # Save combined multi-object particles with color coding
    if len(vol_combined) > 0:
        # Create vertex colors based on object IDs
        colors = torch.zeros((len(vol_combined), 3), device=vol_combined.device)
        colors[object_labels == 1] = torch.tensor([1.0, 0.784, 0.157], device=vol_combined.device)  # Object 1: yellow/gold
        colors[object_labels == 2] = torch.tensor([0.004, 0.267, 0.129], device=vol_combined.device)  # Object 2: green
        
        # Convert to numpy for saving
        colors_np = (colors * 255).cpu().numpy().astype(np.uint8)
        
        # Save combined particles with colors and object labels using plyfile
        from utils.system_utils import write_ply_with_labels
        filepath = os.path.join(dataset.model_path, 'mpm', 'multi_object_0.ply')
        os.makedirs(os.path.join(dataset.model_path, 'mpm'), exist_ok=True)
        write_ply_with_labels(filepath, vol_combined, colors_np, object_labels)
        print(f"Saved combined particles with object labels to {dataset.model_path}/mpm/multi_object_0.ply")
        
        # Create per-particle material types from object IDs and sub_objects config
        particle_materials = torch.zeros(len(vol_combined), dtype=torch.int32, device=vol_combined.device)
        for obj_config in phys_args.sub_objects:
            obj_id = obj_config['object_id']
            material_type = obj_config['material']  # Should be 10 (elastic) for both
            particle_materials[object_labels == obj_id] = material_type
        
        # Store in cam_info for later use
        cam_info["particle_materials"] = particle_materials
        print(f"Material types - Object 1: {phys_args.sub_objects[0]['material']}, Object 2: {phys_args.sub_objects[1]['material']}")
    
    gts_per_object = {
        1: gts1,
        2: gts2
    }
    
    # Save per-object particles separately for verification
    if len(vol_combined) > 0:
        mpm_path = os.path.join(dataset.model_path, "mpm")
        os.makedirs(mpm_path, exist_ok=True)
        
        # Save particles for Object 1
        obj1_mask = object_labels == 1
        if obj1_mask.any():
            obj1_particles = vol_combined[obj1_mask]
            # Create yellow/gold color for object 1
            obj1_colors = torch.full((len(obj1_particles), 3), fill_value=255, device=obj1_particles.device, dtype=torch.uint8)
            obj1_colors[:, 0] = 255  # R
            obj1_colors[:, 1] = 200  # G  
            obj1_colors[:, 2] = 40   # B (yellow/gold)
            obj1_colors_np = obj1_colors.cpu().numpy()
            
            write_particles(obj1_particles, 0, dataset.model_path, 'object1_particles', vertex_colors=obj1_colors_np)
            print(f"Saved Object 1 particles ({len(obj1_particles)} points) to {mpm_path}/object1_particles_0.ply")
        
        # Save particles for Object 2
        obj2_mask = object_labels == 2
        if obj2_mask.any():
            obj2_particles = vol_combined[obj2_mask]
            # Create green color for object 2
            obj2_colors = torch.full((len(obj2_particles), 3), fill_value=255, device=obj2_particles.device, dtype=torch.uint8)
            obj2_colors[:, 0] = 1    # R
            obj2_colors[:, 1] = 68   # G
            obj2_colors[:, 2] = 33   # B (green)
            obj2_colors_np = obj2_colors.cpu().numpy()
            
            write_particles(obj2_particles, 0, dataset.model_path, 'object2_particles', vertex_colors=obj2_colors_np)
            print(f"Saved Object 2 particles ({len(obj2_particles)} points) to {mpm_path}/object2_particles_0.ply")
        
        # Print statistics
        print(f"\nParticle Statistics:")
        print(f"  Total particles: {len(object_labels)}")
        print(f"  Object 1 particles: {obj1_mask.sum().item()} ({obj1_mask.sum().item()/len(object_labels)*100:.1f}%)")
        print(f"  Object 2 particles: {obj2_mask.sum().item()} ({obj2_mask.sum().item()/len(object_labels)*100:.1f}%)")
    
    return gts_per_object, gts_combined, vol_combined, vol_densities_combined, grid_size, volume_surface_combined, cam_info, vol1, vol2


def prepare_gt(dataset: ModelParams, iteration: int, pipeline: PipelineParams, phys_args, gaussians=None, scene=None, object_id=None):
    gts = []
    if gaussians is None:
        gaussians = GaussianModel(dataset.sh_degree)
    if scene is None:
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales=[image_scale], object_id=object_id)
    deform = DeformModel(dataset)
    deform.load_weights(dataset.model_path)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    views = scene.getTrainCameras(scale=image_scale)
    fids = torch.unique(torch.stack([view.fid for view in views]))
    xyz_canonical = gaussians.get_xyz.detach()
    opacitiy = gaussians.get_opacity.squeeze()
    grid_size = phys_args.density_grid_size
    density_min_th = phys_args.density_min_th
    density_max_th = phys_args.density_max_th
    num_iter = 4 if phys_args.random_sample else 5
    filling_grid_size = grid_size / 2 ** 5
    opacity_threshold = phys_args.opacity_threshold
    with torch.no_grad():
        for idx, fid in enumerate(tqdm(fids, desc="Filling progress")):
            if getattr(phys_args, "n_frames", None) and idx >= phys_args.n_frames:
                break
            time_input = fid.unsqueeze(0).expand(1, -1)
            d_xyz, d_rotation, d_scaling = deform.step(xyz_canonical, time_input)
            xyzt = xyz_canonical + d_xyz
            # xyzt = xyzt[opacitiy > opacity_threshold]
            bbox_mins = xyzt[opacitiy > opacity_threshold].min(dim=0)[0] - grid_size
            bbox_maxs = xyzt[opacitiy > opacity_threshold].max(dim=0)[0] + grid_size
            bbox_bounds = bbox_maxs - bbox_mins
            volume_size = torch.round(bbox_bounds / filling_grid_size).to(torch.int64) + 1
            grid_ids = [torch.arange(size) for size in volume_size]
            grid_coords = torch.stack(torch.meshgrid(*grid_ids, indexing='ij'), dim=-1).reshape(-1, 3) * filling_grid_size
            grid_coords = grid_coords.to(xyzt)
            init_inner_points = grid_coords + bbox_mins.reshape(1, 3)
            curr_views = [view for view in views if view.fid == fid]  
            for viewpoint_cam in tqdm(curr_views, desc="Rendering progress"):
                results = render(viewpoint_cam, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, False)
                depth = results["depth"][0]
                render_mask = torch.logical_and(results["render"].sum(0) != 0, 
                                                viewpoint_cam.original_image.sum(0) != 0)
                pix_w, pix_h, pix_d = viewpoint_cam.pw2pix(init_inner_points)
                # remove points that are outside the image space
                in_mask = viewpoint_cam.is_in_view(pix_w, pix_h)
                init_inner_points = init_inner_points[in_mask]
                pix_w, pix_h, pix_d = pix_w[in_mask], pix_h[in_mask], pix_d[in_mask]
                # remove points that the projected pixels are outside the object mask
                pix_mask = render_mask[pix_h, pix_w]
                init_inner_points = init_inner_points[pix_mask]
                pix_w, pix_h, pix_d = pix_w[pix_mask], pix_h[pix_mask], pix_d[pix_mask]
                # remove points whose depth values are smaller than those from depth map
                render_pix_d = depth[pix_h, pix_w]
                depth_mask = render_pix_d < pix_d
                init_inner_points = init_inner_points[depth_mask]
                # remove outliers in xyzt
                render_mask = results["render"].sum(0) > 1 / 255
                pix_w, pix_h, pix_d = viewpoint_cam.pw2pix(xyzt)
                in_mask = viewpoint_cam.is_in_view(pix_w, pix_h)
                pix_w, pix_h, pix_d = pix_w[in_mask], pix_h[in_mask], pix_d[in_mask]
                xyzt = xyzt[in_mask]
                pix_mask = render_mask[pix_h, pix_w]
                xyzt = xyzt[pix_mask]
            curr_grid_size = grid_size / 2
            volume_size = torch.round(bbox_bounds / curr_grid_size).to(torch.int64) + 1
            bbox_maxs = bbox_mins + (volume_size - 1) * curr_grid_size
            bbox_bounds = bbox_maxs - bbox_mins
            density_volume = torch.zeros(volume_size.cpu().numpy().tolist()).to(init_inner_points)
            ids = torch.round((init_inner_points - bbox_mins.reshape(1, 3)) / curr_grid_size).to(torch.int64)
            density_volume[ids.T[0], ids.T[1], ids.T[2]] = 1.0
            ids = torch.round((xyzt - bbox_mins.reshape(1, 3)) / curr_grid_size).to(torch.int64)
            density_volume[ids.T[0], ids.T[1], ids.T[2]] = 1.0
            weight = torch.ones((1, 1, 3, 3, 3)).to(xyzt)
            weight = weight / weight.sum()
            for i in range(2, num_iter):
                curr_grid_size = grid_size / 2 ** i
                volume_size = torch.round(bbox_bounds / curr_grid_size).to(torch.int64) + 1
                bbox_maxs = bbox_mins + (volume_size - 1) * curr_grid_size
                grid_xyz = torch.stack(torch.meshgrid(
                    torch.linspace(0, volume_size[0]-1, volume_size[0]),
                    torch.linspace(0, volume_size[1]-1, volume_size[1]),
                    torch.linspace(0, volume_size[2]-1, volume_size[2]),
                ), dim=-1).to(bbox_mins) * curr_grid_size + bbox_mins[None, None, None]
                ids_norm = (grid_xyz - bbox_mins[None, None, None]) / bbox_bounds[None, None, None] * 2 - 1
                ids_norm = ids_norm[None].flip((-1,))
                density_volume = torch.nn.functional.grid_sample(density_volume[None, None], ids_norm, mode='bilinear', align_corners=True)
                density_volume = torch.nn.functional.conv3d(density_volume, weight=weight, padding='same')[0, 0]
                density_volume[density_volume < 0.5] = 0.0
                ids = torch.round((init_inner_points - bbox_mins.reshape(1, 3)) / curr_grid_size).to(torch.int64)
                density_volume[ids.T[0], ids.T[1], ids.T[2]] = 1.0
                ids = torch.round((xyzt - bbox_mins.reshape(1, 3)) / curr_grid_size).to(torch.int64)
                density_volume[ids.T[0], ids.T[1], ids.T[2]] = 1.0
                bbox_bounds = bbox_maxs - bbox_mins
            for i in range(20):
                density_volume = torch.nn.functional.conv3d(density_volume[None, None], weight=weight, padding='same')[0, 0]
                density_volume[density_volume < 0.5] = 0.0
                ids = torch.round((init_inner_points - bbox_mins.reshape(1, 3)) / curr_grid_size).to(torch.int64)
                density_volume[ids.T[0], ids.T[1], ids.T[2]] = 1.0
                ids = torch.round((xyzt - bbox_mins.reshape(1, 3)) / curr_grid_size).to(torch.int64)
                density_volume[ids.T[0], ids.T[1], ids.T[2]] = 1.0
            if phys_args.random_sample:
                density_volume = torch.nn.functional.conv3d(density_volume[None, None], weight=weight, padding='same')[0, 0]
                half_grid_xyz = torch.stack(torch.meshgrid(
                    torch.linspace(0, volume_size[0]-0.5, 2 * (volume_size[0]-1)),
                    torch.linspace(0, volume_size[1]-0.5, 2 * (volume_size[1]-1)),
                    torch.linspace(0, volume_size[2]-0.5, 2 * (volume_size[2]-1)),
                ), -1).to(bbox_mins) * curr_grid_size + bbox_mins[None, None, None]
                ids_norm = (half_grid_xyz - bbox_mins[None, None, None]) / bbox_bounds[None, None, None] * 2 - 1
                ids_norm = ids_norm[None].flip((-1,))
                density_half_grid_xyz = torch.nn.functional.grid_sample(density_volume[None, None], ids_norm, mode='bilinear', align_corners=True)[0, 0]
                half_grid_xyz = half_grid_xyz[density_half_grid_xyz > 0.5]
                delta = (torch.rand_like(half_grid_xyz) * curr_grid_size * 0.5).to(xyzt)
                particles = half_grid_xyz + delta
                ids_norm = (particles[None,None] - bbox_mins[None, None, None]) / bbox_bounds[None, None, None] * 2 - 1
                ids_norm = ids_norm[None].flip((-1,))
                density_particles = torch.nn.functional.grid_sample(density_volume[None, None], ids_norm, mode='bilinear', align_corners=True)[0, 0, 0, 0]
                sampled_pts = particles[density_particles > density_min_th]
                surface_pts = particles[(density_particles > density_min_th)*(density_particles < density_max_th)]
                gts.append(surface_pts)
                curr_grid_size = curr_grid_size / 2
                if fid == 0.0:
                    vol = sampled_pts
                    vol_densities = density_particles[density_particles > density_min_th]
                    vol_surface_mask = density_particles[density_particles > density_min_th] < density_max_th
                    vol_surface = torch.arange(vol_surface_mask.shape[0]).to(vol.device).to(torch.int64)[vol_surface_mask]
            else:
                density_volume = torch.nn.functional.conv3d(density_volume[None, None], weight=weight, padding='same')[0, 0]
                internal_mask = density_volume >= density_min_th
                sampled_pts = torch.stack(torch.where(internal_mask), dim=-1) * curr_grid_size + bbox_mins.reshape(1, 3)
                density_volume_smoothed = torch.nn.functional.conv3d(density_volume[None, None], weight=weight, padding='same')[0, 0]
                surface_mask = (density_volume_smoothed > 0) * (density_volume_smoothed < density_max_th) * internal_mask
                surface_pts = torch.stack(torch.where(surface_mask == 1), dim=-1) * curr_grid_size + bbox_mins.reshape(1, 3)
                gts.append(surface_pts)
                if fid == 0.0:
                    vol = sampled_pts
                    vol_densities = density_volume[internal_mask]
                    vol_surface_mask = surface_mask[internal_mask]
                    vol_surface = torch.arange(vol_surface_mask.shape[0]).to(vol.device).to(torch.int64)[vol_surface_mask]

        train_cams, test_cams, cameras_extent = scene.overwrite_alphas(pipeline, dataset, deform, object_id)
        cam_info = {
            "train_cams": train_cams,
            "test_cams": test_cams,
            "cameras_extent": cameras_extent,
        }
    return gts, vol, vol_densities, torch.tensor([curr_grid_size]), vol_surface, cam_info

def filter_gaussians_by_object_with_pcds(
    gaussians,
    object_id: int,
    gt_pcs_per_object: dict,
    use_clustering_fix: bool = False,   # 现在不再用，保留只是为了接口兼容
):
    """
    根据 GT 点云的长度和顺序来给 Gaussians 做硬划分：
      - 假设 canonical 顺序为 [obj1 所有点, obj2 所有点]
      - N1 = gt_pcs_per_object[1][0].shape[0]
      - N2 = gt_pcs_per_object[2][0].shape[0]
      - object_id == 1 -> 取 [0:N1]
      - object_id == 2 -> 取 [N1:N1+N2]

    返回一个新的 GaussianModel，里面只保留该 object 对应的 Gaussians。
    """

    assert object_id in (1, 2), f"object_id 只能是 1 或 2，当前是 {object_id}"

    device = gaussians._xyz.device
    N_total = gaussians._xyz.shape[0]

    # ---- 1. 从 GT 点云推 N1 / N2 ----
    assert 1 in gt_pcs_per_object and 2 in gt_pcs_per_object, \
        f"gt_pcs_per_object.keys() = {list(gt_pcs_per_object.keys())}，需要包含 1 和 2"

    gt_obj1_frames = gt_pcs_per_object[1]   # list of [N1,3]
    gt_obj2_frames = gt_pcs_per_object[2]   # list of [N2,3]

    N1 = gt_obj1_frames[0].shape[0]
    N2 = gt_obj2_frames[0].shape[0]

    assert N1 + N2 == N_total, \
        f"Gaussians 数量 {N_total} 与 GT 点云长度 N1+N2 = {N1}+{N2} 不一致，" \
        f"请检查 train_fixed_pcd 是否 densify/prune 了，或点顺序是否改变。"

    if object_id == 1:
        idx = torch.arange(0, N1, device=device)
    else:  # object_id == 2
        idx = torch.arange(N1, N1 + N2, device=device)

    print(f"[filter_gaussians_by_object_with_pcds] Object {object_id}: "
          f"{idx.numel()} / {N_total} Gaussians")

    # ---- 2. 构造一个新的 GaussianModel，只拷贝该 object 的那部分属性 ----
    filtered = GaussianModel(gaussians.active_sh_degree)

    # 这些是 per-Gaussian 的 tensor 属性，都按 idx 过滤
    for attr in ["_xyz", "_features_dc", "_features_rest",
                 "_scaling", "_rotation", "_opacity"]:
        if hasattr(gaussians, attr):
            setattr(filtered, attr, getattr(gaussians, attr)[idx])

    # 如果有 object_logits（多物体分类用），也顺便切一下
    if hasattr(gaussians, "_object_logits"):
        filtered._object_logits = gaussians._object_logits[idx]

    # 拷贝一些 scalar / 配置型属性（不依赖 idx）
    filtered.active_sh_degree = gaussians.active_sh_degree
    filtered.max_sh_degree = gaussians.max_sh_degree

    # 其他比如 activation 函数、optimizer 之类，在 prepare_gt 里一般用不到，
    # 只要几何和外观参数在就够了。如果后面需要，也可以按需拷贝。

    return filtered





def prepare_gt_multi_with_pcds(dataset: ModelParams, iteration: int, pipeline: PipelineParams, phys_args, gt_pcs_per_object, use_clustering_fix=False):
    """
    Prepare ground truth for multi-object scene by processing each object separately.
    """
    import copy

    print("Preparing ground truth for multi-object scene...")

    # Load full Gaussians
    gaussians_full = GaussianModel(dataset.sh_degree)
    scene_full = Scene(dataset, gaussians_full, load_iteration=iteration, shuffle=False, resolution_scales=[image_scale], load_fix_pcd = True)

    # Store original gaussians
    original_gaussians = scene_full.gaussians

    # Process Object 1
    print("\n=== Processing Object 1 ===")
    gaussians_obj1 = filter_gaussians_by_object_with_pcds(original_gaussians, object_id=1, gt_pcs_per_object=gt_pcs_per_object, use_clustering_fix=use_clustering_fix)
    scene_full.gaussians = gaussians_obj1  # Update scene's gaussians
    output1 = prepare_gt_with_pcds(dataset, iteration, pipeline, phys_args, gaussians=gaussians_obj1, scene=scene_full, object_id=1, gt_pcs_per_object=gt_pcs_per_object)
    # output1 = prepare_gt_with_pcds(dataset, iteration, pipeline, phys_args, gaussians=original_gaussians, scene=scene_full, object_id=1, gt_pcs_per_object=gt_pcs_per_object)
    print(f"Object 1 ({phys_args.sub_objects[0]['name']}) after filling: {len(output1[1]):,} particles")

    # Process Object 2
    print("\n=== Processing Object 2 ===")
    gaussians_obj2 = filter_gaussians_by_object_with_pcds(original_gaussians, object_id=2, gt_pcs_per_object=gt_pcs_per_object, use_clustering_fix=use_clustering_fix)
    scene_full.gaussians = gaussians_obj2  # Update scene's gaussians
    output2 = prepare_gt_with_pcds(dataset, iteration, pipeline, phys_args, gaussians=gaussians_obj2, scene=scene_full, object_id=2, gt_pcs_per_object=gt_pcs_per_object)
    # output2 = prepare_gt_with_pcds(dataset, iteration, pipeline, phys_args, gaussians=original_gaussians, scene=scene_full, object_id=2, gt_pcs_per_object=gt_pcs_per_object)
    print(f"Object 2 ({phys_args.sub_objects[1]['name']}) after filling: {len(output2[1]):,} particles")
    
    # Restore original gaussians
    scene_full.gaussians = original_gaussians
    
    # Unpack outputs
    gts1, vol1, vol_densities1, grid_size1, volume_surface1, cam_info1 = output1
    gts2, vol2, vol_densities2, grid_size2, volume_surface2, cam_info2 = output2
    
    print("\n=== Combining Results ===")
    
    # Simple copies (same for both)
    grid_size = grid_size1
    cam_info = cam_info1

    # Combine volumes and densities using torch
    if len(vol1) > 0 and len(vol2) > 0:
        vol_combined = torch.cat([vol1, vol2], dim=0)
        vol_densities_combined = torch.cat([vol_densities1, vol_densities2], dim=0)
        
        # Fix surface indices for second object (add offset)
        volume_surface2_offset = volume_surface2 + len(vol1)
        volume_surface_combined = torch.cat([volume_surface1, volume_surface2_offset])
        
        # Create object labels for physics
        object_labels = torch.cat([
            torch.ones(len(vol1), dtype=torch.int32, device=vol1.device),
            torch.ones(len(vol2), dtype=torch.int32, device=vol2.device) * 2
        ])
        
        print(f"Combined: {len(vol_combined)} particles (Obj1: {len(vol1)}, Obj2: {len(vol2)})")
    elif len(vol1) > 0:
        vol_combined = vol1
        vol_densities_combined = vol_densities1
        volume_surface_combined = volume_surface1
        object_labels = np.ones(len(vol1), dtype=np.int32)
        print(f"Only Object 1: {len(vol1)} particles")
    elif len(vol2) > 0:
        vol_combined = vol2
        vol_densities_combined = vol_densities2
        volume_surface_combined = volume_surface2
        object_labels = np.ones(len(vol2), dtype=np.int32) * 2
        print(f"Only Object 2: {len(vol2)} particles")
    else:
        vol_combined = np.array([])
        vol_densities_combined = np.array([])
        volume_surface_combined = np.array([])
        object_labels = np.array([])
        print("Warning: No particles found!")
    
    # Combine gts frames using torch
    gts_combined = []
    max_frames = max(len(gts1) if gts1 else 0, len(gts2) if gts2 else 0)
    for i in range(max_frames):
        frame_points = []
        if i < len(gts1):
            frame_points.append(gts1[i])
        if i < len(gts2):
            frame_points.append(gts2[i])
        if frame_points:
            gts_combined.append(torch.cat(frame_points, dim=0))
    
    # Store object labels in cam_info
    cam_info["object_labels"] = object_labels

    # Save GT surface points for all frames
    if gts_combined:
        for frame_idx, frame_gts in enumerate(gts_combined):
            # Create colors based on which object each point came from
            # Estimate based on position in concatenated tensor
            n_pts = len(frame_gts)
            colors = torch.zeros((n_pts, 3), device=frame_gts.device)
            
            # Rough estimate: first part is obj1, second part is obj2
            if frame_idx < len(gts1) and frame_idx < len(gts2):
                n_obj1 = len(gts1[frame_idx])
                colors[:n_obj1] = torch.tensor([1.0, 0.784, 0.157], device=frame_gts.device)  # Object 1: yellow/gold
                colors[n_obj1:] = torch.tensor([0.004, 0.267, 0.129], device=frame_gts.device)  # Object 2: green
            elif frame_idx < len(gts1):
                colors[:] = torch.tensor([1.0, 0.784, 0.157], device=frame_gts.device)  # All obj1
            else:
                colors[:] = torch.tensor([0.004, 0.267, 0.129], device=frame_gts.device)  # All obj2
            
            colors_np = (colors * 255).cpu().numpy().astype(np.uint8)
            write_particles(frame_gts, frame_idx, dataset.model_path, 'gt_surface', vertex_colors=colors_np)
        
        print(f"Saved GT surface points for all {len(gts_combined)} frames to {dataset.model_path}/mpm/gt_surface_*.ply")
        print(f"Points in frame 0: {len(gts_combined[0]) if gts_combined else 0}")
    
    # Save combined multi-object particles with color coding
    if len(vol_combined) > 0:
        # Create vertex colors based on object IDs
        colors = torch.zeros((len(vol_combined), 3), device=vol_combined.device)
        colors[object_labels == 1] = torch.tensor([1.0, 0.784, 0.157], device=vol_combined.device)  # Object 1: yellow/gold
        colors[object_labels == 2] = torch.tensor([0.004, 0.267, 0.129], device=vol_combined.device)  # Object 2: green
        
        # Convert to numpy for saving
        colors_np = (colors * 255).cpu().numpy().astype(np.uint8)
        
        # Save combined particles with colors and object labels using plyfile
        from utils.system_utils import write_ply_with_labels
        filepath = os.path.join(dataset.model_path, 'mpm', 'multi_object_0.ply')
        os.makedirs(os.path.join(dataset.model_path, 'mpm'), exist_ok=True)
        write_ply_with_labels(filepath, vol_combined, colors_np, object_labels)
        print(f"Saved combined particles with object labels to {dataset.model_path}/mpm/multi_object_0.ply")
        
        # Create per-particle material types from object IDs and sub_objects config
        particle_materials = torch.zeros(len(vol_combined), dtype=torch.int32, device=vol_combined.device)
        for obj_config in phys_args.sub_objects:
            obj_id = obj_config['object_id']
            material_type = obj_config['material']  # Should be 10 (elastic) for both
            particle_materials[object_labels == obj_id] = material_type
        
        # Store in cam_info for later use
        cam_info["particle_materials"] = particle_materials
        print(f"Material types - Object 1: {phys_args.sub_objects[0]['material']}, Object 2: {phys_args.sub_objects[1]['material']}")
    
    gts_per_object = {
        1: gts1,
        2: gts2
    }
    
    # Save per-object particles separately for verification
    if len(vol_combined) > 0:
        mpm_path = os.path.join(dataset.model_path, "mpm")
        os.makedirs(mpm_path, exist_ok=True)
        
        # Save particles for Object 1
        obj1_mask = object_labels == 1
        if obj1_mask.any():
            obj1_particles = vol_combined[obj1_mask]
            # Create yellow/gold color for object 1
            obj1_colors = torch.full((len(obj1_particles), 3), fill_value=255, device=obj1_particles.device, dtype=torch.uint8)
            obj1_colors[:, 0] = 255  # R
            obj1_colors[:, 1] = 200  # G  
            obj1_colors[:, 2] = 40   # B (yellow/gold)
            obj1_colors_np = obj1_colors.cpu().numpy()
            
            write_particles(obj1_particles, 0, dataset.model_path, 'object1_particles', vertex_colors=obj1_colors_np)
            print(f"Saved Object 1 particles ({len(obj1_particles)} points) to {mpm_path}/object1_particles_0.ply")
        
        # Save particles for Object 2
        obj2_mask = object_labels == 2
        if obj2_mask.any():
            obj2_particles = vol_combined[obj2_mask]
            # Create green color for object 2
            obj2_colors = torch.full((len(obj2_particles), 3), fill_value=255, device=obj2_particles.device, dtype=torch.uint8)
            obj2_colors[:, 0] = 1    # R
            obj2_colors[:, 1] = 68   # G
            obj2_colors[:, 2] = 33   # B (green)
            obj2_colors_np = obj2_colors.cpu().numpy()
            
            write_particles(obj2_particles, 0, dataset.model_path, 'object2_particles', vertex_colors=obj2_colors_np)
            print(f"Saved Object 2 particles ({len(obj2_particles)} points) to {mpm_path}/object2_particles_0.ply")
        
        # Print statistics
        print(f"\nParticle Statistics:")
        print(f"  Total particles: {len(object_labels)}")
        print(f"  Object 1 particles: {obj1_mask.sum().item()} ({obj1_mask.sum().item()/len(object_labels)*100:.1f}%)")
        print(f"  Object 2 particles: {obj2_mask.sum().item()} ({obj2_mask.sum().item()/len(object_labels)*100:.1f}%)")
    
    return gts_per_object, gts_combined, vol_combined, vol_densities_combined, grid_size, volume_surface_combined, cam_info, vol1, vol2






# def prepare_gt_with_pcds(dataset: ModelParams,
#                iteration: int,
#                pipeline: PipelineParams,
#                phys_args,
#                gaussians=None,
#                scene=None,
#                object_id=None,
#                gt_pcs_per_object=None):
#     """
#     使用 GT 点云 (gt_pcs_per_object) 为单个 object 构建:
#       - gts: 每一帧的 surface 粒子列表
#       - vol: 第 0 帧的体内粒子
#       - vol_densities: 对应 vol 的密度
#       - grid_size tensor
#       - vol_surface: vol 里哪些 index 是“表面”
#       - cam_info: 相机以及 alpha mask 信息（通过 overwrite_alphas_with_pcds 填充）
#     """

#     gts = []
   
    
#     bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
#     background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
#     views = scene.getTrainCameras(scale=image_scale)
#     fids = torch.unique(torch.stack([view.fid for view in views]))
#     xyz_canonical = gaussians.get_xyz.detach()
#     # opacitiy = gaussians.get_opacity.squeeze()
#     grid_size = phys_args.density_grid_size
#     density_min_th = phys_args.density_min_th
#     density_max_th = phys_args.density_max_th
#     num_iter = 4 if phys_args.random_sample else 5
#     filling_grid_size = grid_size / 2 ** 5
#     opacity_threshold = phys_args.opacity_threshold
#     with torch.no_grad():
#         for idx, fid in enumerate(tqdm(fids, desc="Filling progress")):
#             if getattr(phys_args, "n_frames", None) and idx >= phys_args.n_frames:
#                 break
            
            
#             xyzt = gt_pcs_per_object[object_id][idx]
#             d_xyz = xyzt - xyz_canonical
#             bbox_mins = xyzt.min(0)[0] - grid_size
#             bbox_maxs = xyzt.max(0)[0] + grid_size
#             bbox_bounds = bbox_maxs - bbox_mins
#             volume_size = torch.round(bbox_bounds / filling_grid_size).to(torch.int64) + 1
#             grid_ids = [torch.arange(size) for size in volume_size]
#             grid_coords = torch.stack(torch.meshgrid(*grid_ids, indexing='ij'), dim=-1).reshape(-1, 3) * filling_grid_size
#             grid_coords = grid_coords.to(xyzt)
#             init_inner_points = grid_coords + bbox_mins.reshape(1, 3)
#             curr_views = [view for view in views if view.fid == fid]  
#             for viewpoint_cam in tqdm(curr_views, desc="Rendering progress"):
#                 results = render(viewpoint_cam, gaussians, pipeline, background, d_xyz, 0.0, 0.0, False)
#                 depth = results["depth"][0]
#                 render_mask = torch.logical_and(results["render"].sum(0) != 0, 
#                                                 viewpoint_cam.original_image.sum(0) != 0)
#                 pix_w, pix_h, pix_d = viewpoint_cam.pw2pix(init_inner_points)
#                 # remove points that are outside the image space
#                 in_mask = viewpoint_cam.is_in_view(pix_w, pix_h)
#                 init_inner_points = init_inner_points[in_mask]
#                 pix_w, pix_h, pix_d = pix_w[in_mask], pix_h[in_mask], pix_d[in_mask]
#                 # remove points that the projected pixels are outside the object mask
#                 pix_mask = render_mask[pix_h, pix_w]
#                 init_inner_points = init_inner_points[pix_mask]
#                 pix_w, pix_h, pix_d = pix_w[pix_mask], pix_h[pix_mask], pix_d[pix_mask]
#                 # remove points whose depth values are smaller than those from depth map
#                 render_pix_d = depth[pix_h, pix_w]
#                 depth_mask = render_pix_d < pix_d
#                 init_inner_points = init_inner_points[depth_mask]
#                 # remove outliers in xyzt
#                 render_mask = results["render"].sum(0) > 1 / 255
#                 pix_w, pix_h, pix_d = viewpoint_cam.pw2pix(xyzt)
#                 in_mask = viewpoint_cam.is_in_view(pix_w, pix_h)
#                 pix_w, pix_h, pix_d = pix_w[in_mask], pix_h[in_mask], pix_d[in_mask]
#                 xyzt = xyzt[in_mask]
#                 pix_mask = render_mask[pix_h, pix_w]
#                 xyzt = xyzt[pix_mask]
#             curr_grid_size = grid_size / 2
#             volume_size = torch.round(bbox_bounds / curr_grid_size).to(torch.int64) + 1
#             bbox_maxs = bbox_mins + (volume_size - 1) * curr_grid_size
#             bbox_bounds = bbox_maxs - bbox_mins
#             density_volume = torch.zeros(volume_size.cpu().numpy().tolist()).to(init_inner_points)
#             ids = torch.round((init_inner_points - bbox_mins.reshape(1, 3)) / curr_grid_size).to(torch.int64)
#             density_volume[ids.T[0], ids.T[1], ids.T[2]] = 1.0
#             ids = torch.round((xyzt - bbox_mins.reshape(1, 3)) / curr_grid_size).to(torch.int64)
#             density_volume[ids.T[0], ids.T[1], ids.T[2]] = 1.0
#             weight = torch.ones((1, 1, 3, 3, 3)).to(xyzt)
#             weight = weight / weight.sum()
#             for i in range(2, num_iter):
#                 curr_grid_size = grid_size / 2 ** i
#                 volume_size = torch.round(bbox_bounds / curr_grid_size).to(torch.int64) + 1
#                 bbox_maxs = bbox_mins + (volume_size - 1) * curr_grid_size
#                 grid_xyz = torch.stack(torch.meshgrid(
#                     torch.linspace(0, volume_size[0]-1, volume_size[0]),
#                     torch.linspace(0, volume_size[1]-1, volume_size[1]),
#                     torch.linspace(0, volume_size[2]-1, volume_size[2]),
#                 ), dim=-1).to(bbox_mins) * curr_grid_size + bbox_mins[None, None, None]
#                 ids_norm = (grid_xyz - bbox_mins[None, None, None]) / bbox_bounds[None, None, None] * 2 - 1
#                 ids_norm = ids_norm[None].flip((-1,))
#                 density_volume = torch.nn.functional.grid_sample(density_volume[None, None], ids_norm, mode='bilinear', align_corners=True)
#                 density_volume = torch.nn.functional.conv3d(density_volume, weight=weight, padding='same')[0, 0]
#                 density_volume[density_volume < 0.5] = 0.0
#                 ids = torch.round((init_inner_points - bbox_mins.reshape(1, 3)) / curr_grid_size).to(torch.int64)
#                 density_volume[ids.T[0], ids.T[1], ids.T[2]] = 1.0
#                 ids = torch.round((xyzt - bbox_mins.reshape(1, 3)) / curr_grid_size).to(torch.int64)
#                 density_volume[ids.T[0], ids.T[1], ids.T[2]] = 1.0
#                 bbox_bounds = bbox_maxs - bbox_mins
#             for i in range(20):
#                 density_volume = torch.nn.functional.conv3d(density_volume[None, None], weight=weight, padding='same')[0, 0]
#                 density_volume[density_volume < 0.5] = 0.0
#                 ids = torch.round((init_inner_points - bbox_mins.reshape(1, 3)) / curr_grid_size).to(torch.int64)
#                 density_volume[ids.T[0], ids.T[1], ids.T[2]] = 1.0
#                 ids = torch.round((xyzt - bbox_mins.reshape(1, 3)) / curr_grid_size).to(torch.int64)
#                 density_volume[ids.T[0], ids.T[1], ids.T[2]] = 1.0
#             if phys_args.random_sample:
#                 density_volume = torch.nn.functional.conv3d(density_volume[None, None], weight=weight, padding='same')[0, 0]
#                 half_grid_xyz = torch.stack(torch.meshgrid(
#                     torch.linspace(0, volume_size[0]-0.5, 2 * (volume_size[0]-1)),
#                     torch.linspace(0, volume_size[1]-0.5, 2 * (volume_size[1]-1)),
#                     torch.linspace(0, volume_size[2]-0.5, 2 * (volume_size[2]-1)),
#                 ), -1).to(bbox_mins) * curr_grid_size + bbox_mins[None, None, None]
#                 ids_norm = (half_grid_xyz - bbox_mins[None, None, None]) / bbox_bounds[None, None, None] * 2 - 1
#                 ids_norm = ids_norm[None].flip((-1,))
#                 density_half_grid_xyz = torch.nn.functional.grid_sample(density_volume[None, None], ids_norm, mode='bilinear', align_corners=True)[0, 0]
#                 half_grid_xyz = half_grid_xyz[density_half_grid_xyz > 0.5]
#                 delta = (torch.rand_like(half_grid_xyz) * curr_grid_size * 0.5).to(xyzt)
#                 particles = half_grid_xyz + delta
#                 ids_norm = (particles[None,None] - bbox_mins[None, None, None]) / bbox_bounds[None, None, None] * 2 - 1
#                 ids_norm = ids_norm[None].flip((-1,))
#                 density_particles = torch.nn.functional.grid_sample(density_volume[None, None], ids_norm, mode='bilinear', align_corners=True)[0, 0, 0, 0]
#                 sampled_pts = particles[density_particles > density_min_th]
#                 surface_pts = particles[(density_particles > density_min_th)*(density_particles < density_max_th)]
#                 gts.append(surface_pts)
#                 curr_grid_size = curr_grid_size / 2
#                 if fid == 0.0:
#                     vol = sampled_pts
#                     vol_densities = density_particles[density_particles > density_min_th]
#                     vol_surface_mask = density_particles[density_particles > density_min_th] < density_max_th
#                     vol_surface = torch.arange(vol_surface_mask.shape[0]).to(vol.device).to(torch.int64)[vol_surface_mask]
#             else:
#                 density_volume = torch.nn.functional.conv3d(density_volume[None, None], weight=weight, padding='same')[0, 0]
#                 internal_mask = density_volume >= density_min_th
#                 sampled_pts = torch.stack(torch.where(internal_mask), dim=-1) * curr_grid_size + bbox_mins.reshape(1, 3)
#                 density_volume_smoothed = torch.nn.functional.conv3d(density_volume[None, None], weight=weight, padding='same')[0, 0]
#                 surface_mask = (density_volume_smoothed > 0) * (density_volume_smoothed < density_max_th) * internal_mask
#                 surface_pts = torch.stack(torch.where(surface_mask == 1), dim=-1) * curr_grid_size + bbox_mins.reshape(1, 3)
#                 gts.append(surface_pts)
#                 if fid == 0.0:
#                     vol = sampled_pts
#                     vol_densities = density_volume[internal_mask]
#                     vol_surface_mask = surface_mask[internal_mask]
#                     vol_surface = torch.arange(vol_surface_mask.shape[0]).to(vol.device).to(torch.int64)[vol_surface_mask]

#     # ---------- 5. 用你写好的 overwrite_alphas_with_pcds 填充相机的 gt_alpha_mask ----------
#     train_cams, test_cams, cameras_extent = scene.overwrite_alphas_with_pcds(
#         pipeline, dataset, gt_pcs_per_object, object_id=object_id
#     )
#     cam_info = {
#         "train_cams": train_cams,
#         "test_cams": test_cams,
#         "cameras_extent": cameras_extent,
#     }
    
#     # print(f"[prepare_gt] fid={float(fid):.1f}")
#     # print("  bbox_bounds:", bbox_bounds.detach().cpu().numpy())
#     # print("  curr_grid_size:", float(curr_grid_size))
#     # print("  volume_size:", volume_size.cpu().numpy().tolist())
#     # num_voxels = int(volume_size[0]*volume_size[1]*volume_size[2])
#     # print("  density_volume voxels:", num_voxels,
#     #   "  ~ mem:", num_voxels*4/1024/1024, "MB (float32)")

#     # 返回格式保持不变
#     return gts, vol, vol_densities, torch.tensor([curr_grid_size]), vol_surface, cam_info


def prepare_gt_with_pcds(dataset: ModelParams,
               iteration: int,
               pipeline: PipelineParams,
               phys_args,
               gaussians=None,
               scene=None,
               object_id=None,
               gt_pcs_per_object=None):
    """
    使用 GT 点云 (gt_pcs_per_object) 为单个 object 构建:
      - gts: 每一帧的 surface 粒子列表
      - vol: 第 0 帧的体内粒子
      - vol_densities: 对应 vol 的密度
      - grid_size tensor
      - vol_surface: vol 里哪些 index 是“表面”
      - cam_info: 相机以及 alpha mask 信息（通过 overwrite_alphas_with_pcds 填充）
    """
    device = "cuda"

    gts = []
    vol = None
    vol_densities = None
    vol_surface = None
    curr_grid_size = phys_args.voxel_size  # 或者用 bbox_extent 的一个比例

    views = scene.getTrainCameras(scale=image_scale)
    fids = torch.unique(torch.stack([view.fid for view in views]))

    for idx, fid in enumerate(fids):
        xyzt = gt_pcs_per_object[object_id][idx].to(device)  # [N_t, 3]

        # 直接把当前帧 GT 点云当作 surface
        gts.append(xyzt)

        if idx == 0:
            # 第 0 帧：直接把 GT 当作 vol
            vol = xyzt.clone()
            # 给一个常数密度，或者根据别的规则
            vol_densities = torch.ones(vol.shape[0], device=vol.device)
            # 把所有点都当 surface
            vol_surface = torch.arange(vol.shape[0], device=vol.device, dtype=torch.long)

    # 用 GT pcds 给相机写 alpha mask
    train_cams, test_cams, cameras_extent = scene.overwrite_alphas_with_pcds(
        pipeline, dataset, gt_pcs_per_object, object_id=object_id
    )
    cam_info = {
        "train_cams": train_cams,
        "test_cams": test_cams,
        "cameras_extent": cameras_extent,
    }

    return gts, vol, vol_densities, torch.tensor([curr_grid_size], device=vol.device), vol_surface, cam_info






def forward(estimator: Estimator, img_backward=True):
    dt = estimator.simulator.dt_ori[None]
    pos_sequence = []  # Store positions for video saving
    while True:
        pos_sequence.clear()  # Clear previous attempts
        for idx in range(estimator.max_f):
            if idx == 0:
                estimator.initialize()
                estimator.simulator.set_dt(dt)
            x = estimator.forward(idx, img_backward)
            # Only store positions if we're going to succeed (otherwise we'll restart)
            if estimator.succeed() and isinstance(x, torch.Tensor):
                pos_sequence.append(x.detach().clone())
            else:
                pos_sequence.append(None)
        if not estimator.succeed():
            dt /= 2
            print('cfl condition dissatisfy, shrink dt {}, step cnt {}'.format(dt, estimator.simulator.n_substeps[None] * 2))
        else:
            break
    return pos_sequence

def save_training_debug(estimator: Estimator, iteration, save_path, pos_sequence, stage_name, max_frames=None):
    """
    Save both video and point clouds for debugging during training.
    pos_sequence: list of particle positions from the current forward pass
    stage_name: 'velocity' or 'physical_params' to identify the training stage
    """
    if max_frames is None:
        max_frames = min(len(pos_sequence), estimator.max_f)
    
    # Create debug folder structure
    debug_path = os.path.join(save_path, "debug")
    video_path = os.path.join(debug_path, "videos")
    points_path = os.path.join(debug_path, "point_clouds")
    sil_path = os.path.join(debug_path, "sil")
    os.makedirs(video_path, exist_ok=True)
    os.makedirs(points_path, exist_ok=True)
    os.makedirs(sil_path, exist_ok=True)
    
    rendered_frames = []
    gt_frames = []
    pred_positions = []
    gt_positions = []
    sil_frames = []  # Collect silhouette comparisons for video
    
    # Collect frames and positions
    for f in range(max_frames):
        if f >= len(pos_sequence) or pos_sequence[f] is None:
            break
            
        particle_pos_tensor = pos_sequence[f]
        
        # Store predicted positions (surface particles)
        # Get surface particle indices for this frame
        if hasattr(estimator, 'sim_surface_index') and hasattr(estimator, 'sim_surface_cnt'):
            surface_cnt = int(estimator.sim_surface_cnt[f])
            if surface_cnt > 0:
                # Taichi fields don't support slicing, need to copy one by one
                surface_indices = []
                for i in range(surface_cnt):
                    surface_indices.append(int(estimator.sim_surface_index[f, i]))
                surface_indices = np.array(surface_indices)
                pred_surface_pos = particle_pos_tensor[surface_indices].cpu().numpy()
            else:
                pred_surface_pos = particle_pos_tensor.cpu().numpy()
        else:
            pred_surface_pos = particle_pos_tensor.cpu().numpy()
        pred_positions.append(pred_surface_pos)
        
        # Store GT positions
        if f < estimator.max_f and hasattr(estimator, 'gt') and hasattr(estimator, 'num_particles_surface'):
            # Get the number of GT particles for this frame
            gt_count = int(estimator.num_particles_surface[f])
            if gt_count > 0:
                # Copy GT positions from Taichi field
                gt_pos = []
                for i in range(gt_count):
                    pos = []
                    for j in range(3):  # x, y, z
                        pos.append(float(estimator.gt[f, i][j]))
                    gt_pos.append(pos)
                gt_pos = np.array(gt_pos)
            else:
                gt_pos = np.array([]).reshape(0, 3)
            gt_positions.append(gt_pos)
        else:
            gt_positions.append(np.array([]).reshape(0, 3))
        
        # Render for video - collect 8 views (0-7) for 4x4 grid
        if hasattr(estimator, 'views') and estimator.views and f < len(estimator.views):
            gaussians = estimator.scene.gaussians if hasattr(estimator, 'scene') and estimator.scene else None
            if gaussians is not None:
                views = estimator.views[f]
                background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
                d_xyz = particle_pos_tensor - gaussians.get_xyz
                
                # Collect renders from views 0-7 (or as many as available)
                frame_renders = []
                frame_gts = []
                num_views_needed = 8
                
                # Sort views by uid to ensure consistent ordering
                sorted_views = sorted(views, key=lambda v: v.uid)
                
                for view_idx in range(min(num_views_needed, len(sorted_views))):
                    view = sorted_views[view_idx]
                    with torch.no_grad():
                        # Isotropic Gaussians (num_attribute=4)
                        results = render(view, gaussians, estimator.pipeline, background, d_xyz, 0.0, 0.0, False)
                        # Anisotropic Gaussians (num_attribute=10) - fallback
                        # num_gaussians = gaussians._xyz.shape[0]
                        # d_rotation = torch.zeros((num_gaussians, 4), device=gaussians._xyz.device)
                        # d_scaling = torch.zeros((num_gaussians, 3), device=gaussians._xyz.device)
                        # results = render(view, gaussians, estimator.pipeline, background, d_xyz, d_rotation, d_scaling, False)
                        rendered_image = results["render"]
                        gt_image = view.original_image.cuda()
                        rendered_np = (rendered_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        gt_np = (gt_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        frame_renders.append(rendered_np)
                        frame_gts.append(gt_np)
                
                # Pad with black frames if fewer than 8 views
                if len(frame_renders) > 0:
                    h, w = frame_renders[0].shape[:2]
                    while len(frame_renders) < num_views_needed:
                        frame_renders.append(np.zeros((h, w, 3), dtype=np.uint8))
                        frame_gts.append(np.zeros((h, w, 3), dtype=np.uint8))
                    
                    rendered_frames.append(frame_renders)
                    gt_frames.append(frame_gts)

                # Save silhouette comparisons for this frame (matching render_forward exactly)
                if hasattr(gaussians, 'get_object_probs') and len(views) > 0:
                    # Get object assignments using argmax (same as render_forward)
                    object_probs = gaussians.get_object_probs  # [N, 3]
                    object_assignments = torch.argmax(object_probs, dim=1)  # [N] with values 0,1,2
                    obj1_mask = (object_assignments == 1).float()  # Mask for object 1
                    obj2_mask = (object_assignments == 2).float()  # Mask for object 2

                    # Store original opacities (raw _opacity values)
                    original_opacity = gaussians._opacity.detach().clone()

                    # Process first view for silhouette comparison
                    view = sorted_views[0] if len(sorted_views) > 0 else views[0]
                    if hasattr(view, 'gt_alpha_mask_obj1') and hasattr(view, 'gt_alpha_mask_obj2'):
                        # Render object 1 only (matching render_forward)
                        gaussians._opacity.data = original_opacity * obj1_mask.unsqueeze(1) - 1e2 * (1 - obj1_mask.unsqueeze(1))
                        with torch.no_grad():
                            # Create zero deformation tensors for anisotropic Gaussians
                            # Isotropic Gaussians (num_attribute=4)
                            results_obj1 = render(view, gaussians, estimator.pipeline, background, d_xyz, 0.0, 0.0, False)
                            # Anisotropic Gaussians (num_attribute=10) - fallback
                            # num_gaussians = gaussians._xyz.shape[0]
                            # d_rotation = torch.zeros((num_gaussians, 4), device=gaussians._xyz.device)
                            # d_scaling = torch.zeros((num_gaussians, 3), device=gaussians._xyz.device)
                            # results_obj1 = render(view, gaussians, estimator.pipeline, background, d_xyz, d_rotation, d_scaling, False)
                            alpha_obj1 = results_obj1["alpha"].cpu().numpy()[0]  # [H, W]

                        # Render object 2 only (matching render_forward)
                        gaussians._opacity.data = original_opacity * obj2_mask.unsqueeze(1) - 1e2 * (1 - obj2_mask.unsqueeze(1))
                        with torch.no_grad():
                            # Isotropic Gaussians (num_attribute=4)
                            results_obj2 = render(view, gaussians, estimator.pipeline, background, d_xyz, 0.0, 0.0, False)
                            # Anisotropic Gaussians (num_attribute=10) - fallback
                            # results_obj2 = render(view, gaussians, estimator.pipeline, background, d_xyz, d_rotation, d_scaling, False)
                            alpha_obj2 = results_obj2["alpha"].cpu().numpy()[0]  # [H, W]

                        # Get GT silhouettes (same as used in render_forward)
                        gt_alpha_obj1 = view.gt_alpha_mask_obj1.cpu().numpy()[0]  # [H, W]
                        gt_alpha_obj2 = view.gt_alpha_mask_obj2.cpu().numpy()[0]  # [H, W]

                        # Convert to uint8
                        alpha_obj1 = (alpha_obj1 * 255).astype(np.uint8)
                        alpha_obj2 = (alpha_obj2 * 255).astype(np.uint8)
                        gt_alpha_obj1 = (gt_alpha_obj1 * 255).astype(np.uint8)
                        gt_alpha_obj2 = (gt_alpha_obj2 * 255).astype(np.uint8)

                        # Create side-by-side comparison
                        h, w = alpha_obj1.shape
                        comparison_obj1 = np.zeros((h, w * 2), dtype=np.uint8)
                        comparison_obj1[:, :w] = gt_alpha_obj1  # GT on left
                        comparison_obj1[:, w:] = alpha_obj1     # Rendered on right

                        comparison_obj2 = np.zeros((h, w * 2), dtype=np.uint8)
                        comparison_obj2[:, :w] = gt_alpha_obj2  # GT on left
                        comparison_obj2[:, w:] = alpha_obj2     # Rendered on right

                        # Stack vertically: obj1 on top, obj2 on bottom
                        full_comparison = np.vstack([comparison_obj1, comparison_obj2])

                        # Add labels
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        thickness = 2
                        cv2.putText(full_comparison, "GT", (10, 30), font, font_scale, 255, thickness)
                        cv2.putText(full_comparison, "Rendered", (w + 10, 30), font, font_scale, 255, thickness)
                        cv2.putText(full_comparison, "Obj1", (10, h - 10), font, font_scale, 255, thickness)
                        cv2.putText(full_comparison, "Obj2", (10, h * 2 - 10), font, font_scale, 255, thickness)

                        # Collect for video (convert to RGB for video encoding)
                        sil_frames.append(cv2.cvtColor(full_comparison, cv2.COLOR_GRAY2RGB))

                        # Restore original opacities
                        gaussians._opacity.data = original_opacity
    
    # Save point clouds
    from plyfile import PlyData, PlyElement
    
    # Save individual frames
    for f in range(len(pred_positions)):
        if f < len(gt_positions):
            # Save predicted points
            pred_pos = pred_positions[f]
            vertex = np.array([(x, y, z) for x, y, z in pred_pos],
                            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            el = PlyElement.describe(vertex, 'vertex')
            ply_data = PlyData([el])
            ply_path = os.path.join(points_path, f"{stage_name}_iter_{iteration:04d}_frame_{f:03d}_pred.ply")
            ply_data.write(ply_path)
            
            # Save GT points
            gt_pos = gt_positions[f]
            vertex = np.array([(x, y, z) for x, y, z in gt_pos],
                            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            el = PlyElement.describe(vertex, 'vertex')
            ply_data = PlyData([el])
            ply_path = os.path.join(points_path, f"{stage_name}_iter_{iteration:04d}_frame_{f:03d}_gt.ply")
            ply_data.write(ply_path)
    
    # Save combined trajectory (all frames with time as 4th dimension)
    if pred_positions and gt_positions:
        all_pred = []
        all_gt = []
        for f in range(min(len(pred_positions), len(gt_positions))):
            pred_with_time = np.concatenate([pred_positions[f], np.full((pred_positions[f].shape[0], 1), f)], axis=1)
            gt_with_time = np.concatenate([gt_positions[f], np.full((gt_positions[f].shape[0], 1), f)], axis=1)
            all_pred.append(pred_with_time)
            all_gt.append(gt_with_time)
        
        all_pred = np.vstack(all_pred)
        all_gt = np.vstack(all_gt)
        
        # Save combined trajectories
        for data, name in [(all_gt, "gt"), (all_pred, "pred")]:
            vertex = np.array([(x, y, z, t) for x, y, z, t in data],
                            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('time', 'f4')])
            el = PlyElement.describe(vertex, 'vertex')
            ply_data = PlyData([el])
            ply_path = os.path.join(points_path, f"{stage_name}_iter_{iteration:04d}_trajectory_{name}.ply")
            ply_data.write(ply_path)
    
    # Save videos as 4x4 grid
    if rendered_frames:
        comparison_frames = []
        
        for frame_idx in range(len(rendered_frames)):
            frame_renders = rendered_frames[frame_idx]  # List of 8 rendered images
            frame_gts = gt_frames[frame_idx]  # List of 8 GT images
            
            # Create 4x4 grid: each row has GT, Render, GT, Render
            # Total 4 rows, each with 2 GT-Render pairs
            rows = []
            for row_idx in range(4):
                view_idx = row_idx * 2  # Views 0-1, 2-3, 4-5, 6-7
                
                if view_idx < len(frame_gts):
                    # First pair: GT | Render
                    gt1 = frame_gts[view_idx]
                    rend1 = frame_renders[view_idx]
                    
                    # Second pair: GT | Render  
                    if view_idx + 1 < len(frame_gts):
                        gt2 = frame_gts[view_idx + 1]
                        rend2 = frame_renders[view_idx + 1]
                    else:
                        # Use black frames if not enough views
                        h, w = gt1.shape[:2]
                        gt2 = np.zeros((h, w, 3), dtype=np.uint8)
                        rend2 = np.zeros((h, w, 3), dtype=np.uint8)
                    
                    # Concatenate: GT1 | Rend1 | GT2 | Rend2
                    row = np.concatenate([gt1, rend1, gt2, rend2], axis=1)
                    rows.append(row)
            
            # Stack all rows vertically to create 4x4 grid
            grid = np.concatenate(rows, axis=0)
            comparison_frames.append(grid)
        
        comparison_path = os.path.join(video_path, f"{stage_name}_iter_{iteration:04d}_grid_comparison.mp4")
        imageio.mimwrite(comparison_path, comparison_frames, fps=15, quality=8)

        print(f"Saved 4x4 grid video to {comparison_path}")
        print(f"Saved point clouds to {points_path}")

    # Save silhouette comparison video
    if sil_frames:
        sil_video_path = os.path.join(sil_path, f"{stage_name}_iter_{iteration:04d}_silhouette_comparison.mp4")
        imageio.mimwrite(sil_video_path, sil_frames, fps=15, quality=8)
        print(f"Saved silhouette comparison video to {sil_video_path}")

def save_trajectory(estimator, dataset, config_id, iteration=None, prediction_frames=30, is_novel=False):
    """Save trajectory at current state of estimator"""
    if is_novel:
        pred_traj_dir = os.path.join(dataset.model_path, 'pred_traj_novel_material')
    else:
        pred_traj_dir = os.path.join(dataset.model_path, 'pred_traj')
    os.makedirs(pred_traj_dir, exist_ok=True)

    # Save original max_f
    old_max_f = estimator.max_f
    estimator.max_f = prediction_frames

    # Reinitialize with new frame count
    estimator.initialize()

    trajectory = []
    # Generate trajectory without computing losses
    for f in range(prediction_frames):
        with torch.no_grad():
            # Run simulation step
            if f > 0:
                estimator.simulator.advance(f - 1)
            # Get particle positions directly
            particle_pos = np.zeros((estimator.num_particles[None], 3), dtype=np.float32)
            estimator.simulator.get_x(f, particle_pos)
            trajectory.append(particle_pos.copy())

    # Restore original max_f
    estimator.max_f = old_max_f

    # ALWAYS overwrite the same file (like pred.json)
    trajectory_path = os.path.join(pred_traj_dir, f'{config_id}-pred_traj.npy')

    # Save current iteration info in a separate metadata file if needed
    if iteration is not None:
        metadata_path = os.path.join(pred_traj_dir, f'{config_id}-pred_traj_metadata.json')
        metadata = {
            'current_iteration': iteration,
            'prediction_frames': prediction_frames,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    np.save(trajectory_path, trajectory)
    return trajectory_path

def save_loss_plot(losses, save_path, stage_name):
    """
    Save loss curve plot for the given training stage.
    
    Args:
        losses: List of loss values
        save_path: Path to save the plot
        stage_name: 'velocity' or 'physical_params' to identify the training stage
    """
    if len(losses) == 0:
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'{stage_name.replace("_", " ").title()} Training Loss Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add min loss annotation
    min_loss = min(losses)
    min_idx = losses.index(min_loss)
    plt.annotate(f'Min: {min_loss:.6f} at iter {min_idx}',
                xy=(min_idx, min_loss), xytext=(min_idx + len(losses)*0.1, min_loss + (max(losses)-min(losses))*0.1),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    # Save the plot
    plot_path = os.path.join(save_path, f'{stage_name}_loss_curve.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved loss curve to {plot_path}")

def backward(estimator: Estimator):
    if hasattr(estimator, 'per_object_sil_losses_accum'):
        obj1_sil_loss = estimator.per_object_sil_losses_accum.get(1, 0.0)
        obj2_sil_loss = estimator.per_object_sil_losses_accum.get(2, 0.0)

        # Check if we have per-object geometry losses
        if hasattr(estimator, 'get_per_object_losses'):
            per_obj_geo = estimator.get_per_object_losses()
            print('Geometry loss {:.6f} (Obj1: {:.6f}, Obj2: {:.6f}), Sil loss - Obj1: {:.6f}, Obj2: {:.6f}, step {}'.format(
                estimator.loss[None], per_obj_geo[1], per_obj_geo[2], obj1_sil_loss, obj2_sil_loss, estimator.simulator.n_substeps[None]))
        else:
            print('Geometry loss {:.6f}, Sil loss - Obj1: {:.6f}, Obj2: {:.6f}, step {}'.format(
                estimator.loss[None], obj1_sil_loss, obj2_sil_loss, estimator.simulator.n_substeps[None]))
    else:
        print('Geometry loss {}, image loss {}, step {}'.format(
            estimator.loss[None], estimator.image_loss, estimator.simulator.n_substeps[None]))
    max_f = estimator.max_f
    pbar = trange(max_f)
    pbar.set_description(f"[Backward]")
    
    estimator.loss.grad[None] = 1
    estimator.clear_grads()
    
    for ri in pbar:
        i = max_f - 1 - ri
        if i > 0:
            estimator.backward(i)
        else:
            pos_grad, velocity_grad, mu_grad, lam_grad, \
            yield_stress_grad, viscosity_grad, \
            friction_alpha_grad, cohesion_grad, rho_grad = estimator.backward(i)
            estimator.init_velocities.backward(retain_graph=True, gradient=velocity_grad)
            estimator.init_rhos.backward(retain_graph=True, gradient=rho_grad)
            estimator.init_pos.backward(retain_graph=True, gradient=pos_grad)
            estimator.init_mu.backward(retain_graph=True, gradient=mu_grad)
            estimator.init_lam.backward(retain_graph=True, gradient=lam_grad)

            if any('yield_stress' in obj_params for obj_params in estimator.training_params.values()):
                estimator.init_yield_stress.backward(retain_graph=True, gradient=yield_stress_grad)
            if any('plastic_viscosity' in obj_params for obj_params in estimator.training_params.values()):
                estimator.init_plastic_viscosity.backward(retain_graph=True, gradient=viscosity_grad)
            if any('friction_alpha' in obj_params for obj_params in estimator.training_params.values()):
                estimator.init_friction_alpha.backward(gradient=friction_alpha_grad)
            # Note: cohesion is not trainable, so no backward needed

    # Print gradient norms for parameters being optimized
    if estimator.stage[None] == Estimator.velocity_stage and hasattr(estimator, 'object_velocities'):
        grad_info = []
        for obj_id, vel_param in estimator.object_velocities.items():
            if vel_param.grad is not None:
                grad_norm = torch.norm(vel_param.grad).item()
                grad_info.append(f"Obj{obj_id}: {grad_norm:.6f}")
            else:
                grad_info.append(f"Obj{obj_id}: None")
        print(f"Velocity gradient norms - {', '.join(grad_info)}")

    elif estimator.stage[None] == Estimator.physical_params_stage and hasattr(estimator, 'training_params'):
        grad_info = []
        for obj_id, params_dict in estimator.training_params.items():
            obj_grads = []
            for param_name, param in params_dict.items():
                if hasattr(param, 'grad') and param.grad is not None:
                    grad_norm = torch.norm(param.grad).item()
                    obj_grads.append(f"{param_name}: {grad_norm:.6f}")
            if obj_grads:
                grad_info.append(f"Obj{obj_id[3:]}: [{', '.join(obj_grads)}]")
        if grad_info:
            print(f"Parameter gradient norms - {', '.join(grad_info)}")

def train(estimator: Estimator, phys_args, max_f=None, dataset=None, use_wandb=False, config_id=None):
    losses = []
    estimated_params = []
    if estimator.stage[None] == Estimator.velocity_stage:
        iter_cnt = phys_args.vel_iter_cnt
        stage_name = "velocity"
    elif estimator.stage[None] == Estimator.physical_params_stage:
        iter_cnt = phys_args.iter_cnt
        stage_name = "physical_params"

    # Initialize wandb if requested and available
    wandb_enabled = use_wandb and WANDB_AVAILABLE and dataset is not None
    if wandb_enabled:
        print(f"Initializing wandb for {stage_name} stage...")
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            wandb.init(
                project="GIC_MO",
                name=f"{os.path.basename(dataset.model_path)}_{stage_name}_{timestamp}",
                reinit=True,  # Allow multiple runs in same session
                config={
                    "stage": stage_name,
                    "iterations": iter_cnt,
                    "model_path": dataset.model_path,
                    "voxel_size": phys_args.voxel_size if hasattr(phys_args, 'voxel_size') else None,
                    "density_grid_size": phys_args.density_grid_size if hasattr(phys_args, 'density_grid_size') else None
                }
            )
            print(f"Wandb initialized successfully. View at: {wandb.run.url}")
        except Exception as e:
            print(f"Failed to initialize wandb: {e}")
            wandb_enabled = False

    if max_f is not None:
        estimator.max_f = max_f
    
    for stage, train_param in enumerate(zip([max_f], [iter_cnt])):
        max_f, iter_cnt = train_param
        if max_f is not None:
            estimator.max_f = max_f
        for i in range(iter_cnt):
            # 1. record current params
            d = {}
            param_groups = estimator.get_optimizer().param_groups
            report_msg = ''
            report_msg += f'iter {i}'
            # Report per-object velocities
            for obj_id, vel_param in estimator.object_velocities.items():
                report_msg += f'\nObject {obj_id} velocity: {vel_param.cpu().detach().tolist()}'
            for params in param_groups:
                name = params['name']
                p = params['params'][0].detach().cpu()

                display_name = name
                if name.startswith('obj'):
                    # Extract base parameter name (e.g., "obj1_Youngs_modulus" -> "Youngs modulus")
                    parts = name.split('_', 1)  # ['obj1', 'Youngs_modulus' or 'Poisson_ratio']
                    if len(parts) >= 2:
                        obj_id = parts[0][3:]
                        base_param_name = parts[1].replace('_', ' ')
                        
                        # Apply proper transformations based on parameter name
                        if base_param_name == 'Poisson ratio':
                            display_name = f"Object {obj_id} - Poisson's ratio"
                            nu_values = estimator.get_nu()
                            obj_key = str(int(obj_id))
                            if obj_key in nu_values:
                                p_value = nu_values[obj_key].mean().item() if hasattr(nu_values[obj_key], 'mean') else nu_values[obj_key].item()
                            else:
                                p_value = p.mean().item()  # Fallback
                        elif base_param_name == 'Youngs modulus':
                            display_name = f"Object {obj_id} - Young's modulus"
                            p_value = (10**p).mean().item()  # Convert from log scale
                        elif base_param_name == 'Yield stress':
                            display_name = f"Object {obj_id} - Yield stress"
                            p_value = (10**p).mean().item()
                        elif base_param_name == 'plastic viscosity':
                            display_name = f"Object {obj_id} - Plastic viscosity"
                            p_value = (10**p).mean().item()
                        elif base_param_name == 'shear modulus':
                            display_name = f"Object {obj_id} - Shear modulus"
                            p_value = (10**p).mean().item()
                        elif base_param_name == 'bulk modulus':
                            display_name = f"Object {obj_id} - Bulk modulus"
                            p_value = (10**p).mean().item()
                        elif base_param_name == 'friction angle':
                            display_name = f"Object {obj_id} - Friction angle"
                            p_value = p.mean().item()
                        else:
                            display_name = f"Object {obj_id} - {base_param_name}"
                            p_value = p.mean().item() if p.numel() > 1 else p.item()
                        report_msg += f'\n{display_name}: {p_value:.4f}'
                        d.update({name: p_value})
                elif name.startswith('velocity_obj_'):
                    # Already reported above
                    d.update({name: p})
            print(report_msg)
            estimated_params.append(d)

            # Save JSON and trajectory during physical params training (EVERY iteration)
            if estimator.stage[None] == Estimator.physical_params_stage and dataset is not None and config_id is not None:
                # Call export_result with current parameters every iteration
                export_result(dataset, phys_args, estimator, losses, estimated_params, config_id)

                # Save trajectory EVERY iteration
                traj_path = save_trajectory(estimator, dataset, config_id, iteration=i, prediction_frames=30)

                # Only print message every 10 iterations to avoid console spam
                if i % 10 == 0:
                    print(f"  Saved parameters and trajectory (iteration {i}): {os.path.basename(traj_path)}")

            # 2. forward, backward, and update
            estimator.zero_grad()
            estimator.loss[None] = 0.0
            pos_sequence = forward(estimator)
            total_loss = estimator.loss[None] + estimator.image_loss

            # Add auxiliary loss for yield stress range constraint (optional)
            yield_stress_penalty = torch.tensor(0.0, requires_grad=True)
            use_yield_stress_penalty = getattr(phys_args, 'use_yield_stress_penalty', True)
            use_yield_stress_penalty = False
            if use_yield_stress_penalty and estimator.stage[None] == Estimator.physical_params_stage:
                penalty_weight = getattr(phys_args, 'yield_stress_penalty_weight', 1.0)
                # Yield stress should be in range [3, 4] in log10 scale (corresponding to [1000, 10000])
                min_log_yield = 3.1
                max_log_yield = 3.9

                # Find yield stress parameters in optimizer param groups
                optimizer = estimator.get_optimizer()
                for param_group in optimizer.param_groups:
                    param_name = param_group['name']
                    if 'Yield_stress' in param_name:  # Note: capital Y in the param name
                        log_yield_tensor = param_group['params'][0]  # Get the parameter tensor (in log10 scale)
                        log_yield_value = log_yield_tensor.item() if log_yield_tensor.numel() == 1 else log_yield_tensor.mean().item()

                        # Apply ReLU penalty: zero inside [min, max], linear outside
                        penalty = penalty_weight * (torch.relu(min_log_yield - log_yield_tensor) + torch.relu(log_yield_tensor - max_log_yield))
                        yield_stress_penalty = yield_stress_penalty + penalty
                        if penalty > 0:
                            if log_yield_value < min_log_yield:
                                print(f"    Yield stress {10**log_yield_value:.1f} (log={log_yield_value:.3f}) < {10**min_log_yield:.0f}, penalty={penalty.item():.6f}")
                            else:
                                print(f"    Yield stress {10**log_yield_value:.1f} (log={log_yield_value:.3f}) > {10**max_log_yield:.0f}, penalty={penalty.item():.6f}")

                total_loss = total_loss + yield_stress_penalty
                if yield_stress_penalty > 0:
                    # Convert to scalar for printing
                    penalty_value = yield_stress_penalty.item() if torch.is_tensor(yield_stress_penalty) else yield_stress_penalty
                    print(f"  Yield stress penalty: {penalty_value:.6f}")

            # Ensure total_loss is a scalar for the losses list
            if torch.is_tensor(total_loss):
                losses.append(total_loss.item())
            else:
                losses.append(total_loss)

            # Log to wandb if enabled
            if wandb_enabled:
                log_dict = {
                    "loss/total": total_loss,
                    "loss/geometry": estimator.loss[None],
                    "loss/image_total": estimator.image_loss,
                    "iteration": i
                }

                # Add yield stress penalty to wandb if applicable
                if use_yield_stress_penalty and estimator.stage[None] == Estimator.physical_params_stage:
                    log_dict["loss/yield_stress_penalty"] = yield_stress_penalty

                # Add per-object image losses if available
                if hasattr(estimator, 'per_object_sil_losses_accum'):
                    log_dict["loss/image_obj1"] = estimator.per_object_sil_losses_accum.get(1, 0.0)
                    log_dict["loss/image_obj2"] = estimator.per_object_sil_losses_accum.get(2, 0.0)

                # Add learning rates for physical parameters being optimized
                if estimator.stage[None] == Estimator.physical_params_stage:
                    optimizer = estimator.get_optimizer()
                    for param_group in optimizer.param_groups:
                        param_name = param_group['name']
                        log_dict[f"lr/{param_name}"] = param_group['lr']

                wandb.log(log_dict)

            # Save debug info (video + point clouds) every 10 iterations during physical params stage
            if estimator.stage[None] == Estimator.physical_params_stage and dataset is not None:
                if i % 10 == 0 or i == 0:
                    save_training_debug(estimator, i, dataset.model_path, pos_sequence, "physical_params")


            yield_stress_penalty.backward(retain_graph=True)
            backward(estimator)
            estimator.step(i)

            # 3. record loss and save best params
            min_idx = losses.index(min(losses))
            best_params = estimated_params[min_idx]
            print("Best params: ", best_params, 'in {} iteration'.format(min_idx))
            print("Min loss: {}".format(losses[min_idx]))
    
    if estimator.stage[None] == Estimator.velocity_stage and len(losses) > 0:
        min_idx = losses.index(min(losses))
        best_params = estimated_params[min_idx]
        # Update per-object velocities with best values
        for obj_id in estimator.object_velocities.keys():
            param_name = f'velocity_obj_{obj_id}'
            if param_name in best_params:
                estimator.object_velocities[obj_id] = nn.Parameter(best_params[param_name].to(estimator.device))
    
    # Save loss plot if dataset is provided
    if dataset is not None and len(losses) > 0:
        save_path = os.path.join(dataset.model_path, "mpm")
        os.makedirs(save_path, exist_ok=True)

        if estimator.stage[None] == Estimator.velocity_stage:
            save_loss_plot(losses, save_path, "velocity")
        elif estimator.stage[None] == Estimator.physical_params_stage:
            save_loss_plot(losses, save_path, "physical_params")

    # Finish wandb run if it was initialized
    if wandb_enabled:
        wandb.finish()
        print("Wandb run finished")

    return losses, estimated_params

def export_result(dataset, phys_args, estimator: Estimator, losses, estimated_params, config_id):
    save_attr = ['mpm_iter_cnt', 'voxel_size', 'gravity', 'bc', 'fps', 'density_grid_size']
    pred = dict()
    pred['config_id'] = config_id

    # Check if multi-object mode
    if hasattr(phys_args, 'sub_objects') and phys_args.sub_objects:
        # Multi-object mode
        pred['multi_object'] = True
        pred['sub_objects'] = []

        # Get best parameters
        if len(losses) > 0 and len(estimated_params) > 0:
            min_idx = losses.index(min(losses))
            best_params = estimated_params[min_idx]
        else:
            best_params = {}

        # Process each object
        for obj_config in phys_args.sub_objects:
            obj_id = obj_config['object_id']
            obj_data = dict()
            obj_data['object_id'] = obj_id
            obj_data['name'] = obj_config['name']

            # Get learned velocity for this object from best_params or estimator
            # The estimator stores velocities with string keys ('1', '2') not int (1, 2)
            obj_id_str = str(obj_id)
            vel_param_name = f'velocity_obj_{obj_id_str}'

            if vel_param_name in best_params:
                # Get from best_params
                vel_tensor = best_params[vel_param_name]
                vel = vel_tensor.detach().cpu().numpy().tolist() if torch.is_tensor(vel_tensor) else vel_tensor
            elif hasattr(estimator, 'object_velocities') and obj_id_str in estimator.object_velocities:
                # Get directly from estimator's current values
                vel = estimator.object_velocities[obj_id_str].detach().cpu().numpy().tolist()
                print(f"Using estimator velocity for object {obj_id}: {vel}")
            else:
                # Fallback to initial velocity
                vel = obj_config.get('init_vel', [0.0, 0.0, 0.0])
                print(f"WARNING: No learned velocity found for object {obj_id}, using init_vel: {vel}")
            obj_data['vel'] = vel

            # Get material parameters for this object
            mat_params = dict()
            m = obj_config['material']
            mat_params['material'] = m
            mat_params['rho'] = obj_config.get('rho', 1000.0)

            # Extract learned parameters based on material type
            # Parameter names in best_params are like: obj1_Youngs_modulus, obj1_Poisson_ratio, etc.
            if m == 11:  # Newtonian fluid (viscous_fluid)
                # Look for mu and kappa in best_params
                param_name = f'obj{obj_id}_shear_modulus'
                if param_name in best_params:
                    mat_params['mu'] = float(best_params[param_name])
                else:
                    mat_params['mu'] = obj_config.get('mu', 10.0)

                param_name = f'obj{obj_id}_bulk_modulus'
                if param_name in best_params:
                    mat_params['kappa'] = float(best_params[param_name])
                else:
                    mat_params['kappa'] = obj_config.get('kappa', 1e4)

            elif m == 10:  # Elastic
                param_name = f'obj{obj_id}_Youngs_modulus'
                if param_name in best_params:
                    mat_params['E'] = float(best_params[param_name])
                else:
                    mat_params['E'] = obj_config.get('init_E', 1e5)

                param_name = f'obj{obj_id}_Poisson_ratio'
                if param_name in best_params:
                    mat_params['nu'] = float(best_params[param_name])
                else:
                    mat_params['nu'] = obj_config.get('init_nu', 0.25)

            elif m == 12:  # Elastoplastic (von_mises)
                param_name = f'obj{obj_id}_Youngs_modulus'
                if param_name in best_params:
                    mat_params['E'] = float(best_params[param_name])
                else:
                    mat_params['E'] = obj_config.get('init_E', 1e4)

                param_name = f'obj{obj_id}_Poisson_ratio'
                if param_name in best_params:
                    mat_params['nu'] = float(best_params[param_name])
                else:
                    mat_params['nu'] = obj_config.get('init_nu', 0.25)

                param_name = f'obj{obj_id}_Yield_stress'
                if param_name in best_params:
                    mat_params['yield_stress'] = float(best_params[param_name])
                else:
                    mat_params['yield_stress'] = obj_config.get('init_yield_stress', 1e3)

            elif m == 13:  # Sand/Snow (drucker_prager)
                # E and nu usually fixed for sand/snow, but check anyway
                param_name = f'obj{obj_id}_Youngs_modulus'
                if param_name in best_params:
                    mat_params['E'] = float(best_params[param_name])
                else:
                    mat_params['E'] = obj_config.get('init_E', 1e6)

                param_name = f'obj{obj_id}_Poisson_ratio'
                if param_name in best_params:
                    mat_params['nu'] = float(best_params[param_name])
                else:
                    mat_params['nu'] = obj_config.get('init_nu', 0.3)

                param_name = f'obj{obj_id}_friction_angle'
                if param_name in best_params:
                    mat_params['friction_alpha'] = float(best_params[param_name])
                else:
                    mat_params['friction_alpha'] = obj_config.get('init_friction_alpha', 15.0)

            obj_data['mat_params'] = mat_params
            pred['sub_objects'].append(obj_data)

        print("\n=== Exported Multi-Object Parameters ===")
        for obj in pred['sub_objects']:
            print(f"Object {obj['object_id']} ({obj['name']}):")
            print(f"  Velocity: {obj['vel']}")
            print(f"  Material: {obj['mat_params']}")
    else:
        # Single object mode (legacy)
        v = estimator.init_vel.detach().cpu().numpy().tolist()
        pred['vel'] = v
        if len(losses) > 0 and len(estimated_params) > 0:
            min_idx = losses.index(min(losses))
            best_params = estimated_params[min_idx]
        else:
            best_params = {}

        mat_params = dict()
        m = phys_args.material
        mat_params['material'] = m
        mat_params['rho'] = getattr(phys_args, 'rho', 1000.0)

        if (m == MPMSimulator.von_mises and estimator.simulator.non_newtonian == 1) or \
            m == MPMSimulator.viscous_fluid:
            # non_newtonian & newtonian
            mu = best_params.get('shear modulus', 10.0)
            kappa = best_params.get('bulk modulus', 1e4)
            mat_params['mu'] = mu
            mat_params['kappa'] = kappa
        else:
            # elasticity, drucker_prager, plasticine
            if 'Youngs modulus' in best_params and 'Poisson ratio' in best_params:
                E = best_params['Youngs modulus']
                nu = best_params['Poisson ratio']
            else:
                E = float((10 ** estimator.E).detach().cpu().numpy()) if hasattr(estimator, 'E') else 1e5
                nu = float((estimator.get_nu()).detach().cpu().numpy()) if hasattr(estimator, 'get_nu') else 0.25
            mat_params['E'] = E
            mat_params['nu'] = nu

        if m == MPMSimulator.drucker_prager:
            mat_params['friction_alpha'] = best_params.get('friction angle', 15.0)

        if m == MPMSimulator.von_mises:
            ys = best_params.get('Yield stress', 1e3)
            mat_params['yield_stress'] = ys
            if estimator.simulator.non_newtonian == 1:
                eta = best_params.get('plastic viscosity', 1.0)
                mat_params['plastic_viscosity'] = eta

        pred['mat_params'] = mat_params

    # Save common attributes
    for attr in save_attr:
        if hasattr(phys_args, attr):
            pred[attr] = getattr(phys_args, attr)

    # Save the prediction JSON
    output_path = os.path.join(dataset.model_path, f'{config_id}-pred.json')
    with open(output_path, 'w') as f:
        json.dump(pred, f, indent=4)

    print(f"\nSaved predictions to: {output_path}")

if __name__ == "__main__":
    # Set up command line argument parser
    
    # ===========================
    # 0. 解析参数 & 配置
    # ===========================
    
    start_time = time.time()

    parser = ArgumentParser(description="Physical parameter estimation")
    model = ModelParams(parser)#, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    
    
    parser.add_argument("--config_file", default='config/torus.json', type=str)
    parser.add_argument("--skip_velocity", action='store_true', help="Skip velocity estimation and use GT from config")
    parser.add_argument("--use_wandb", action='store_true', help="Enable Weights & Biases logging for real-time tracking")
    parser.add_argument("--resume_from_pred", action='store_true', help="Resume training from existing pred.json file instead of initial config values")
    parser.add_argument("--novel_materials", action='store_true', help="Swap material types and parameters between objects for novel material combination testing")
    # get_combined_args 负责从 parser 生成 gs_args（高斯/渲染相关）和 phys_args（物理相关）
    gs_args, phys_args = get_combined_args(parser)
    config_id = phys_args.id
    args = parser.parse_args()

    # If resuming from pred.json, load and update phys_args
    
    if args.resume_from_pred:
        pred_json_path = os.path.join(model.extract(gs_args).model_path, f'{config_id}-pred.json')
        if os.path.exists(pred_json_path):
            print(f"\n{'='*60}")
            print(f"RESUMING FROM EXISTING PRED.JSON: {pred_json_path}")
            print(f"{'='*60}")

            with open(pred_json_path, 'r') as f:
                pred_data = json.load(f)

            # Update phys_args with learned parameters
            if 'sub_objects' in pred_data:
                for pred_obj in pred_data['sub_objects']:
                    obj_id = pred_obj['object_id']

                    # Find corresponding object in phys_args
                    for phys_obj in phys_args.sub_objects:
                        if phys_obj['object_id'] == obj_id:
                            # Update velocity
                            if 'vel' in pred_obj:
                                phys_obj['init_vel'] = pred_obj['vel']
                                print(f"  Object {obj_id}: Updated velocity to {pred_obj['vel']}")

                            # Update material parameters
                            if 'mat_params' in pred_obj:
                                mat_params = pred_obj['mat_params']
                                material = mat_params.get('material', phys_obj['material'])

                                if material == 10:  # Elastic
                                    if 'E' in mat_params:
                                        phys_obj['init_E'] = mat_params['E']
                                        print(f"  Object {obj_id}: Updated E to {mat_params['E']}")
                                    if 'nu' in mat_params:
                                        phys_obj['init_nu'] = mat_params['nu']
                                        print(f"  Object {obj_id}: Updated nu to {mat_params['nu']}")

                                elif material == 11:  # Fluid
                                    if 'mu' in mat_params:
                                        phys_obj['mu'] = mat_params['mu']
                                        print(f"  Object {obj_id}: Updated mu to {mat_params['mu']}")
                                    if 'kappa' in mat_params:
                                        phys_obj['kappa'] = mat_params['kappa']
                                        print(f"  Object {obj_id}: Updated kappa to {mat_params['kappa']}")

                                elif material == 12:  # Plastic
                                    if 'E' in mat_params:
                                        phys_obj['init_E'] = mat_params['E']
                                        print(f"  Object {obj_id}: Updated E to {mat_params['E']}")
                                    if 'nu' in mat_params:
                                        phys_obj['init_nu'] = mat_params['nu']
                                        print(f"  Object {obj_id}: Updated nu to {mat_params['nu']}")
                                    if 'yield_stress' in mat_params:
                                        phys_obj['init_yield_stress'] = mat_params['yield_stress']
                                        print(f"  Object {obj_id}: Updated yield_stress to {mat_params['yield_stress']}")

                                elif material == 13:  # Sand
                                    if 'E' in mat_params:
                                        phys_obj['init_E'] = mat_params['E']
                                        print(f"  Object {obj_id}: Updated E to {mat_params['E']}")
                                    if 'nu' in mat_params:
                                        phys_obj['init_nu'] = mat_params['nu']
                                        print(f"  Object {obj_id}: Updated nu to {mat_params['nu']}")
                                    if 'friction_alpha' in mat_params:
                                        phys_obj['init_friction_alpha'] = mat_params['friction_alpha']
                                        print(f"  Object {obj_id}: Updated friction_alpha to {mat_params['friction_alpha']}")
                            break

            # Handle novel material combinations if requested
            if args.novel_materials:
                print(f"\n{'='*60}")
                print("CREATING NOVEL MATERIAL COMBINATION")
                print(f"{'='*60}")

                # Get the two objects from pred.json
                pred_obj1 = pred_data['sub_objects'][0]
                pred_obj2 = pred_data['sub_objects'][1]

                # Extract material info from both objects
                mat1 = pred_obj1['mat_params']
                mat2 = pred_obj2['mat_params']

                # Swap materials in phys_args
                for phys_obj in phys_args.sub_objects:
                    if phys_obj['object_id'] == 1:
                        # Object 1 gets Object 2's material
                        old_material = phys_obj['material']
                        phys_obj['material'] = mat2['material']
                        print(f"  Object 1: Material {old_material} -> {mat2['material']}")

                        # Set new parameters from mat2
                        if mat2['material'] == 10:  # Elastic
                            phys_obj['init_E'] = mat2.get('E', 1e5)
                            phys_obj['init_nu'] = mat2.get('nu', 0.25)
                        elif mat2['material'] == 11:  # Fluid
                            phys_obj['mu'] = mat2.get('mu', 10.0)
                            phys_obj['kappa'] = mat2.get('kappa', 1e4)
                        elif mat2['material'] == 12:  # Plastic
                            phys_obj['init_E'] = mat2.get('E', 1e4)
                            phys_obj['init_nu'] = mat2.get('nu', 0.25)
                            phys_obj['init_yield_stress'] = mat2.get('yield_stress', 1e3)
                        elif mat2['material'] == 13:  # Sand
                            phys_obj['init_E'] = mat2.get('E', 1e6)
                            phys_obj['init_nu'] = mat2.get('nu', 0.3)
                            phys_obj['init_friction_alpha'] = mat2.get('friction_alpha', 15.0)

                    elif phys_obj['object_id'] == 2:
                        # Object 2 gets Object 1's material
                        old_material = phys_obj['material']
                        phys_obj['material'] = mat1['material']
                        print(f"  Object 2: Material {old_material} -> {mat1['material']}")

                        # Set new parameters from mat1
                        if mat1['material'] == 10:  # Elastic
                            phys_obj['init_E'] = mat1.get('E', 1e5)
                            phys_obj['init_nu'] = mat1.get('nu', 0.25)
                        elif mat1['material'] == 11:  # Fluid
                            phys_obj['mu'] = mat1.get('mu', 10.0)
                            phys_obj['kappa'] = mat1.get('kappa', 1e4)
                        elif mat1['material'] == 12:  # Plastic
                            phys_obj['init_E'] = mat1.get('E', 1e4)
                            phys_obj['init_nu'] = mat1.get('nu', 0.25)
                            phys_obj['init_yield_stress'] = mat1.get('yield_stress', 1e3)
                        elif mat1['material'] == 13:  # Sand
                            phys_obj['init_E'] = mat1.get('E', 1e6)
                            phys_obj['init_nu'] = mat1.get('nu', 0.3)
                            phys_obj['init_friction_alpha'] = mat1.get('friction_alpha', 15.0)

                print(f"{'='*60}")

                # Set iterations to 0 for prediction only with novel materials
                print("\n  Setting iter_cnt to 0 for prediction only")
                phys_args.iter_cnt = 0
                phys_args.vel_iter_cnt = 0

            print(f"{'='*60}\n")
        else:
            print(f"\nWARNING: pred.json not found at {pred_json_path}, using initial config values\n")

    print(phys_args)
    safe_state(gs_args.quiet)

    # 1. train def gs with multi-object support
    dataset = model.extract(gs_args)
    if not check_gs_model(dataset.model_path, gs_args.save_iterations):
        raise RuntimeError(
            f"No pre-trained Gaussian model found in {dataset.model_path}. "
            f"Please run your static Gaussian reconstruction script first."
        )
    # 不再调用 training_MO(...)
    torch.cuda.empty_cache()   
    print("Loading ground truth point clouds for multi-object scenario...")
    print(dataset.source_path)
    gt_pcs_per_object =  load_multiobject_gt_pcs(dataset.source_path)
    
    print_torch_memory("after loading GT point clouds")
    
    
        
    # 2. estimate velocity (multi-object)
    gts_per_object, gts_combined, vol, vol_densities, grid_size, volume_surface, cam_info, vol1, vol2 = prepare_gt_multi_with_pcds(model.extract(gs_args), gs_args.iterations, pipeline.extract(gs_args), phys_args, gt_pcs_per_object=gt_pcs_per_object)
    
    torch.cuda.empty_cache()
    
    print_torch_memory("after prepare_gt_multi")
    print_torch_memory("before Estimator")
    
    ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_fraction=0.5)
    print('gt point count: {}'.format(gts_combined[0].shape[0] if gts_combined else 0))
    
    # Get particle materials and object labels from cam_info
    particle_materials = cam_info.get("particle_materials", None)
    object_labels = cam_info.get("object_labels", None)
    print("Object labels:")
    print(object_labels)
    
    
    
    
    # print("\n" + "=" * 80)
    # print("Estimator input summary")
    # print("=" * 80)

    # # ------- helper for安全打印 -------
    # def summarize_tensor(name, t, max_print_shape_only=True):
    #     if t is None:
    #         print(f"{name}: None")
    #         return
    #     if isinstance(t, list):
    #         print(f"{name}: list, len = {len(t)}")
    #         if len(t) > 0 and torch.is_tensor(t[0]):
    #             print(f"  [{name}[0]] shape = {t[0].shape}, dtype = {t[0].dtype}, device = {t[0].device}")
    #         return
    #     if torch.is_tensor(t):
    #         print(f"{name}: tensor, shape = {tuple(t.shape)}, dtype = {t.dtype}, device = {t.device}")
    #         if t.numel() > 0:
    #             print(f"  {name}: min = {t.min().item():.4e}, max = {t.max().item():.4e}")
    #         return
    #     if isinstance(t, np.ndarray):
    #         print(f"{name}: np.ndarray, shape = {t.shape}, dtype = {t.dtype}")
    #         if t.size > 0:
    #             print(f"  {name}: min = {t.min():.4e}, max = {t.max():.4e}")
    #         return
    #     print(f"{name}: type = {type(t)}")

    # # ------- gts_combined / per-object gts -------
    # summarize_tensor("vol (init_vol)", vol)
    # summarize_tensor("vol_densities", vol_densities)
    # summarize_tensor("grid_size", grid_size)
    # summarize_tensor("volume_surface", volume_surface)

    # print("\nGT combined frames:")
    # if gts_combined is None:
    #     print("  gts_combined = None")
    # else:
    #     print(f"  len(gts_combined) = {len(gts_combined)}")
    #     if len(gts_combined) > 0:
    #         print(f"  gts_combined[0].shape = {gts_combined[0].shape}, "
    #               f"dtype = {gts_combined[0].dtype}, device = {gts_combined[0].device}")
    #         total_gt_pts = sum([g.shape[0] for g in gts_combined])
    #         print(f"  total GT points over all frames = {total_gt_pts}")

    # print("\nGT per object:")
    # if gts_per_object is None:
    #     print("  gts_per_object = None")
    # else:
    #     for oid, frames in gts_per_object.items():
    #         if frames is None:
    #             print(f"  object {oid}: None")
    #             continue
    #         print(f"  object {oid}: {len(frames)} frames")
    #         if len(frames) > 0:
    #             print(f"    frame 0 shape = {frames[0].shape}, dtype = {frames[0].dtype}, device = {frames[0].device}")
    #             print(f"    total points = {sum([f.shape[0] for f in frames])}")

    # # ------- cam_info / labels / materials -------
    # particle_materials = cam_info.get("particle_materials", None)
    # object_labels = cam_info.get("object_labels", None)

    # print("\ncam_info basic keys:", list(cam_info.keys()))
    # summarize_tensor("particle_materials", particle_materials)
    # summarize_tensor("object_labels", object_labels)

    # if torch.is_tensor(particle_materials):
    #     uniq_mats, counts_mats = torch.unique(particle_materials, return_counts=True)
    #     print("  particle_materials unique:", list(zip(uniq_mats.tolist(), counts_mats.tolist())))
    # if torch.is_tensor(object_labels):
    #     uniq_lbls, counts_lbls = torch.unique(object_labels, return_counts=True)
    #     print("  object_labels unique:", list(zip(uniq_lbls.tolist(), counts_lbls.tolist())))

    # # ------- 一致性检查 (很关键) -------
    # if torch.is_tensor(vol):
    #     n_vol = vol.shape[0]
    # else:
    #     n_vol = len(vol) if vol is not None else 0

    # n_mat = particle_materials.shape[0] if torch.is_tensor(particle_materials) else None
    # n_lbl = object_labels.shape[0] if torch.is_tensor(object_labels) else None
    # print("\nSanity check:")
    # print(f"  n_vol (particles)      = {n_vol}")
    # print(f"  n_mat (materials)      = {n_mat}")
    # print(f"  n_lbl (object_labels)  = {n_lbl}")

    # if n_mat is not None and n_mat != n_vol:
    #     print("  [WARNING] len(particle_materials) != number of particles in vol !!!")
    # if n_lbl is not None and n_lbl != n_vol:
    #     print("  [WARNING] len(object_labels) != number of particles in vol !!!")

    # # ------- phys_args / MPM 相关参数也顺便看一下 -------
    # print("\nphys_args summary:")
    # for attr in ["voxel_size", "density_grid_size", "fps", "mpm_iter_cnt",
    #              "vel_iter_cnt", "iter_cnt"]:
    #     if hasattr(phys_args, attr):
    #         print(f"  {attr} = {getattr(phys_args, attr)}")

    # if hasattr(phys_args, "sub_objects"):
    #     print("\nphys_args.sub_objects:")
    #     for sobj in phys_args.sub_objects:
    #         print("  ", sobj)

    # print("=" * 80 + "\n")
    
    print_torch_memory("before Estimator")
    
    
    
    # Pass both combined GTs (for compatibility) and per-object GTs (for per-object loss)
    estimator = Estimator(phys_args, 'float32', gts_combined, init_vol=vol,
                          gts_per_object=gts_per_object, 
                          surface_index=volume_surface, dynamic_scene=None,
                          image_scale=image_scale, pipeline=pipeline.extract(gs_args), 
                          image_op=op.extract(gs_args), particle_materials=particle_materials,
                          object_labels=object_labels)
    print("I'm here")
    # Check if we should skip velocity estimation
    args = parser.parse_args()
    if args.skip_velocity:
        print("\n" + "="*50)
        print("SKIPPING VELOCITY ESTIMATION - USING CONFIG VALUES")
        print("="*50)
        for obj_config in phys_args.sub_objects:
            print(f"Object {obj_config['object_id']}: velocity = {obj_config['init_vel']}")
        print("="*50 + "\n")
        losses, e_s = [], []
    else:
        estimator.set_stage(Estimator.velocity_stage)
        losses, e_s = train(estimator, phys_args, phys_args.vel_estimation_frames, dataset, use_wandb=False, config_id=config_id)
        torch.cuda.empty_cache()
    
    scene = assign_gs_to_pcd(vol, vol_densities, dataset, op.extract(gs_args),
                                    pipeline.extract(gs_args),
                                    cam_info, phys_args.density_grid_size)
    estimator.set_scene(scene)
    max_f = len(gts_combined)
    estimator.set_stage(Estimator.physical_params_stage)
    losses, e_s = train(estimator, phys_args, max_f, dataset, use_wandb=args.use_wandb, config_id=config_id)
    print(phys_args)
    # Print per-object velocities
    for obj_id, velocity in estimator.object_velocities.items():
        print(f"Object {obj_id} final velocity: {velocity.detach().cpu().numpy()}")
    print(config_id)
    export_result(dataset, phys_args, estimator, losses, e_s, config_id)

    # Save final trajectory after training
    print("Saving final training trajectory (30 frames)...")
    # Check if this is novel material mode
    is_novel = args.novel_materials if 'args' in locals() and hasattr(args, 'novel_materials') else False
    trajectory_path = save_trajectory(estimator, dataset, config_id, iteration=None, prediction_frames=30, is_novel=is_novel)
    print(f"Saved final trajectory to: {trajectory_path}")

    print("consume time {}".format(time.time() - start_time))
    