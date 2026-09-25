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

# Try to import wandb for real-time tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with 'pip install wandb' for real-time tracking")


image_scale = 1.0


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
        classes = list(range(1, object_probs.shape[1]))  # baselines: K objects
        print("Initial distribution - Background: {}, ".format((object_assignments == 0).sum()) +
              ", ".join(f"Obj{c}: {(object_assignments == c).sum()}" for c in classes))

        corrected_assignments = object_assignments.copy()

        # Process each object class separately
        for obj_class in classes:
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
                others = [(c, positions[object_assignments == c]) for c in classes
                          if c != obj_class and (object_assignments == c).sum() > 0]
                for cluster_id in small_clusters:
                    cluster_points = class_indices[labels == cluster_id]
                    # Reassign each point to the nearest other class when it is close enough
                    for point_idx in cluster_points:
                        point_pos = positions[point_idx:point_idx + 1]
                        best_c, best_d = None, np.inf
                        for c, opos in others:
                            d = np.linalg.norm(opos - point_pos, axis=1).min()
                            if d < best_d:
                                best_c, best_d = c, d
                        if best_c is not None and best_d < distance_threshold * 2:
                            corrected_assignments[point_idx] = best_c

                print(f"    Reassigned {len(small_clusters)} small clusters")

        print("Final distribution - Background: {}, ".format((corrected_assignments == 0).sum()) +
              ", ".join(f"Obj{c}: {(corrected_assignments == c).sum()}" for c in classes))

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
        print("Initial distribution - Background: {}, ".format((object_assignments == 0).sum()) +
              ", ".join(f"Obj{c}: {(object_assignments == c).sum()}" for c in range(1, object_probs.shape[1])))

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

                # Count occurrences of each object label (baselines: K objects)
                total_count = len(non_bg_neighbors)
                counts = np.bincount(non_bg_neighbors, minlength=object_probs.shape[1])
                majority_label = int(counts[1:].argmax()) + 1
                majority_ratio = counts[majority_label] / total_count

                # Change label if strong majority and different from current
                if majority_ratio >= min_ratio and corrected_assignments[i] != majority_label:
                    new_assignments[i] = majority_label
                    changes += 1

            corrected_assignments = new_assignments
            print(f"  Iteration {iteration+1}: {changes} particles corrected")

            if changes == 0:
                break

        print("Final distribution - Background: {}, ".format((corrected_assignments == 0).sum()) +
              ", ".join(f"Obj{c}: {(corrected_assignments == c).sum()}" for c in range(1, object_probs.shape[1])))

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

    # baselines (PhysON): any number of objects, ids 1..K from phys_args.sub_objects
    from utils.object_palette import obj_color_u8
    obj_ids = [int(o['object_id']) for o in phys_args.sub_objects]
    obj_names = {int(o['object_id']): o.get('name', f"obj{o['object_id']}") for o in phys_args.sub_objects}
    outputs = {}
    for oid in obj_ids:
        print(f"\n=== Processing Object {oid} ({obj_names[oid]}) ===")
        gaussians_obj = filter_gaussians_by_object(original_gaussians, object_id=oid, use_clustering_fix=use_clustering_fix)
        scene_full.gaussians = gaussians_obj  # Update scene's gaussians
        outputs[oid] = prepare_gt(dataset, iteration, pipeline, phys_args, gaussians=gaussians_obj, scene=scene_full, object_id=oid)
        print(f"Object {oid} ({obj_names[oid]}) after filling: {len(outputs[oid][1]):,} particles")

    # Restore original gaussians
    scene_full.gaussians = original_gaussians

    print("\n=== Combining Results ===")
    grid_size = outputs[obj_ids[0]][3]
    cam_info = outputs[obj_ids[0]][5]
    vols = {oid: outputs[oid][1] for oid in obj_ids}
    present = [oid for oid in obj_ids if len(vols[oid]) > 0]
    if not present:
        raise RuntimeError("No particles found for any object")
    vol_combined = torch.cat([vols[oid] for oid in present], dim=0)
    vol_densities_combined = torch.cat([outputs[oid][2] for oid in present], dim=0)
    surfaces, labels, offset = [], [], 0
    for oid in present:
        surfaces.append(outputs[oid][4] + offset)  # surface indices shifted into the combined array
        labels.append(torch.full((len(vols[oid]),), oid, dtype=torch.int32, device=vol_combined.device))
        offset += len(vols[oid])
    volume_surface_combined = torch.cat(surfaces)
    object_labels = torch.cat(labels)
    print("Combined: {} particles ({})".format(len(vol_combined), ", ".join(f"Obj{oid}: {len(vols[oid])}" for oid in obj_ids)))

    # Combine gts frames (object order = config order, as the per-object GT clouds)
    gts_per_object = {oid: outputs[oid][0] for oid in obj_ids}
    gts_combined = []
    max_frames = max(len(gts_per_object[oid]) for oid in obj_ids)
    for i in range(max_frames):
        frame_points = [gts_per_object[oid][i] for oid in obj_ids if i < len(gts_per_object[oid])]
        if frame_points:
            gts_combined.append(torch.cat(frame_points, dim=0))

    # Store object labels in cam_info
    cam_info["object_labels"] = object_labels

    # Save GT surface points for all frames, coloured per object
    if gts_combined:
        for frame_idx, frame_gts in enumerate(gts_combined):
            cols = [np.tile(np.asarray(obj_color_u8(oid), np.uint8), (len(gts_per_object[oid][frame_idx]), 1))
                    for oid in obj_ids if frame_idx < len(gts_per_object[oid])]
            write_particles(frame_gts, frame_idx, dataset.model_path, 'gt_surface', vertex_colors=np.concatenate(cols, 0))
        print(f"Saved GT surface points for all {len(gts_combined)} frames to {dataset.model_path}/mpm/gt_surface_*.ply")
        print(f"Points in frame 0: {len(gts_combined[0]) if gts_combined else 0}")

    # Save combined multi-object particles with color coding and object labels
    labels_np = object_labels.cpu().numpy()
    palette = np.asarray([obj_color_u8(i) for i in range(int(labels_np.max()) + 1)], dtype=np.uint8)
    colors_np = palette[labels_np]
    from utils.system_utils import write_ply_with_labels
    mpm_path = os.path.join(dataset.model_path, 'mpm')
    os.makedirs(mpm_path, exist_ok=True)
    write_ply_with_labels(os.path.join(mpm_path, 'multi_object_0.ply'), vol_combined, colors_np, object_labels)
    print(f"Saved combined particles with object labels to {mpm_path}/multi_object_0.ply")

    # Create per-particle material types from object IDs and sub_objects config
    particle_materials = torch.zeros(len(vol_combined), dtype=torch.int32, device=vol_combined.device)
    for obj_config in phys_args.sub_objects:
        particle_materials[object_labels == int(obj_config['object_id'])] = int(obj_config['material'])
    cam_info["particle_materials"] = particle_materials
    print("Material types - " + ", ".join(f"Object {o['object_id']}: {o['material']}" for o in phys_args.sub_objects))

    # Save per-object particles separately for verification
    print("\nParticle Statistics:")
    print(f"  Total particles: {len(object_labels)}")
    for oid in obj_ids:
        obj_mask = object_labels == oid
        n = int(obj_mask.sum().item())
        if n > 0:
            write_particles(vol_combined[obj_mask], 0, dataset.model_path, f'object{oid}_particles',
                            vertex_colors=np.tile(np.asarray(obj_color_u8(oid), np.uint8), (n, 1)))
        print(f"  Object {oid} particles: {n} ({n / len(object_labels) * 100:.1f}%)")

    return gts_per_object, gts_combined, vol_combined, vol_densities_combined, grid_size, volume_surface_combined, cam_info, vols


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

                # Save silhouette comparisons for this frame (matching render_forward exactly);
                # baselines: one GT|render row per object
                if hasattr(gaussians, 'get_object_probs') and len(views) > 0:
                    object_probs = gaussians.get_object_probs  # [N, K+1]
                    object_assignments = torch.argmax(object_probs, dim=1)
                    obj_ids_dbg = list(range(1, object_probs.shape[1]))
                    original_opacity = gaussians._opacity.detach().clone()
                    view = sorted_views[0] if len(sorted_views) > 0 else views[0]
                    if all(hasattr(view, f'gt_alpha_mask_obj{k}') for k in obj_ids_dbg):
                        comps = []
                        for k in obj_ids_dbg:
                            m = (object_assignments == k).float()
                            gaussians._opacity.data = original_opacity * m.unsqueeze(1) - 1e2 * (1 - m.unsqueeze(1))
                            with torch.no_grad():
                                alpha_k = render(view, gaussians, estimator.pipeline, background, d_xyz, 0.0, 0.0, False)["alpha"].cpu().numpy()[0]
                            gt_k = getattr(view, f'gt_alpha_mask_obj{k}').cpu().numpy()[0]
                            alpha_k = (alpha_k * 255).astype(np.uint8)
                            gt_k = (gt_k * 255).astype(np.uint8)
                            h, w = alpha_k.shape
                            comp = np.zeros((h, w * 2), dtype=np.uint8)
                            comp[:, :w] = gt_k
                            comp[:, w:] = alpha_k
                            cv2.putText(comp, f"Obj{k}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
                            comps.append(comp)
                        full_comparison = np.vstack(comps)
                        w = comps[0].shape[1] // 2
                        cv2.putText(full_comparison, "GT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
                        cv2.putText(full_comparison, "Rendered", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
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

                # Save trajectory every traj_save_interval iterations (baselines: upstream saved it EVERY
                # iteration, which re-simulates prediction_frames frames and doubles the cost)
                traj_every = int(getattr(phys_args, 'traj_save_interval', 1))
                if traj_every > 0 and i % traj_every == 0:
                    traj_path = save_trajectory(estimator, dataset, config_id, iteration=i,
                                                prediction_frames=int(getattr(phys_args, 'n_frames', 30)))
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
    gs_args, phys_args = get_combined_args(parser)
    config_id = phys_args.id

    # If resuming from pred.json, load and update phys_args
    args = parser.parse_args()
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
    from scene.gaussian_model import set_num_objects
    set_num_objects(len(phys_args.sub_objects))  # baselines: background + K object channels

    # 1. train def gs with multi-object support
    dataset = model.extract(gs_args)
    if not check_gs_model(dataset.model_path, gs_args.save_iterations):
        # training(dataset, op.extract(gs_args), pipeline.extract(gs_args), gs_args.test_iterations + list(range(10000, 40001, 1000)), gs_args.save_iterations)
        training_MO(dataset, op.extract(gs_args), pipeline.extract(gs_args), gs_args.test_iterations + list(range(10000, 70001, 1000)), gs_args.save_iterations)
    torch.cuda.empty_cache()
    
    # 2. estimate velocity (multi-object)
    gts_per_object, gts_combined, vol, vol_densities, grid_size, volume_surface, cam_info, vols_per_object = prepare_gt_multi(model.extract(gs_args), gs_args.iterations, pipeline.extract(gs_args), phys_args)
    torch.cuda.empty_cache()
    if os.environ.get('TI_DEVICE_MEMORY_GB'):  # baselines: shared GPUs, cap taichi preallocation
        ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_GB=float(os.environ['TI_DEVICE_MEMORY_GB']))
    else:
        ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_fraction=0.5)
    print('gt point count: {}'.format(gts_combined[0].shape[0] if gts_combined else 0))
    
    # Get particle materials and object labels from cam_info
    particle_materials = cam_info.get("particle_materials", None)
    object_labels = cam_info.get("object_labels", None)
    
    # Pass both combined GTs (for compatibility) and per-object GTs (for per-object loss)
    estimator = Estimator(phys_args, 'float32', gts_combined, init_vol=vol,
                          gts_per_object=gts_per_object, 
                          surface_index=volume_surface, dynamic_scene=None,
                          image_scale=image_scale, pipeline=pipeline.extract(gs_args), 
                          image_op=op.extract(gs_args), particle_materials=particle_materials,
                          object_labels=object_labels)
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
    # 2.5. Analyze impact for elastic/plastic materials only
    # from utils.impact_utils import analyze_impact_from_trajectory
    # from simulator import MPMSimulator
    #
    # # Check which objects are elastic or plastic (not fluid or sand)
    # impact_results = {}
    # for obj_config in phys_args.sub_objects:
    #     obj_id = obj_config['object_id']
    #     material = obj_config['material']
    #
    #     # Only analyze elastic and plastic materials
    #     # 10 = elasticity (elastic)
    #     # 11 = viscous_fluid (skip)
    #     # 12 = von_mises (plastic)
    #     # 13 = drucker_prager (sand, skip)
    #     if material == MPMSimulator.elasticity or material == MPMSimulator.von_mises:
    #         material_name = "elastic" if material == MPMSimulator.elasticity else "plastic"
    #         print(f"\nAnalyzing impact for Object {obj_id} (material={material}, {material_name})...")
    #
    #         # Get per-object GT frames
    #         if obj_id in gts_per_object:
    #             obj_frames = gts_per_object[obj_id]
    #
    #             # Analyze impact
    #             from utils.impact_utils import estimate_material_parameters
    #             impact_metrics = analyze_impact_from_trajectory(obj_frames, fps=phys_args.fps, return_raw=False)
    #
    #             if impact_metrics:
    #                 # Calculate total mass for this object using MPM particles
    #                 # Mass = rho * volume, where volume = voxel_size^3 * num_particles
    #                 obj_rho = obj_config.get('rho', 1000.0)  # Get density from config
    #                 voxel_size = phys_args.voxel_size
    #
    #                 # Get actual MPM particle count for this object from vol (init particles)
    #                 # vol contains the initial particle positions from MPM
    #                 if obj_id == 1:
    #                     print("num of point: ", len(vol1))
    #                     obj_mass = obj_rho * ((voxel_size*0.5) ** 3) * len(vol1)
    #                 else:
    #                     print("num of point: ", len(vol2))
    #                     obj_mass = obj_rho * ((voxel_size*0.5) ** 3) * len(vol2)
    #
    #                 impact_results[obj_id] = impact_metrics
    #                 impact_metrics['mass'] = obj_mass  # Add mass to metrics
    #
    #                 print(f"  Object {obj_id} impact analysis:")
    #                 print(f"    Mass: {obj_mass:.4f} kg")
    #                 print(f"    Impact velocity: {impact_metrics['speed_impact']:.3f} m/s")
    #                 print(f"    Contact duration: {impact_metrics['t_contact']:.3f} s")
    #                 print(f"    Max deformation: {impact_metrics['delta_L_max']:.3f} m ({impact_metrics['strain_axial']:.1%})")
    #                 print(f"    Max contact area: {impact_metrics['A_max']:.4f} m²")
    #                 print(f"    Impact type: {impact_metrics['impact_type']}")
    #
    #                 # Estimate material parameters
    #                 param_estimates = estimate_material_parameters(
    #                     impact_metrics,
    #                     mass=obj_mass,
    #                     material_type=material,
    #                     fps=phys_args.fps
    #                 )
    #
    #                 if 'E' in param_estimates:
    #                     print(f"    E estimate: {param_estimates['E']:.2e} Pa")
    #                     impact_metrics['E_estimate'] = param_estimates['E']
    #
    #                 if 'nu' in param_estimates:
    #                     print(f"    ν estimate: {param_estimates['nu']:.3f}")
    #                     impact_metrics['nu_estimate'] = param_estimates['nu']
    #
    #                 if 'yield_stress' in param_estimates:
    #                     print(f"    σ_y estimate: {param_estimates['yield_stress']:.2e} Pa")
    #                     impact_metrics['yield_stress_estimate'] = param_estimates['yield_stress']
    #
    #                 # Store in estimator for later use in material estimation
    #                 if not hasattr(estimator, 'impact_results'):
    #                     estimator.impact_results = {}
    #                 estimator.impact_results[obj_id] = impact_metrics
    #             else:
    #                 print(f"  No impact detected for Object {obj_id}")
    #     else:
    #         material_names = {11: "fluid", 13: "sand"}
    #         print(f"Skipping impact analysis for Object {obj_id} ({material_names.get(material, 'unknown')} material={material})")

    # 3. estimate physical parameters
    # scene = train_gs_with_fixed_pcd(vol, dataset, op.extract(gs_args),
    #                                 pipeline.extract(gs_args),
    #                                 gs_args.test_iterations + list(range(10000, 40001, 1000)),
    #                                 gs_args.save_iterations, None, phys_args.fps, True,
    #                                 cam_info, phys_args.density_grid_size)
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
    