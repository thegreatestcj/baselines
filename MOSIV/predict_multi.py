import os
import time
import json
import scipy
import torch
import torchvision
import random
import subprocess
import numpy as np
import taichi as ti
import open3d as o3d
import trimesh as tm
from tqdm import tqdm
from simulator.simulator_multi import MultiObjectSimulator
from utils.general_utils import safe_state
from pytorch3d.loss import chamfer_distance
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from new_trajectory import load_pcd_file, read_estimation_result, gen_xyz_list, render_new
from train_gs_fixed_pcd import train_gs_with_fixed_pcd
from utils.image_utils import psnr
from utils.loss_utils import ssim
from gaussian_renderer import render
from pathlib import Path


def discretize(pcd, vs):
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd.cpu().numpy())
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_o3d, voxel_size=vs)
    voxels = voxel_grid.get_voxels()
    np_pcd = np.zeros((len(voxels), 3))
    for j in range(len(voxels)):
        c = voxel_grid.get_voxel_center_coordinate(voxels[j].grid_index)
        np_pcd[j, :] = c
    return torch.from_numpy(np_pcd).to('cuda', dtype=torch.float32).contiguous()


def load_gt_pcds(path):
    """Load multi-object GT from folders 0/ and 1/"""
    gts = []
    obj0_path = os.path.join(path, "0")
    obj1_path = os.path.join(path, "1")

    if not os.path.exists(obj0_path) or not os.path.exists(obj1_path):
        raise FileNotFoundError(f"GT folders not found. Expected {obj0_path} and {obj1_path}")

    # Get frame indices from first object folder
    indices = [ply.split('.')[0] for ply in os.listdir(obj0_path) if ply.endswith('.ply')]
    indices.sort()
    n_digits = len(indices[0])
    indices = [int(idx) for idx in indices]
    indices.sort()

    for idx in range(len(indices)):
        # Load object 0
        pcd0 = tm.load_mesh(os.path.join(obj0_path, f"{idx:0{n_digits}d}.ply"))
        np_pcd0 = np.array(pcd0.vertices)

        # Load object 1
        pcd1 = tm.load_mesh(os.path.join(obj1_path, f"{idx:0{n_digits}d}.ply"))
        np_pcd1 = np.array(pcd1.vertices)

        # Combine both objects
        combined_pcd = np.vstack([np_pcd0, np_pcd1])
        gts.append(torch.from_numpy(combined_pcd).to('cuda', dtype=torch.float32).contiguous())

    return gts


def emd_func(x, y, pkg="torch"):
    if pkg == "numpy":
        # numpy implementation
        x_ = np.repeat(np.expand_dims(x, axis=1), y.shape[0], axis=1)  # x: [N, M, D]
        y_ = np.repeat(np.expand_dims(y, axis=0), x.shape[0], axis=0)  # y: [N, M, D]
        cost_matrix = np.linalg.norm(x_ - y_, 2, axis=2)
        try:
            ind1, ind2 = scipy.optimize.linear_sum_assignment(
                cost_matrix, maximize=False
            )
        except:
            # pdb.set_trace()
            print("Error in linear sum assignment!")

        emd = np.mean(np.linalg.norm(x[ind1] - y[ind2], 2, axis=1))
    else:
        # torch implementation
        x_ = x[:, None, :].repeat(1, y.size(0), 1)  # x: [N, M, D]
        y_ = y[None, :, :].repeat(x.size(0), 1, 1)  # y: [N, M, D]
        dis = torch.norm(torch.add(x_, -y_), 2, dim=2)  # dis: [N, M]
        cost_matrix = dis.detach().cpu().numpy()
        try:
            ind1, ind2 = scipy.optimize.linear_sum_assignment(
                cost_matrix, maximize=False
            )
        except:
            # pdb.set_trace()
            print("Error in linear sum assignment!")

        emd = torch.mean(torch.norm(torch.add(x[ind1], -y[ind2]), 2, dim=1))

    return emd


def evaluate(preds, gts, train_frames, loss_type='CD'):
    print(f"Prediction sequence {len(preds)}, gts sequence {len(gts)}")
    if len(preds) != len(gts):
        print("[Error]: The prediction sequence is not align with the gt sequence.")
        return

    # Initial point counts
    pred_points_initial = preds[0].shape[0]
    gt_points_initial = gts[0].shape[0]
    print(f"Initial - Prediction points: {pred_points_initial}, GT points: {gt_points_initial}")
    print(f"Point count ratio (pred/gt): {pred_points_initial/gt_points_initial:.2f}")

    max_f = len(preds)
    fit_loss = 0.0
    predict_loss = 0.0

    for f in tqdm(range(max_f), desc=f"Evaluate {loss_type} Loss"):
        pcd0_original = preds[f]  # Prediction
        pcd1_original = gts[f]    # Ground truth

        pred_points = pcd0_original.shape[0]
        gt_points = pcd1_original.shape[0]

        # Uniform downsampling to match point counts
        if gt_points > pred_points:
            # GT has more points, downsample GT to match prediction
            indices = np.linspace(0, gt_points-1, pred_points, dtype=int)
            pcd1 = pcd1_original[indices, :]
            pcd0 = pcd0_original
            if f == 0:  # Print only for first frame
                print(f"Downsampling GT from {gt_points} to {pred_points} points")
        elif pred_points > gt_points:
            # Prediction has more points, downsample prediction to match GT
            indices = np.linspace(0, pred_points-1, gt_points, dtype=int)
            pcd0 = pcd0_original[indices, :]
            pcd1 = pcd1_original
            if f == 0:  # Print only for first frame
                print(f"Downsampling Prediction from {pred_points} to {gt_points} points")
        else:
            # Equal points
            pcd0 = pcd0_original
            pcd1 = pcd1_original
            if f == 0:
                print(f"Equal point counts: {pred_points}")

        # Verify downsampling worked
        if f == 0:
            print(f"After downsampling - Pred: {pcd0.shape[0]}, GT: {pcd1.shape[0]}")

        # Determine sample size for evaluation
        n_sample = 2048 if loss_type == 'EMD' else 8192
        n_sample = min(n_sample, pcd0.shape[0], pcd1.shape[0])

        if f == 0:
            print(f"Sampling {n_sample} points for {loss_type} evaluation")

        # Random sampling for evaluation
        sample_indices_0 = random.sample(range(pcd0.shape[0]), n_sample)
        sample_indices_1 = random.sample(range(pcd1.shape[0]), n_sample)
        pcd0_sampled = pcd0[sample_indices_0, :]
        pcd1_sampled = pcd1[sample_indices_1, :]

        if loss_type == "CD":
            loss = (chamfer_distance(pcd0_sampled[None], pcd1_sampled[None])[0] * 1e3).item()
        elif loss_type == "EMD":
            loss = emd_func(pcd0_sampled, pcd1_sampled)
        else:
            print("[Error]: undefined error type.")

        if f < train_frames:
            fit_loss += loss
        else:
            predict_loss += loss

    fit_loss /= train_frames
    if max_f - train_frames > 0.0:
        predict_loss /= (max_f - train_frames)

    print(f"\n{loss_type} Results:")
    print(f"  Train ({train_frames} frames): {fit_loss:.4f}")
    print(f"  Test ({max_f - train_frames} frames): {predict_loss:.4f}")

    return fit_loss, predict_loss


if __name__ == "__main__":
    start_time = time.time()

    parser = ArgumentParser(description="Prediction")
    parser.add_argument("--predict_frames", default=30, type=int)
    parser.add_argument("--train_frames", type=int)  # , default=14, type=int)
    parser.add_argument("--gt_path", type=str)
    parser.add_argument('-cid', '--config_id', type=int, default=0)
    model = ModelParams(parser)  # , sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    gs_args, phys_args = get_combined_args(parser)
    # phys_args.id already contains the correct config ID from the JSON file
    # print(phys_args)
    safe_state(gs_args.quiet)
    dataset = model.extract(gs_args)
    opt = op.extract(gs_args)
    pipe = pipeline.extract(gs_args)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_fraction=0.5)

    model_path = Path(dataset.model_path)
    obj_name = dataset.model_path.split('/')[-1]
    # Keep old folders for backward compatibility
    (model_path / f'{obj_name}_img_render').mkdir(exist_ok=True)
    (model_path / f'{obj_name}_img_gt').mkdir(exist_ok=True)

    # 0. Load trained pcd with object labels
    vol, object_labels = load_pcd_file(dataset.model_path, gs_args.iterations, return_labels=True)
    if object_labels is None:
        raise ValueError("No object labels found in PLY file. Please re-run training to save labels.")

    estimation_params = Namespace(**read_estimation_result(dataset, phys_args))
    config_file = gs_args.config_file if hasattr(gs_args, 'config_file') else None
    if not config_file:
        # Try to construct config file path from dataset ID
        config_id = phys_args.id if hasattr(phys_args, 'id') else '01'
        config_file = f'config/genesismo/{config_id}.json'
    # Load saved trajectory from training for perfect consistency
    config_id = phys_args.id if hasattr(phys_args, 'id') else '01'
    pred_traj_path = os.path.join(dataset.model_path, 'pred_traj', f'{config_id}-pred_traj.npy')

    if os.path.exists(pred_traj_path):
        print(f"Loading saved trajectory from: {pred_traj_path}")
        saved_trajectory = np.load(pred_traj_path)
        # Convert to d_xyz_list (differences from initial position)
        d_xyz_list = []
        for f in range(min(len(saved_trajectory), 20)):  # Use first 20 frames for training
            xyz = torch.from_numpy(saved_trajectory[f]).to('cuda', dtype=torch.float32)
            if f == 0:
                d_xyz_list.append(torch.zeros_like(xyz))
            else:
                d_xyz_list.append((xyz - vol).detach())
    else:
        print(f"Warning: No saved trajectory found at {pred_traj_path}, generating new one")
        simulator = MultiObjectSimulator(estimation_params, vol, object_labels, config_file)
        d_xyz_list = gen_xyz_list(simulator, gs_args.predict_frames, diff=True, save_ply=False, path=dataset.model_path)
        d_xyz_list = d_xyz_list[:20]
    # Check if trained model exists and only retrain if necessary
    from utils.system_utils import check_gs_model
    model_exists = check_gs_model(dataset.model_path, gs_args.save_iterations, fix_pcd=True)

    if model_exists:
        print(f"Found existing trained GS model at iteration {opt.iterations}, loading it...")
        force_train = False
    else:
        print(f"No existing trained GS model found, training new model...")
        force_train = True

    scene = train_gs_with_fixed_pcd(
        vol, dataset,
        opt, pipe,
        gs_args.test_iterations + list(range(10000, 40001, 1000)), gs_args.save_iterations,
        d_xyz_list, estimation_params.fps,
        force_train=force_train,
        grid_size=estimation_params.density_grid_size
    )
    views = scene.getTrainCameras(scale=dataset.res_scale).copy()
    # views = [view for view in views if view.fid > gs_args.train_frames / estimation_params.fps - 1e-4]

    # Get unique view UIDs and create folders for each view
    unique_view_ids = sorted(list(set([view.uid for view in views])))
    print(f"Found {len(unique_view_ids)} unique views: {unique_view_ids}")
    view_folders = {}
    for view_id in unique_view_ids:
        render_folder = model_path / f'{obj_name}_img_render_view{view_id}'
        gt_folder = model_path / f'{obj_name}_img_gt_view{view_id}'
        render_folder.mkdir(parents=True, exist_ok=True)
        gt_folder.mkdir(parents=True, exist_ok=True)
        view_folders[view_id] = {'render': render_folder, 'gt': gt_folder}

    # Check if we already loaded the saved trajectory
    if 'saved_trajectory' not in locals():
        # Need to load or generate trajectory for rendering
        if os.path.exists(pred_traj_path):
            print(f"Using saved trajectory for rendering")
            saved_trajectory = np.load(pred_traj_path)
        else:
            print(f"Generating trajectory for rendering")
            simulator = MultiObjectSimulator(estimation_params, vol, object_labels, config_file)
            saved_trajectory = []
            simulator.initialize()
            for f in range(30):
                xyz = simulator.forward(f)
                saved_trajectory.append(xyz.detach().cpu().numpy())
            saved_trajectory = np.array(saved_trajectory)

    train_psnr_list = []
    train_ssim_list = []
    test_psnr_list = []
    test_ssim_list = []
    with torch.no_grad():
        max_f = min(gs_args.predict_frames, len(saved_trajectory))
        print(f"\nRendering {max_f} frames...")
        # Debug: Check view fids
        unique_fids = sorted(list(set([view.fid for view in views])))
        print(f"Unique frame IDs in views: {unique_fids[:10]}...")  # Show first 10

        for f in range(max_f):
            xyz = torch.from_numpy(saved_trajectory[f]).to('cuda', dtype=torch.float32)
            curr_views = [view for view in views if torch.abs(view.fid - f / estimation_params.fps) < 1e-4]
            if f == 0:  # Debug first frame
                print(f"Frame {f}: Looking for fid={f / estimation_params.fps:.4f}, found {len(curr_views)} views")
                if curr_views:
                    print(f"  Matching views have UIDs: {[v.uid for v in curr_views]}")
            d_xyz = xyz - scene.gaussians.get_xyz.detach()
            for view in curr_views:
                # Isotropic Gaussians (num_attribute=4)
                results = render(view, scene.gaussians, pipeline, background, d_xyz, 0.0, 0.0, False)
                # Anisotropic Gaussians (num_attribute=10) - fallback
                # num_gaussians = scene.gaussians._xyz.shape[0]
                # d_rotation = torch.zeros((num_gaussians, 4), device='cuda')
                # d_scaling = torch.zeros((num_gaussians, 3), device='cuda')
                # results = render(view, scene.gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, False)
                image = results["render"]
                gt_image = view.original_image.cuda()
                # Save images for all views in their respective folders
                torchvision.utils.save_image(image, view_folders[view.uid]['render'] / f'{f:05d}.png')
                torchvision.utils.save_image(gt_image, view_folders[view.uid]['gt'] / f'{f:05d}.png')
                # Also save view 3 to the original folders for backward compatibility
                if view.uid == 3:
                    torchvision.utils.save_image(image, model_path / f'{obj_name}_img_render' / f'{f:05d}.png')
                    torchvision.utils.save_image(gt_image, model_path / f'{obj_name}_img_gt' / f'{f:05d}.png')
                # Compute metrics for both train and test frames
                curr_psnr = psnr(image, gt_image)
                curr_ssim = ssim(image, gt_image)
                if f < gs_args.train_frames:
                    train_psnr_list.append(curr_psnr)
                    train_ssim_list.append(curr_ssim)
                else:
                    test_psnr_list.append(curr_psnr)
                    test_ssim_list.append(curr_ssim)

    # Compute mean metrics
    train_psnr = torch.mean(torch.stack(train_psnr_list)) if train_psnr_list else torch.tensor(0.0)
    train_ssim = torch.mean(torch.stack(train_ssim_list)) if train_ssim_list else torch.tensor(0.0)
    test_psnr = torch.mean(torch.stack(test_psnr_list)) if test_psnr_list else torch.tensor(0.0)
    test_ssim = torch.mean(torch.stack(test_ssim_list)) if test_ssim_list else torch.tensor(0.0)

    # 1. Use saved trajectory for evaluation
    if 'saved_trajectory' in locals():
        print("Using saved trajectory for evaluation")
        seq = [torch.from_numpy(saved_trajectory[f]).to('cuda', dtype=torch.float32) for f in range(len(saved_trajectory))]
    else:
        # Fallback to generating new trajectory if needed
        if 'simulator' not in locals():
            simulator = MultiObjectSimulator(estimation_params, vol, object_labels, config_file)
        seq = gen_xyz_list(simulator, gs_args.predict_frames, diff=False, save_ply=True, path=dataset.model_path)

    # 2. Load Gt
    gts = load_gt_pcds(gs_args.gt_path)

    # 3.evaluate CD loss & EMD loss
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    cd_train, cd_test = evaluate(seq, gts, gs_args.train_frames, 'CD')
    emd_train, emd_test = evaluate(seq, gts, gs_args.train_frames, 'EMD')

    print("\n" + "="*60)
    print("FINAL METRICS SUMMARY")
    print("="*60)
    print(f'Dataset: {obj_name}')
    print(f'Train frames: 0-{gs_args.train_frames-1}, Test frames: {gs_args.train_frames}-{gs_args.predict_frames-1}')
    print(f'-------------------------')
    print(f'PSNR - Train: {train_psnr:.4f}, Test: {test_psnr:.4f}')
    print(f'SSIM - Train: {train_ssim:.4f}, Test: {test_ssim:.4f}')
    print(f'CD   - Train: {cd_train:.4f}, Test: {cd_test:.4f}')
    print(f'EMD  - Train: {emd_train:.4f}, Test: {emd_test:.4f}')
    print("="*60 + "\n")

    # Save metrics to file
    metrics_data = {
        'dataset': obj_name,
        'train_frames': gs_args.train_frames,
        'test_frames': gs_args.predict_frames - gs_args.train_frames,
        'total_frames': gs_args.predict_frames,
        'metrics': {
            'train': {
                'psnr': float(train_psnr),
                'ssim': float(train_ssim),
                'cd': float(cd_train),
                'emd': float(emd_train)
            },
            'test': {
                'psnr': float(test_psnr),
                'ssim': float(test_ssim),
                'cd': float(cd_test),
                'emd': float(emd_test)
            }
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    metrics_file = model_path / f'{obj_name}_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Metrics saved to: {metrics_file}")

    # Also save a human-readable log file
    log_file = model_path / f'{obj_name}_metrics.log'
    with open(log_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("FINAL METRICS SUMMARY\n")
        f.write("="*60 + "\n")
        f.write(f'Dataset: {obj_name}\n')
        f.write(f'Timestamp: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Train frames: 0-{gs_args.train_frames-1}, Test frames: {gs_args.train_frames}-{gs_args.predict_frames-1}\n')
        f.write('-------------------------\n')
        f.write(f'PSNR - Train: {train_psnr:.4f}, Test: {test_psnr:.4f}\n')
        f.write(f'SSIM - Train: {train_ssim:.4f}, Test: {test_ssim:.4f}\n')
        f.write(f'CD   - Train: {cd_train:.4f}, Test: {cd_test:.4f}\n')
        f.write(f'EMD  - Train: {emd_train:.4f}, Test: {emd_test:.4f}\n')
        f.write("="*60 + "\n")
    print(f"Log saved to: {log_file}")

    # Create videos for each view
    print("\nGenerating videos for each view...")


    # This function will run ffmpeg and provide detailed error feedback
    def create_video(input_pattern, output_path, framerate=30):
        command = [
            'ffmpeg', '-y', '-framerate', str(framerate),
            '-i', str(input_pattern),
            '-c:v', 'mpeg4',  # <-- Use the more common MPEG4 encoder
            '-q:v', '2',     # Add a quality setting (lower is better)
            str(output_path)
        ]

        print(f"    Executing FFmpeg command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"    ❌ FFmpeg failed to create {output_path.name}")
            # Print the last few lines of the error log, which usually has the key info
            error_lines = result.stderr.strip().split('\n')
            print("    --- FFmpeg Error Log (last 5 lines) ---")
            for line in error_lines[-5:]:
                print(f"    {line}")
            print("    -----------------------------------------")
            return False
        else:
            print(f"    ✅ Successfully created {output_path.name}")
            return True


    # --- Loop 1: Process per-view folders ---
    for view_id in unique_view_ids:
        print(f"  Processing view {view_id}...")
        render_folder = view_folders[view_id]['render']
        gt_folder = view_folders[view_id]['gt']

        # Create render video
        if render_folder.exists() and any(render_folder.glob('*.png')):
            # Use an absolute path for the input pattern
            input_pattern = render_folder / "%05d.png"
            output_video = render_folder / f'render_view{view_id}.mp4'
            create_video(input_pattern, output_video)
        else:
            print(f"    No PNG files found in {render_folder}")

        # Create GT video
        if gt_folder.exists() and any(gt_folder.glob('*.png')):
            input_pattern = gt_folder / "%05d.png"
            output_video = gt_folder / f'gt_view{view_id}.mp4'
            create_video(input_pattern, output_video)
        else:
            print(f"    No PNG files found in {gt_folder}")

    # --- Loop 2: Process backward compatibility folders ---
    print("  Processing view 3 (backward compatibility)...")
    render_path = model_path / f'{obj_name}_img_render'
    gt_path = model_path / f'{obj_name}_img_gt'

    if render_path.exists() and any(render_path.glob('*.png')):
        input_pattern = render_path / "%05d.png"
        output_video = render_path / 'render.mp4'
        create_video(input_pattern, output_video)
    else:
        print(f"    No PNG files found in backward compat render folder: {render_path}")

    if gt_path.exists() and any(gt_path.glob('*.png')):
        input_pattern = gt_path / "%05d.png"
        output_video = gt_path / 'gt.mp4'
        create_video(input_pattern, output_video)
    else:
        print(f"    No PNG files found in backward compat GT folder: {gt_path}")

    print("\nVideo generation complete!")
