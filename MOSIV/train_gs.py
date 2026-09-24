#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import time 
import torch
import torchvision
from random import randint

from sympy import false

from utils.loss_utils import l1_loss, ssim, kl_divergence
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
from utils.reg_utils import o3d_knn, mini_batch_knn, quat_mult, build_rotation, weighted_l2_loss_v2
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
import trimesh
import cv2

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def save_deformed_trajectory(gaussians, deform, scene, dataset, iteration, start_frame=0, end_frame=30):
    """
    Save deformed Gaussian positions for trajectory visualization during training.

    Args:
        gaussians: GaussianModel object
        deform: DeformModel object
        scene: Scene object
        dataset: ModelParams object
        iteration: Current training iteration
        start_frame: Starting frame index
        end_frame: Ending frame index
    """
    trajectory_dir = os.path.join(dataset.model_path, f'trajectory/iteration_{iteration}')
    os.makedirs(trajectory_dir, exist_ok=True)

    xyz_canonical = gaussians.get_xyz.detach()
    opacity = gaussians.get_opacity.detach().cpu().numpy()

    # Filter out background Gaussians (object_id=0) if object probabilities exist
    if hasattr(gaussians, 'get_object_probs'):
        print("Filtering out background Gaussians from trajectory...")
        object_probs = gaussians.get_object_probs.detach()  # [N, 3]
        object_assignments = torch.argmax(object_probs, dim=1)  # [N] with values 0,1,2

        # Keep only object 1 and object 2 (exclude background with id=0)
        object_mask = (object_assignments > 0)  # True for obj1 and obj2

        # Apply filter
        xyz_canonical_filtered = xyz_canonical[object_mask]
        opacity_filtered = opacity[object_mask.cpu().numpy()]

        # Create colors based on object IDs
        obj1_mask = (object_assignments == 1)[object_mask]
        obj2_mask = (object_assignments == 2)[object_mask]

        vertex_colors = np.zeros((len(xyz_canonical_filtered), 3), dtype=np.uint8)
        vertex_colors[obj1_mask.cpu().numpy()] = [255, 200, 40]  # Object 1: yellow/gold
        vertex_colors[obj2_mask.cpu().numpy()] = [1, 68, 33]     # Object 2: green

        print(f"  Kept {len(xyz_canonical_filtered)}/{len(xyz_canonical)} Gaussians")
        print(f"  Object 1: {obj1_mask.sum().item()} Gaussians")
        print(f"  Object 2: {obj2_mask.sum().item()} Gaussians")

        xyz_canonical = xyz_canonical_filtered
        opacity = opacity_filtered
    else:
        # No object probabilities, use opacity for coloring as before
        vertex_colors = np.concatenate([(opacity * 255).astype(np.uint8)] * 3, axis=1)

    # Save canonical state
    trimesh.Trimesh(
        xyz_canonical.cpu().numpy(),
        vertex_colors=vertex_colors
    ).export(os.path.join(trajectory_dir, 'gs_canonical.ply'))
    
    # Save deformed states for all frames
    deform_stats = []
    views = scene.getTrainCameras(scale=1.0)
    unique_times = torch.unique(torch.stack([view.fid for view in views])).sort()[0]

    for frame_idx in range(start_frame, min(end_frame + 1, len(unique_times))):
        # Use actual time value, not frame index
        time_input = unique_times[frame_idx].unsqueeze(0).unsqueeze(0)  # Shape: [1, 1]
        with torch.no_grad():
            d_xyz, d_rotation, d_scaling = deform.step(xyz_canonical, time_input)
        
        xyz_deformed = xyz_canonical + d_xyz
        
        # Save deformed Gaussians
        trimesh.Trimesh(
            xyz_deformed.cpu().numpy(),
            vertex_colors=vertex_colors
        ).export(os.path.join(trajectory_dir, f'gs_frame_{frame_idx:04d}.ply'))
        
        # Collect statistics
        deform_mag = torch.norm(d_xyz, dim=1)
        deform_stats.append({
            'frame': frame_idx,
            'mean': deform_mag.mean().item(),
            'max': deform_mag.max().item(),
            'std': deform_mag.std().item()
        })
    
    # Save statistics summary
    with open(os.path.join(trajectory_dir, 'deformation_stats.txt'), 'w') as f:
        f.write(f"Iteration {iteration} - Deformation Statistics\n")
        f.write(f"{'Frame':<10} {'Mean':<15} {'Max':<15} {'Std':<15}\n")
        f.write("-" * 55 + "\n")
        for stat in deform_stats:
            f.write(f"{stat['frame']:<10} {stat['mean']:<15.6f} {stat['max']:<15.6f} {stat['std']:<15.6f}\n")
    
    print(f"[ITER {iteration}] Saved deformed trajectory to {trajectory_dir}")



def save_training_video(gaussians, deform, scene, dataset, pipe, background, iteration, fps=30):
    """
    Save comparison videos of GT vs rendered for all camera views in a 4x4 grid.
    
    Args:
        gaussians: GaussianModel object
        deform: DeformModel object
        scene: Scene object
        dataset: ModelParams object
        pipe: Pipeline parameters
        background: Background tensor
        iteration: Current training iteration
        fps: Frames per second for output video
    """
    from gaussian_renderer import render
    
    # Create subfolder for this iteration
    video_dir = os.path.join(dataset.model_path, 'gs_training_video', f'iter_{iteration:06d}')
    os.makedirs(video_dir, exist_ok=True)
    
    # Get all training views
    views = scene.getTrainCameras(scale=dataset.res_scale)
    
    # Group views by camera ID
    cam_views = {}
    for view in views:
        cam_id = view.uid
        if cam_id not in cam_views:
            cam_views[cam_id] = []
        cam_views[cam_id].append(view)
    
    # Sort views by frame time within each camera
    for cam_id in cam_views:
        cam_views[cam_id].sort(key=lambda x: x.fid)
    
    # Get sorted camera IDs
    sorted_cam_ids = sorted(cam_views.keys())
    num_cams = len(sorted_cam_ids)
    
    if num_cams == 0:
        return
    
    # Create 4x4 grid layout (up to 16 cameras, 2 columns per camera for GT/Rendered)
    grid_rows = min(4, (num_cams + 1) // 2)  # Max 4 rows
    grid_cols = 4  # 2 cameras per row, each with GT and Rendered
    
    # Get dimensions from first camera
    first_view = cam_views[sorted_cam_ids[0]][0]
    img_height = first_view.image_height
    img_width = first_view.image_width
    
    # Add space for labels
    label_height = 30
    cell_height = img_height + label_height
    cell_width = img_width
    
    # Create video writer for grid
    grid_height = cell_height * grid_rows
    grid_width = cell_width * grid_cols
    output_path = os.path.join(video_dir, f'grid_comparison.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (grid_width, grid_height))
    
    # Get max frames across all cameras
    max_frames = max(len(views_list) for views_list in cam_views.values())
    
    # Process each frame
    for frame_idx in range(max_frames):
        # Create grid frame
        grid_frame = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        # Fill grid with camera views
        for cam_idx, cam_id in enumerate(sorted_cam_ids[:grid_rows * 2]):
            if cam_idx >= len(cam_views):
                break
                
            row = cam_idx // 2
            col_offset = (cam_idx % 2) * 2  # 0 or 2
            
            views_list = cam_views[cam_id]
            if frame_idx >= len(views_list):
                # Use last frame if this camera has fewer frames
                view = views_list[-1]
            else:
                view = views_list[frame_idx]
            
            # Load GT image
            if dataset.load2gpu_on_the_fly:
                view.load2device()
            
            # Get deformation for this frame
            if iteration < 3000:  # warm_up period
                d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
            else:
                time_input = view.fid.unsqueeze(0)
                with torch.no_grad():
                    d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input)
            
            # Render
            with torch.no_grad():
                render_pkg = render(view, gaussians, pipe, background, 
                                   d_xyz, d_rotation, d_scaling, dataset.is_6dof)
                rendered_image = render_pkg["render"]
            
            # Convert to numpy arrays
            gt_np = view.original_image.cpu().permute(1, 2, 0).numpy()
            gt_np = (np.clip(gt_np, 0, 1) * 255).astype(np.uint8)
            gt_bgr = cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR)
            
            rendered_np = rendered_image.cpu().permute(1, 2, 0).numpy()
            rendered_np = (np.clip(rendered_np, 0, 1) * 255).astype(np.uint8)
            rendered_bgr = cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR)
            
            # Add GT image to grid (left column of pair)
            y_start = row * cell_height + label_height
            y_end = y_start + img_height
            x_start_gt = col_offset * cell_width
            x_end_gt = x_start_gt + cell_width
            
            # Create labeled GT image
            gt_cell = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
            gt_cell[label_height:, :] = gt_bgr
            fid_value = view.fid.item() if torch.is_tensor(view.fid) else view.fid
            cv2.putText(gt_cell, f"Cam{cam_id} GT F:{fid_value:.2f}", (5, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Create labeled Rendered image
            rendered_cell = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
            rendered_cell[label_height:, :] = rendered_bgr
            cv2.putText(rendered_cell, f"Cam{cam_id} Render", (5, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Place in grid
            grid_frame[row * cell_height:(row + 1) * cell_height, 
                      col_offset * cell_width:(col_offset + 1) * cell_width] = gt_cell
            grid_frame[row * cell_height:(row + 1) * cell_height, 
                      (col_offset + 1) * cell_width:(col_offset + 2) * cell_width] = rendered_cell
        
        # Add iteration info
        cv2.putText(grid_frame, f"Iteration: {iteration}", (grid_width - 150, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        video_writer.write(grid_frame)
    
    video_writer.release()


def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel(dataset)
    deform.train_setting(opt)
    scene = Scene(dataset, gaussians, resolution_scales=[dataset.res_scale])
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
    neighbor_sq_dist, neighbor_indices = None, None
    update_knn = True
    for iteration in range(1, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            views = scene.getTrainCameras(scale=dataset.res_scale).copy()
            if iteration < opt.warm_up:
                # viewpoint_stack = views
                warm_up_fid = torch.unique(torch.stack([view.fid for view in views]))[0]
                viewpoint_stack = [view for view in views if view.fid == warm_up_fid]
            else:
                viewpoint_stack = views

        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0)

            ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda') * time_interval * smooth_term(iteration)
            d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise)

        # Render
        render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
        image, viewspace_point_tensor, visibility_filter, radii, alpha = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"], render_pkg_re["alpha"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        # mask = torch.logical_or(gt_image.sum(0) != 0,
        #                         image.sum(0) != 0)
        # ids = torch.where(mask)
        # h_min, h_max, w_min, w_max = ids[0].min(), ids[0].max(), ids[1].min(), ids[1].max()
        # h_min, h_max = max(h_min-50, 0), min(h_max+50, image.shape[1])
        # w_min, w_max = max(w_min-50, 0), min(w_max+50, image.shape[2])
        # image = image[:, h_min:h_max, w_min:w_max]
        # gt_image = gt_image[:, h_min:h_max, w_min:w_max]
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if opt.reg_alpha:
            # l1 loss
            L_alpha = l1_loss(alpha, viewpoint_cam.gt_alpha_mask)
            # cross entropy loss
            # alpha = torch.clamp(alpha, 1e-4, 1-1e-4)
            # L_alpha = torch.mean(-viewpoint_cam.gt_alpha_mask * torch.log(alpha) - (1 - viewpoint_cam.gt_alpha_mask) * torch.log(1 - alpha)) * 0.05
            # weighted loss
            # flag = viewpoint_cam.gt_alpha_mask == 1
            # not_flag = torch.logical_not(flag)
            # weighted l1 loss
            # pos_a = torch.mean(torch.abs(alpha[flag] - viewpoint_cam.gt_alpha_mask[flag]))
            # neg_a = torch.mean(torch.abs(alpha[not_flag] - viewpoint_cam.gt_alpha_mask[not_flag]))
            # L_alpha = 0.5 * (pos_a + neg_a)
            # weighted cross entropy
            # alpha = torch.clamp(alpha, 1e-4, 1-1e-4)
            # pos_a = torch.mean(-torch.log(alpha[flag]))
            # neg_a = torch.mean(-torch.log(1 - alpha[not_flag]))
            # L_alpha = 0.5 * (pos_a + neg_a) * 0.1
            loss += L_alpha
        if iteration >= opt.warm_up:

            if opt.reg_rigid:
                xyz_t = gaussians.get_xyz + d_xyz
                rotations_t = gaussians.get_rotation + d_rotation
                inv_rotations_t = rotations_t  # just indicate that it's the inverse of rotation
                inv_rotations_t[:, 1:] = -1 * inv_rotations_t[:, 1:]
                d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + 0.01)
                xyz_t_1 = gaussians.get_xyz + d_xyz
                rotations_t_1 = gaussians.get_rotation + d_rotation
                rel_rot = quat_mult(rotations_t_1, inv_rotations_t)
                rot = build_rotation(rel_rot)
                if neighbor_sq_dist is None or update_knn:
                    neighbor_sq_dist, neighbor_indices = mini_batch_knn(xyz_t.detach(), xyz_t.detach(), opt.num_knn)
                    # neighbor_sq_dist, neighbor_indices = o3d_knn(xyz_t.detach().cpu().numpy(), opt.num_knn)
                    # neighbor_sq_dist = torch.from_numpy(neighbor_sq_dist).to(xyz_t)
                    # neighbor_indices = torch.from_numpy(neighbor_indices).to('cuda').to(torch.int64)
                    neighbor_weight = torch.exp(-2000 * neighbor_sq_dist)
                    update_knn = False
                curr_neighbor_pts = xyz_t[neighbor_indices]
                curr_offset = curr_neighbor_pts - xyz_t[:, None]
                next_neighbor_pts = xyz_t_1[neighbor_indices]
                next_offset = next_neighbor_pts - xyz_t_1[:, None]
                next_offset_in_curr_coord = (rot.transpose(2, 1)[:, None] @ next_offset[:, :, :, None]).squeeze(-1)
                L_rigid = weighted_l2_loss_v2(next_offset_in_curr_coord, curr_offset, neighbor_weight)
                loss = loss + L_rigid
            if opt.reg_scale:
                scales = gaussians.get_scaling + d_scaling
                L_scale = torch.mean(torch.abs(scales)) * 10
                # if iteration < opt.densify_until_iter:
                #     diff_scales = torch.nn.functional.relu(scales-1e-4) + torch.nn.functional.relu(0-scales)
                # else:
                #     diff_scales = torch.nn.functional.relu(0-scales) # torch.nn.functional.relu(0-scales)
                # diff_scales = torch.nn.functional.relu(scales-1e-4) + torch.nn.functional.relu(0-scales)
                # L_scale = torch.mean(diff_scales) * 10
                loss = loss + L_scale
            if opt.reg_tgs and iteration > opt.warm_up * 5:
                depth = render_pkg_re["depth"][0]
                xyz_t = gaussians.get_xyz + d_xyz
                gs_w, gs_h, gs_d = viewpoint_cam.pw2pix(xyz_t)
                in_mask = viewpoint_cam.is_in_view(gs_w, gs_h)
                render_pix_d = depth[gs_h[in_mask], gs_w[in_mask]]
                diff_depth = gs_d[in_mask] - render_pix_d
                depth_mask = torch.logical_and(opt.tgs_bound >= diff_depth, render_pix_d > 0)
                if depth_mask.sum() == 0:
                    L_tgs = 0.0
                else:
                    L_tgs = torch.mean(torch.abs(diff_depth[depth_mask])) * 0.1
                loss = loss + L_tgs
                gaussians.add_diff_depth_stats(diff_depth.detach(), in_mask)
        loss.backward()

        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            # Log and save
            cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render, (pipe, background), deform,
                                       dataset.load2gpu_on_the_fly, dataset.res_scale, dataset.is_6dof)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(dataset.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent*0.5, size_threshold)
                    update_knn = True
                    # print(gaussians.get_xyz.shape[0])

                if iteration > opt.warm_up and iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if opt.reg_tgs and opt.warm_up * 5 < iteration <= opt.tgs_densify_until_iter and iteration % (opt.densification_interval*5) == 0:
                gaussians.truncate_gs(opt.tgs_bound)
                gaussians.densify_surface(opt.tgs_bound, max_screen_size=opt.tgs_max_screen_size)
                print(gaussians.get_xyz.shape[0])

            if iteration % 1000 == 0:
                vertex_colors = np.concatenate([(gaussians.get_opacity.detach().cpu().numpy()*255).astype(np.uint8)]*3, axis=1)
                if iteration < opt.warm_up:
                    xyz0 = gaussians.get_xyz
                else:
                    xyz0 = gaussians.get_xyz+deform.step(gaussians.get_xyz, torch.tensor([[0.0]]).to('cuda'))[0]  # 0.5416666666666666
                trimesh.Trimesh(
                    xyz0.detach().cpu().numpy(),
                    vertex_colors=vertex_colors,
                    ).export(os.path.join(dataset.model_path, f'gs/gs_{iteration}.ply'))
                cat_img = torch.cat([image, gt_image], dim=2)
                torchvision.utils.save_image(cat_img, os.path.join(dataset.model_path, f'img/ren_gt_{iteration}.png'))
                cat_mask = torch.cat([alpha, viewpoint_cam.gt_alpha_mask], dim=2)
                torchvision.utils.save_image(cat_mask, os.path.join(dataset.model_path, f'img/mask_ren_gt_{iteration}.png'))

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                deform.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)
            
            # Save deformed trajectory every 10k iterations
            if iteration % 10000 == 0 and iteration > 0:
                save_deformed_trajectory(gaussians, deform, scene, dataset, iteration, 
                                        start_frame=0, end_frame=30)
                # Also save colored Gaussians for all frames

            # Save training videos every 5k iterations
            if iteration % 5000 == 0 and iteration > 0:
                save_training_video(gaussians, deform, scene, dataset, pipe, background, iteration)

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


def training_MO(dataset, opt, pipe, testing_iterations, saving_iterations):
    """Modified training function with multi-object mask support"""
    tb_writer = prepare_output_and_logger(dataset)
    
    # Print important training parameters
    print("\n" + "="*60)
    print("TRAINING PARAMETERS")
    print("="*60)
    print(f"Total iterations: {opt.iterations}")
    print(f"Warm-up iterations: {opt.warm_up}")
    print(f"Densification:")
    print(f"  - Start: {opt.densify_from_iter}")
    print(f"  - End: {opt.densify_until_iter}")
    print(f"  - Interval: {opt.densification_interval}")
    print(f"  - Gradient threshold: {opt.densify_grad_threshold}")
    print(f"Opacity reset interval: {opt.opacity_reset_interval}")
    print(f"TGS regularization: {opt.reg_tgs}")
    if opt.reg_tgs:
        print(f"  - TGS densify until: {opt.tgs_densify_until_iter}")
        print(f"  - TGS bound: {opt.tgs_bound}")
    print(f"Learning rates:")
    print(f"  - Position: {opt.position_lr_init} -> {opt.position_lr_final}")
    print(f"  - Feature: {opt.feature_lr}")
    print(f"  - Opacity: {opt.opacity_lr}")
    print(f"  - Scaling: {opt.scaling_lr}")
    print(f"  - Rotation: {opt.rotation_lr}")
    print("="*60 + "\n")
    
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel(dataset)
    deform.train_setting(opt)
    scene = Scene(dataset, gaussians, resolution_scales=[dataset.res_scale])
    gaussians.training_setup(opt)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Multi-object parameters from optimization args
    enable_mask_training = getattr(opt, 'enable_mask_training', True)
    mask_loss_weight = getattr(opt, 'mask_loss_weight', 0.5)
    
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_mask_loss_for_log = 0.0
    best_psnr = 0.0
    best_mask_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress (MO)")
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
    neighbor_sq_dist, neighbor_indices = None, None
    update_knn = True
    
    for iteration in range(1, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
        
        iter_start.record()
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        # Pick a random Camera
        if not viewpoint_stack:
            views = scene.getTrainCameras(scale=dataset.res_scale).copy()
            if iteration < opt.warm_up:
                warm_up_fid = torch.unique(torch.stack([view.fid for view in views]))[0]
                viewpoint_stack = [view for view in views if view.fid == warm_up_fid]
            else:
                viewpoint_stack = views
        
        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame
        
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid
        
        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0)
            ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda') * time_interval * smooth_term(iteration)
            d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise)
        
        # Check if we should render masks - same as features, always when available
        render_mask_flag = (enable_mask_training and
                           hasattr(viewpoint_cam, 'object_mask') and 
                           viewpoint_cam.object_mask is not None)
        # Render with mask support when available

        if render_mask_flag:
            render_pkg_re = render(viewpoint_cam, gaussians, pipe, background,
                                  d_xyz, d_rotation, d_scaling, dataset.is_6dof,
                                  render_object_mask=True)
        else:
            # Call exactly like original training when not doing mask rendering
            render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, 
                                  d_xyz, d_rotation, d_scaling, dataset.is_6dof)
        
        image, viewspace_point_tensor, visibility_filter, radii, alpha = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"], render_pkg_re["alpha"]
        
        # RGB Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # Alpha Loss
        if opt.reg_alpha:
            L_alpha = l1_loss(alpha, viewpoint_cam.gt_alpha_mask)
            loss += L_alpha
        
        # Object Mask Loss
        mask_loss_val = 0.0
        mask_psnr = 0.0
        if render_mask_flag and "mask" in render_pkg_re:
            rendered_mask = render_pkg_re["mask"]
            # Handle both numpy and tensor cases
            if isinstance(viewpoint_cam.object_mask, np.ndarray):
                gt_mask = torch.from_numpy(viewpoint_cam.object_mask).cuda().float()
                gt_mask = gt_mask.permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]
            else:
                # Already a tensor from Camera class
                gt_mask = viewpoint_cam.object_mask.cuda().float()
                if gt_mask.dim() == 3 and gt_mask.shape[-1] == 3:
                    gt_mask = gt_mask.permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]
            
            # Use same loss structure as RGB: L1 + SSIM
            mask_l1 = l1_loss(rendered_mask, gt_mask)
            mask_ssim = ssim(rendered_mask, gt_mask)
            mask_loss = (1.0 - opt.lambda_dssim) * mask_l1 + opt.lambda_dssim * (1.0 - mask_ssim)
            mask_loss_val = mask_loss.item()
            
            # Calculate mask PSNR
            with torch.no_grad():
                mask_mse = ((rendered_mask - gt_mask) ** 2).mean()
                mask_psnr = 10 * torch.log10(1.0 / mask_mse).item() if mask_mse > 0 else 40.0
            
            loss += mask_loss_weight * mask_loss
        
        # Add regularization losses if needed (rigid, scale, tgs)
        if iteration >= opt.warm_up:
            if opt.reg_rigid:
                xyz_t = gaussians.get_xyz + d_xyz
                rotations_t = gaussians.get_rotation + d_rotation
                inv_rotations_t = rotations_t
                inv_rotations_t[:, 1:] = -1 * inv_rotations_t[:, 1:]
                d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + 0.01)
                xyz_t_1 = gaussians.get_xyz + d_xyz
                rotations_t_1 = gaussians.get_rotation + d_rotation
                rel_rot = quat_mult(rotations_t_1, inv_rotations_t)
                rot = build_rotation(rel_rot)
                if neighbor_sq_dist is None or update_knn:
                    neighbor_sq_dist, neighbor_indices = mini_batch_knn(xyz_t.detach(), xyz_t.detach(), opt.num_knn)
                    neighbor_weight = torch.exp(-2000 * neighbor_sq_dist)
                    update_knn = False
                curr_neighbor_pts = xyz_t[neighbor_indices]
                curr_offset = curr_neighbor_pts - xyz_t[:, None]
                next_neighbor_pts = xyz_t_1[neighbor_indices]
                next_offset = next_neighbor_pts - xyz_t_1[:, None]
                next_offset_in_curr_coord = (rot.transpose(2, 1)[:, None] @ next_offset[:, :, :, None]).squeeze(-1)
                L_rigid = weighted_l2_loss_v2(next_offset_in_curr_coord, curr_offset, neighbor_weight)
                loss = loss + L_rigid
            
            if opt.reg_scale:
                scales = gaussians.get_scaling + d_scaling
                L_scale = torch.mean(torch.abs(scales)) * 10
                loss = loss + L_scale
            
            if opt.reg_tgs and iteration > opt.warm_up * 5:
                depth = render_pkg_re["depth"][0]
                xyz_t = gaussians.get_xyz + d_xyz
                gs_w, gs_h, gs_d = viewpoint_cam.pw2pix(xyz_t)
                in_mask = viewpoint_cam.is_in_view(gs_w, gs_h)
                render_pix_d = depth[gs_h[in_mask], gs_w[in_mask]]
                diff_depth = gs_d[in_mask] - render_pix_d
                depth_mask = torch.logical_and(opt.tgs_bound >= diff_depth, render_pix_d > 0)
                if depth_mask.sum() == 0:
                    L_tgs = 0.0
                else:
                    L_tgs = torch.mean(torch.abs(diff_depth[depth_mask])) * 0.1
                loss = loss + L_tgs
                gaussians.add_diff_depth_stats(diff_depth.detach(), in_mask)
        
        loss.backward()
        
        iter_end.record()
        
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if mask_loss_val > 0:
                ema_mask_loss_for_log = 0.4 * mask_loss_val + 0.6 * ema_mask_loss_for_log
            
            if iteration % 10 == 0:
                # Calculate RGB PSNR for display
                rgb_mse = ((image - gt_image) ** 2).mean()
                rgb_psnr = 10 * torch.log10(1.0 / rgb_mse).item() if rgb_mse > 0 else 40.0
                
                # Update progress bar with both RGB and mask metrics
                if mask_loss_val > 0:
                    progress_bar.set_postfix({
                        "Loss": f"{ema_loss_for_log:.4f}",
                        "RGB_PSNR": f"{rgb_psnr:.2f}",
                        "Mask_Loss": f"{ema_mask_loss_for_log:.4f}",
                        "Mask_PSNR": f"{mask_psnr:.2f}"
                    })
                else:
                    progress_bar.set_postfix({
                        "Loss": f"{ema_loss_for_log:.4f}",
                        "RGB_PSNR": f"{rgb_psnr:.2f}"
                    })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            
            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])
            
            # Log and save
            cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                      testing_iterations, scene, render, (pipe, background), deform,
                                      dataset.load2gpu_on_the_fly, dataset.res_scale, dataset.is_6dof)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration
            
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(dataset.model_path, iteration)
            
            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent*0.5, size_threshold)
                    update_knn = True
            
                if iteration > opt.warm_up and iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if opt.reg_tgs and opt.warm_up * 5 < iteration <= opt.tgs_densify_until_iter and iteration % (opt.densification_interval*5) == 0:
                gaussians.truncate_gs(opt.tgs_bound)
                gaussians.densify_surface(opt.tgs_bound, max_screen_size=opt.tgs_max_screen_size)
                print(gaussians.get_xyz.shape[0])
            
            # Debug visualization every 1000 iterations (same as original training function)
            if iteration % 1000 == 0:
                import torchvision
                import trimesh
                
                vertex_colors = np.concatenate([(gaussians.get_opacity.detach().cpu().numpy()*255).astype(np.uint8)]*3, axis=1)
                if iteration < opt.warm_up:
                    xyz0 = gaussians.get_xyz
                else:
                    xyz0 = gaussians.get_xyz + deform.step(gaussians.get_xyz, torch.tensor([[0.0]]).to('cuda'))[0]
                trimesh.Trimesh(
                    xyz0.detach().cpu().numpy(),
                    vertex_colors=vertex_colors,
                ).export(os.path.join(dataset.model_path, f'gs/gs_{iteration}.ply'))
                cat_img = torch.cat([image, gt_image], dim=2)
                torchvision.utils.save_image(cat_img, os.path.join(dataset.model_path, f'img/ren_gt_{iteration}.png'))
                cat_mask = torch.cat([alpha, viewpoint_cam.gt_alpha_mask], dim=2)
                torchvision.utils.save_image(cat_mask, os.path.join(dataset.model_path, f'img/mask_ren_gt_{iteration}.png'))
                
                # Object mask comparison (new for multi-object)
                if render_mask_flag and "mask" in render_pkg_re:
                    rendered_mask = render_pkg_re["mask"]  # [3, H, W]
                    if isinstance(viewpoint_cam.object_mask, np.ndarray):
                        gt_mask = torch.from_numpy(viewpoint_cam.object_mask).cuda().float()
                        gt_mask = gt_mask.permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]
                    else:
                        # Already a tensor from Camera class
                        gt_mask = viewpoint_cam.object_mask.cuda().float()
                        if gt_mask.dim() == 3 and gt_mask.shape[-1] == 3:
                            gt_mask = gt_mask.permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]
                    
                    # Save individual object masks
                    cat_obj1 = torch.cat([rendered_mask[1:2], gt_mask[1:2]], dim=2)  # Object 1
                    torchvision.utils.save_image(cat_obj1, os.path.join(dataset.model_path, f'img/obj1_mask_ren_gt_{iteration}.png'))
                    
                    cat_obj2 = torch.cat([rendered_mask[2:3], gt_mask[2:3]], dim=2)  # Object 2  
                    torchvision.utils.save_image(cat_obj2, os.path.join(dataset.model_path, f'img/obj2_mask_ren_gt_{iteration}.png'))
                    
                    # Combined RGB visualization: obj1=red, obj2=green, bg=blue
                    rendered_mask_rgb = torch.stack([
                        rendered_mask[1],  # Red channel: Object 1
                        rendered_mask[2],  # Green channel: Object 2
                        rendered_mask[0],  # Blue channel: Background
                    ])
                    gt_mask_rgb = torch.stack([
                        gt_mask[1],  # Red channel: Object 1
                        gt_mask[2],  # Green channel: Object 2
                        gt_mask[0],  # Blue channel: Background
                    ])
                    cat_mask_rgb = torch.cat([rendered_mask_rgb, gt_mask_rgb], dim=2)
                    torchvision.utils.save_image(cat_mask_rgb, os.path.join(dataset.model_path, f'img/mask_rgb_ren_gt_{iteration}.png'))
            
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                deform.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)

            # Save deformed trajectory every 10k iterations
            if iteration % 10000 == 0 and iteration > 0:
                save_deformed_trajectory(gaussians, deform, scene, dataset, iteration,
                                         start_frame=0, end_frame=30)
                # Also save colored Gaussians for all frames
                gaussians.save_ply(path="/ssd2/yizhou/outputs/GIC_MO/01_debug/gs_colored/gs.ply")

            # Save training videos every 5k iterations
            if iteration % 5000 == 0 and iteration > 0:
                save_training_video(gaussians, deform, scene, dataset, pipe, background, iteration)
    
    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(os.path.join(args.model_path, f'gs'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, f'img'), exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly, res_scale=1, is_6dof=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=res_scale)},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras(scale=res_scale)[idx % len(scene.getTrainCameras(scale=res_scale))] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    # time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    time_input = fid.unsqueeze(0)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof)["render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test), flush=True)
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            # tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000] + list(range(10000, 40001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    b = time.perf_counter()
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations)
    e = time.perf_counter()
    # All done
    print(f"\nTraining complete, time elapsed: {(e-b)/60} mins.")
