
import os
import torch
import trimesh
import torchvision
import numpy as np
from tqdm import tqdm
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.system_utils import check_gs_model
from utils.image_utils import psnr
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud


def load_initial_pcd_from_gt(source_path, num_objects=2, frame_idx=0, device="cuda"):
    """
    从 data/03/point_clouds/{0,1}/{frame_idx}.ply 读两个 object 的点云，
    合成一个初始点云 xyz (Tensor[N,3])，用于 fixed pcd 训练。
    """
    all_pts = []

    for obj_id in range(num_objects):
        ply_path = os.path.join(source_path, "point_clouds", str(obj_id), f"{frame_idx}.ply")
        if not os.path.exists(ply_path):
            print(f"[WARN] Missing {ply_path}, skip this object.")
            continue

        mesh = trimesh.load(ply_path, process=False)
        pts = np.asarray(mesh.vertices, dtype=np.float32)
        if pts.shape[0] == 0:
            print(f"[WARN] Empty point cloud: {ply_path}")
            continue

        all_pts.append(pts)

    if len(all_pts) == 0:
        raise RuntimeError(f"No valid point cloud found in {source_path}/point_clouds/*/{frame_idx}.ply")

    xyz = np.concatenate(all_pts, axis=0)
    xyz = torch.from_numpy(xyz).to(device)
    print(f"[INFO] Loaded initial PCD from GT: {xyz.shape[0]} points (frame {frame_idx}, {num_objects} objects).")
    return xyz


def train(dataset, opt, pipe, testing_iterations, saving_iterations, pcd, d_xyz_list=None, fps=24, cam_info=None, grid_size=0.12):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, resolution_scales=[dataset.res_scale], pcd=pcd, load_fix_pcd=True, cam_info=cam_info)
    if d_xyz_list:
        scene.clipTrainCamerasbyframes(len(d_xyz_list))
        scene.clipTestCamerasbyframes(len(d_xyz_list))

    gaussians.training_setup(opt, fix_pcd=True)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training fixed pcd progress")
    print(f"reg alpha {opt.reg_alpha}")
    print(f"reg scale {opt.reg_scale}")
    for iteration in range(1, opt.iterations + 1):

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            views = scene.getTrainCameras(scale=dataset.res_scale).copy()
            if iteration < opt.warm_up or d_xyz_list is None:
                # viewpoint_stack = views
                warm_up_fid = torch.unique(torch.stack([view.fid for view in views]))[0]
                viewpoint_stack = [view for view in views if view.fid == warm_up_fid]
            else:
                viewpoint_stack = views

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            if d_xyz_list is None:
                d_xyz = 0.0
            else:
                d_xyz = d_xyz_list[int(fid/(1/fps))]
            d_rotation, d_scaling = 0.0, 0.0
                
        # Render
        render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
        image, viewspace_point_tensor, visibility_filter, radii, alpha = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"], render_pkg_re["alpha"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if opt.reg_alpha:
            # gt_op_mask = (torch.var(gt_image, dim=0) > 0.001).to(torch.float32)
            # gt_op_mask = (gt_image.sum(dim=0) > 0.0).to(torch.float32)
            L_alpha = l1_loss(alpha, viewpoint_cam.gt_alpha_mask)
            loss += L_alpha

        if iteration >= opt.warm_up:
            if opt.reg_scale:
                scales = gaussians.get_scaling
                diff_scales = torch.nn.functional.relu(scales-grid_size/16) + torch.nn.functional.relu(grid_size/32-scales)
                L_scale = torch.mean(diff_scales) * 10
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
            torch.cuda.empty_cache()
            cur_psnr = psnr_report(iteration, l1_loss,
                                    testing_iterations, scene, render, (pipe, background), 
                                    dataset.load2gpu_on_the_fly, dataset.res_scale, dataset.is_6dof, d_xyz_list, fps)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, fix_pcd=True)
            
            if iteration % 1000 == 0: 
                vertex_colors = np.concatenate([(gaussians.get_opacity.detach().cpu().numpy()*255).astype(np.uint8)]*3, axis=1)
                file_name = f'gs/fix_gs_{iteration}.ply'
                if iteration < opt.warm_up:
                    xyz0 = gaussians.get_xyz
                else:
                    if d_xyz_list is None:
                        d_xyz = 0.0
                    else:
                        d_xyz = d_xyz_list[int(torch.round(fid/(1/fps)))]
                    d_rotation, d_scaling = 0.0, 0.0
                    xyz0 = gaussians.get_xyz + d_xyz
                    trimesh.Trimesh(
                        xyz0.detach().cpu().numpy(), 
                        vertex_colors=vertex_colors, 
                        ).export(os.path.join(dataset.model_path, file_name))
            
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                gaussians.optimizer.zero_grad(set_to_none=True)

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


def psnr_report(iteration, l1_loss, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, load2gpu_on_the_fly, res_scale=1, is_6dof=False, d_xyz_list=None, fps=24):

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=res_scale)},
                              {'name': 'train',
                               'cameras': scene.getTrainCameras(scale=res_scale)})

        for config in validation_configs:
            min_fid = min([view.fid for view in config['cameras']])
            cams = [view for view in config['cameras'] if view.fid == min_fid] if d_xyz_list is None else config['cameras'] 
            if config['cameras'] and len(cams) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                
                for idx, viewpoint in enumerate(cams):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    if d_xyz_list is None:
                        d_xyz = 0.0
                    else:
                        d_xyz = d_xyz_list[int(torch.round(fid/(1/fps)))]
                    d_rotation, d_scaling = 0.0, 0.0
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof)["render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test), flush=True)

        torch.cuda.empty_cache()

    return test_psnr

def train_gs_with_fixed_pcd(xyz, dataset, opt, pipe, testing_iterations, saving_iterations, d_xyz_list, fps, force_train=False, cam_info=None, grid_size=0.12):
    xyz = xyz.cpu().detach().numpy()
    num_pts= xyz.shape[0]
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    if (not check_gs_model(dataset.model_path, saving_iterations, fix_pcd=True)) or force_train:
        train(dataset, opt, pipe, testing_iterations, saving_iterations, pcd, d_xyz_list=d_xyz_list, fps=fps, cam_info=cam_info, grid_size=grid_size)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=opt.iterations, shuffle=False, resolution_scales=[1.0], load_fix_pcd=True, cam_info=cam_info)
    return scene


def assign_gs_to_pcd(xyz, xyz_opacity, dataset, opt, pipe, cam_info, grid_size=0.12, scene=None):
    # TODO remove useless params
    xyz = xyz.cpu().detach().numpy()
    num_pts= xyz.shape[0]
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    gaussians = GaussianModel(dataset.sh_degree)
    if scene is None:
        scene = Scene(dataset, gaussians, resolution_scales=[1.0], pcd=pcd, cam_info=cam_info)
    xyz_opacity = xyz_opacity.reshape(-1, 1)
    scene.gaussians._opacity = torch.nn.Parameter(scene.gaussians.inverse_opacity_activation(torch.clamp(xyz_opacity, max=1-1e-4)).requires_grad_(True))
    scales = torch.ones_like(xyz_opacity) * grid_size / 32 * 0.5
    scene.gaussians._scaling = torch.nn.Parameter(scene.gaussians.scaling_inverse_activation(scales).requires_grad_(True))
    
    # Store corrected assignments if available
    if 'corrected_assignments' in cam_info and cam_info['corrected_assignments'] is not None:
        scene.gaussians._corrected_assignments = cam_info['corrected_assignments']

    # Update object probabilities based on particle object labels
    if 'object_labels' in cam_info and cam_info['object_labels'] is not None:
        object_labels = cam_info['object_labels']
        if torch.is_tensor(object_labels):
            object_labels = object_labels.cpu().numpy()
        
        # Create one-hot encoding for object probabilities: background (0) + K objects (baselines)
        from scene.gaussian_model import num_object_channels
        n_ch = max(int(num_object_channels()), int(np.max(object_labels)) + 1 if len(object_labels) else 1)
        object_probs = np.zeros((num_pts, n_ch), dtype=np.float32)
        labels = np.zeros(num_pts, dtype=np.int64)  # default to background
        m = min(num_pts, len(object_labels))
        labels[:m] = np.asarray(object_labels[:m], dtype=np.int64)
        object_probs[np.arange(num_pts), labels] = 1.0
        
        # Convert probabilities to logits and set in the Gaussian model
        # The Gaussian model expects _object_logits and computes probabilities via softmax
        object_logits = np.log(object_probs + 1e-8)  # Convert to log space (logits)
        scene.gaussians._object_logits = torch.nn.Parameter(
            torch.tensor(object_logits, dtype=torch.float32, device="cuda").requires_grad_(True)
        )
    
    return scene


if __name__ == "__main__":
    from argparse import ArgumentParser
    from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args

    parser = ArgumentParser(description="Train fixed Gaussian model from GT multi-object point clouds")

    # 这三个类会往 parser 里加一堆参数：
    # 比如 --source_path (-s), --model_path (-m), --iterations, --save_iterations, --test_iterations 等
    model_group = ModelParams(parser)
    pipeline_group = PipelineParams(parser)
    opt_group = OptimizationParams(parser)

    # 额外加两个和 GT 点云相关的简单参数
    parser.add_argument("--num_objects", type=int, default=2,
                        help="Number of objects in data/<id>/point_clouds")
    parser.add_argument("--pcd_frame", type=int, default=0,
                        help="Which frame index of GT point clouds to use as initial PCD (default: 0)")

    # 用项目里的工具函数，把 config["gs"] / config["physics"] 合并到 argparse 里
    gs_args, phys_args = get_combined_args(parser)

    # 用 gs_args 来构造 dataset / pipeline / opt（这一步会把 config 的值带进去）
    dataset = model_group.extract(gs_args)
    pipe = pipeline_group.extract(gs_args)
    opt = opt_group.extract(gs_args)

    # 路径：一般在命令行 -s / -m 里给，或在 config 的 gs 里给
    source_path = dataset.source_path     # 对应 data/03
    model_path = dataset.model_path       # 对应 selected_gic_pc_dataset_45_new/03

    print(f"[INFO] Source path: {source_path}")
    print(f"[INFO] Model path:  {model_path}")

    # ✅ 从 config.gs 里拿 test_iterations 和 save_iterations
    # 也就是你 JSON 里那几个：
    #   "test_iterations": [5000, 6000, 7000]
    #   "save_iterations": [7000, 10000]
    testing_iterations = getattr(gs_args, "test_iterations", [opt.iterations])
    saving_iterations = getattr(gs_args, "save_iterations", [opt.iterations])
    # FPS 从 physics 里拿（你 JSON 里 physics.fps = 24）
    fps = getattr(phys_args, "fps", 24)

    # 看你要不要支持 --force_train 这种 flag，没有就默认 False
    force_train = getattr(gs_args, "force_train", False)

    # ===== 1. 从 GT 的 point_clouds/{0,1}/{pcd_frame}.ply 合成初始点云 =====
    xyz = load_initial_pcd_from_gt(
        source_path,
        num_objects=gs_args.num_objects if hasattr(gs_args, "num_objects") else 2,
        frame_idx=gs_args.pcd_frame if hasattr(gs_args, "pcd_frame") else 0,
        device="cuda"
    )

    # ===== 2. 用 train_gs_with_fixed_pcd 做静态重建，保存到 -m 路径下 =====
    _ = train_gs_with_fixed_pcd(
        xyz=xyz,
        dataset=dataset,
        opt=opt,
        pipe=pipe,
        testing_iterations=testing_iterations,
        saving_iterations=saving_iterations,
        d_xyz_list=None,   # 静态重建，不用时序 deform
        fps=fps,
        force_train=force_train,
        cam_info=None,
        grid_size=phys_args.density_grid_size if hasattr(phys_args, "density_grid_size") else 0.12,
    )

    print("[INFO] Fixed PCD Gaussian training finished.")
    print("[INFO] You can now run train_dynamic_MO_LQR.py for system identification.")