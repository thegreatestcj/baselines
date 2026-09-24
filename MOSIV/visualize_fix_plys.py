from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import os
import json
import numpy as np
import open3d as o3d
import torch
import warp as wp
from diff_gauss import GaussianRasterizer as Renderer
from omegaconf import OmegaConf
from tqdm import tqdm, trange
import re

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import Scene, DeformModel
from simulator.estimator import Estimator
from train_gs_fixed_pcd import train_gs_with_fixed_pcd
from utils.general_utils import safe_state
from utils.reg_utils import build_rotation, quat_mult
from visualizer.colormap import colormap
from visualizer.helpers import setup_camera

import time
import os, imageio, cv2

RENDER_MODE = "color"  # 'color', 'depth' or 'centers'
ADDITIONAL_LINES = "trajectories"  # None, 'trajectories' or 'rotations'
REMOVE_BACKGROUND = False  # False or True
FORCE_LOOP = False  # False or True

w, h = 800, 800
near, far = 0.01, 100.0
view_scale = 1.0
fps = 10
traj_frac = 200 # 1% of points
traj_length = 1
def_pix = (
    torch.tensor(np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
)
pix_ones = torch.ones(h * w, 1).cuda().float()

image_scale = 1.0


def init_camera(y_angle=0.0, center_dist=2.4, cam_height=1.3, f_ratio=0.82):
    ry = y_angle * np.pi / 180
    w2c = np.array(
        [
            [np.cos(ry), 0.0, -np.sin(ry), 0.0],
            [0.0, 1.0, 0.0, cam_height],
            [np.sin(ry), 0.0, np.cos(ry), center_dist],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
    return w2c, k


def extract_views(cameras):
    """Extract views function (exactly from predict.py)"""
    views = cameras.copy()
    fids = torch.unique(torch.stack([view.fid for view in views]))
    views = {i: [view for view in views if view.fid == fids[i]] for i in range(len(fids))}
    views = {i: sorted(v, key=lambda x: x.colmap_id) for i, v in views.items()}
    return views


def _sorted_ply_list(ply_dir: str):
    p = Path(ply_dir)
    assert p.exists() and p.is_dir(), f"ply_dir does not exist or is not a directory: {ply_dir}"
    files = list(p.glob("*.ply"))
    assert len(files) > 0, f"No .ply files found in {ply_dir}"

    def key_fn(fp: Path):
        m = re.fullmatch(r"(\d+)\.ply", fp.name)
        if m:
            return int(m.group(1))
        m2 = re.search(r"(\d+)", fp.stem)
        return int(m2.group(1)) if m2 else fp.name
    return sorted(files, key=key_fn)


def _load_ply_as_torch(path: Path, device: torch.device):
    pcd = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(pcd.points, dtype=np.float32)
    assert xyz.size > 0, f"{path} is an empty point cloud"
    pos = torch.from_numpy(xyz).to(device=device, dtype=torch.float32).contiguous()
    return pos


def common_appearance_training_pipeline(dat, opt, ppl, phy, arg, pos0, d_xyz_list):
    """
    A common pipeline that takes a base shape and displacements,
    trains the GS appearance, and returns renderable scene data.
    """
    print("Training appearance with train_gs_with_fixed_pcd...")

    # Force black background for training
    dat.white_background = False

    scene = train_gs_with_fixed_pcd(
        pos0,
        dat,
        opt,
        ppl,
        arg.test_iterations + list(range(10000, 40001, 1000)),
        arg.save_iterations,
        d_xyz_list,
        getattr(phy, 'fps', 24),
        force_train=arg.force_train,
        grid_size=getattr(phy, 'density_grid_size', 0.12),
    )

    if getattr(arg, "split", "train") == "train":
        views = scene.getTrainCameras(scale=dat.res_scale)
    elif getattr(arg, "split", "test") == "test":
        views = scene.getTestCameras(scale=dat.res_scale)
    else:
        raise NotImplementedError

    xyz_canonical = scene.gaussians.get_xyz.detach()
    is_fg = torch.ones_like(xyz_canonical, dtype=torch.bool)[..., 0]
    scene_data = []

    with torch.no_grad():
        for frame_idx in tqdm(range(len(d_xyz_list)), desc="Creating scene data"):
            d_xyz = d_xyz_list[frame_idx]

            if d_xyz.shape[0] != xyz_canonical.shape[0]:
                print(f"[Warning] Mismatch between trajectory points ({d_xyz.shape[0]}) and Gaussian points ({xyz_canonical.shape[0]}). Resizing.")
                if d_xyz.shape[0] < xyz_canonical.shape[0]:
                    pad = xyz_canonical.shape[0] - d_xyz.shape[0]
                    d_xyz = torch.cat([d_xyz, torch.zeros(pad, 3, device=d_xyz.device, dtype=d_xyz.dtype)], dim=0)
                else:
                    d_xyz = d_xyz[:xyz_canonical.shape[0]]

            rendervar = {
                "means3D": xyz_canonical + d_xyz,
                "means2D": torch.zeros_like(xyz_canonical),
                "shs": scene.gaussians.get_features,
                "colors_precomp": None,
                "rotations": scene.gaussians.get_rotation,
                "opacities": scene.gaussians.get_opacity,
                "scales": scene.gaussians.get_scaling.repeat(1, 3),
            }
            if REMOVE_BACKGROUND:
                rendervar = {k: v[is_fg] for k, v in rendervar.items()}
            scene_data.append(rendervar)

    if REMOVE_BACKGROUND:
        is_fg = is_fg[is_fg]

    print(f"Scene data created with {len(scene_data)} frames")
    return scene_data, is_fg, views


def load_data_ply(dat, opt, ppl, phy, arg, ply_dir: str):
    """Loads a sequence of .ply files, trains appearance, and returns scene data."""
    print(f"Loading PLY sequence from: {ply_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = _sorted_ply_list(ply_dir)
    print(f"Found {len(files)} frames, first={files[0].name}, last={files[-1].name}")

    pos0 = _load_ply_as_torch(files[0], device=device)
    N0 = pos0.shape[0]
    print(f"Frame 0 has {N0} points (canonical).")

    total_frames = min(getattr(phy, "n_frames", len(files)), len(files))
    print(f"Using total_frames = {total_frames}")

    d_xyz_list = [torch.zeros_like(pos0)]
    for i in range(1, total_frames):
        pos_t = _load_ply_as_torch(files[i], device=device)
        if pos_t.shape[0] != N0:
            raise ValueError(f"Frame {files[i].name} has {pos_t.shape[0]} points, but frame 0 has {N0}. Point counts must match.")
        d_xyz_list.append(pos_t - pos0)

    return common_appearance_training_pipeline(dat, opt, ppl, phy, arg, pos0, d_xyz_list)


def load_data_npy(dat, opt, ppl, phy, arg, npy_path: str):
    """Loads a single .npy trajectory file from train_dynamic_MO, trains appearance, and returns scene data."""
    print(f"Loading NPY trajectory from: {npy_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trajectory = np.load(npy_path)
    print(f"Loaded trajectory with shape: {trajectory.shape} (frames, particles, 3)")

    if trajectory.ndim != 3 or trajectory.shape[2] != 3:
        raise ValueError("NPY array must have shape (num_frames, num_particles, 3).")

    trajectory_torch = torch.from_numpy(trajectory).to(device=device, dtype=torch.float32).contiguous()

    pos0 = trajectory_torch[0]
    N0 = pos0.shape[0]
    print(f"Frame 0 has {N0} particles (canonical).")

    # Use all frames from the trajectory
    total_frames = len(trajectory_torch)
    print(f"Using all {total_frames} frames from trajectory, FPS = {getattr(phy, 'fps', 24)}")

    d_xyz_list = []
    for i in range(total_frames):
        d_xyz_list.append(trajectory_torch[i] - pos0)

    return common_appearance_training_pipeline(dat, opt, ppl, phy, arg, pos0, d_xyz_list)


# ... (The rest of the functions like make_lineset, render_timestep, etc., remain unchanged) ...
def make_lineset_from_accum(pts, base_cols, num_lines):
    lineset = o3d.geometry.LineSet()
    pts = np.ascontiguousarray(pts, np.float64)
    lineset.points = o3d.utility.Vector3dVector(pts)

    pt_indices = np.arange(len(pts))
    line_indices = np.stack((pt_indices, pt_indices - num_lines), -1)[num_lines:]
    lineset.lines = o3d.utility.Vector2iVector(line_indices.astype(np.int32))

    t = len(pts) // num_lines - 1
    if t > 0:
        cols_lines = np.tile(base_cols, (t, 1)).astype(np.float64)
    else:
        cols_lines = np.zeros((0, 3), dtype=np.float64)
    lineset.colors = o3d.utility.Vector3dVector(cols_lines)
    return lineset


def calculate_trajectories(scene_data, is_fg):
    in_pts = [data["means3D"][is_fg][::traj_frac].contiguous().float().cpu().numpy() for data in scene_data]
    num_lines = len(in_pts[0])

    base_cols = colormap[np.arange(num_lines) % len(colormap)].astype(np.float64)
    if base_cols.max() > 1.0:
        base_cols /= 255.0

    linesets = []
    accum = []
    for t in range(len(in_pts)):
        accum.append(in_pts[t])
        pts_t = np.reshape(np.array(accum), (-1, 3))
        linesets.append(make_lineset_from_accum(pts_t, base_cols, num_lines))
    return linesets


def render_timestep(w2c, k, timestep_data, background=None):
    with torch.no_grad():
        cam = setup_camera(w, h, k, w2c, near, far)
        # Use black background by default
        if background is None:
            background = torch.zeros(3, device='cuda')
        im, depth, alpha, radii = Renderer(raster_settings=cam)(**timestep_data)
        # Apply background
        im = im + (1 - alpha) * background.view(3, 1, 1)
        return im, depth


def rgbd2pcd(im, depth, w2c, k, show_depth=False, project_to_cam_w_scale=None):
    d_near = 1.5
    d_far = 6
    invk = torch.inverse(torch.tensor(k).cuda().float())
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    radial_depth = depth[0].reshape(-1)
    def_rays = (invk @ def_pix.T).T
    def_radial_rays = def_rays / torch.linalg.norm(def_rays, ord=2, dim=-1)[:, None]
    pts_cam = def_radial_rays * radial_depth[:, None]
    z_depth = pts_cam[:, 2]
    if project_to_cam_w_scale is not None:
        pts_cam = project_to_cam_w_scale * pts_cam / z_depth[:, None]
    pts4 = torch.concat((pts_cam, pix_ones), 1)
    pts = (c2w @ pts4.T).T[:, :3]
    if show_depth:
        cols = ((z_depth - d_near) / (d_far - d_near))[:, None].repeat(1, 3)
    else:
        cols = torch.permute(im, (1, 2, 0)).reshape(-1, 3)
    pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())
    cols = o3d.utility.Vector3dVector(cols.contiguous().double().cpu().numpy())
    return pts, cols


if __name__ == "__main__":
    parser = ArgumentParser(description="Trajectory visualization from PLY or NPY data.")
    # --- Add arguments for data source ---
    parser.add_argument("--ply_dir", type=str, default=None,
                        help="Directory containing a sequence of .ply files (e.g., 0.ply, 1.ply...).")
    parser.add_argument("--npy_path", type=str, default=None,
                        help="Path to a single .npy file with shape (frames, points, 3).")

    # --- Other existing arguments ---
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--load_iter", type=int, default=0)
    parser.add_argument("--force_train", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for rendered videos and frames.")

    mdl = ModelParams(parser)
    ppl = PipelineParams(parser)
    opt = OptimizationParams(parser)
    arg, phy = get_combined_args(parser)

    # Preserve the visualization-specific arguments from command line
    cmdline_args = parser.parse_args()
    arg.ply_dir = cmdline_args.ply_dir
    arg.npy_path = cmdline_args.npy_path
    arg.out_dir = cmdline_args.out_dir if cmdline_args.out_dir else arg.model_path
    # Convert force_train from int to bool
    arg.force_train = (cmdline_args.force_train != 0)

    dat = mdl.extract(arg)
    opt = opt.extract(arg)
    ppl = ppl.extract(arg)

    print("Config:", getattr(arg, 'config_path', 'Not specified'))
    print("Data:", arg.source_path)
    print("Output:", arg.model_path)

    safe_state(arg.quiet)
    wp.init()
    wp.config.verify_cuda = True
    wp.ScopedTimer.enabled = False
    wp.set_module_options({"fast_math": False})

    # --- Determine which data loader to use ---
    if arg.ply_dir and arg.npy_path:
        raise ValueError("Please provide either --ply_dir or --npy_path, not both.")
    elif arg.ply_dir:
        scene_data, is_fg, views = load_data_ply(dat, opt, ppl, phy, arg, arg.ply_dir)
    elif arg.npy_path:
        scene_data, is_fg, views = load_data_npy(dat, opt, ppl, phy, arg, arg.npy_path)
    else:
        raise ValueError("You must provide a trajectory data source: either --ply_dir or --npy_path.")

    if not views:
        raise RuntimeError("No camera views were loaded. Check your --source_path and dataset configuration.")

    # --- Group cameras by view ID ---
    cameras_by_view = {}
    for view in views:
        uid = view.uid
        if uid not in cameras_by_view:
            cameras_by_view[uid] = []
        cameras_by_view[uid].append(view)

    print(f"\nFound {len(cameras_by_view)} unique camera views to render.")
    base_out_dir = Path(arg.out_dir)

    # --- Main Multi-View Rendering Loop ---
    for view_id in sorted(cameras_by_view.keys()):
        view_cameras = cameras_by_view[view_id]
        print(f"\n--- Processing View {view_id} ---")

        # Use the first camera of this view for intrinsics and a stable viewpoint
        cam = view_cameras[0]
        w2c, k = cam.world_view_transform.cpu().numpy().T, cam.intrinsic

        # --- Setup view-specific output directories ---
        view_out_dir = base_out_dir / f"view_{view_id}"


        def _ensure(p):
            os.makedirs(p, exist_ok=True)


        rgb_dir = view_out_dir / "rgb_frames";
        _ensure(rgb_dir)
        depth_dir = view_out_dir / "depth_frames";
        _ensure(depth_dir)
        depthvis_dir = view_out_dir / "depthvis_frames";
        _ensure(depthvis_dir)

        rgb_png = lambda i: str(rgb_dir / f"rgb_{i:05d}.png")
        depth_png = lambda i: str(depth_dir / f"depth_mm_{i:05d}.png")
        depthvis_png = lambda i: str(depthvis_dir / f"depth_vis_{i:05d}.png")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        rgb_mp4_path = str(view_out_dir / "rgb_traj.mp4")
        depth_vis_mp4_path = str(view_out_dir / "depth_traj.mp4")
        rgb_vw = None
        depth_vw = None

        # --- Reset trajectory trails for each new view ---
        mask = is_fg.detach().cpu().numpy().astype(bool)
        all_idx = np.where(mask)[0]
        pt_idx = all_idx[::traj_frac] if len(all_idx) > 0 else np.arange(0, scene_data[0]["means3D"].shape[0],
                                                                         traj_frac)
        num_traj_pts = len(pt_idx)
        base_cols = (colormap[np.arange(num_traj_pts) % len(colormap)] * 255.0).astype(np.uint8)
        traj_uvs = [[] for _ in range(num_traj_pts)]


        def _to_rgb_u8(im_tensor, keep_black=True):
            rgb = (im_tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
            if not keep_black:
                bg = (rgb[:, :, 0] < 5) & (rgb[:, :, 1] < 5) & (rgb[:, :, 2] < 5)
                rgb[bg] = 255
            return rgb


        def _depth_u16_mm(depth_tensor):
            d = depth_tensor[0].detach().cpu().numpy()
            return np.clip(d * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)


        def _depth_vis_u8(depth_tensor, d_near=0.1, d_far=10.0):
            d = depth_tensor[0].detach().cpu().numpy()
            x = np.clip((d - d_near) / (d_far - d_near), 0.0, 1.0)
            g = (x * 255.0 + 0.5).astype(np.uint8)
            return np.stack([g, g, g], axis=-1)


        def _project_uv(w2c_np, k_np, pts3d_np):
            N = pts3d_np.shape[0]
            pts_h = np.concatenate([pts3d_np, np.ones((N, 1), dtype=np.float32)], axis=1)
            Xc = (w2c_np @ pts_h.T).T[:, :3]
            z = Xc[:, 2];
            valid_z = z > 1e-6
            u = k_np[0, 0] * (Xc[:, 0] / z) + k_np[0, 2]
            v = k_np[1, 1] * (Xc[:, 1] / z) + k_np[1, 2]
            return np.stack([u, v], 1), valid_z


        background = torch.zeros(3, device='cuda')
        render_fps = getattr(phy, 'fps', 10)

        for t in trange(len(scene_data), desc=f"Rendering View {view_id}"):
            im, depth = render_timestep(w2c, k, scene_data[t], background)

            rgb = _to_rgb_u8(im, keep_black=True)
            depth_u16 = _depth_u16_mm(depth)
            depth_vis = _depth_vis_u8(depth)

            H, W = rgb.shape[:2]
            pts3d = scene_data[t]["means3D"][pt_idx].detach().float().cpu().numpy()
            uv, valid = _project_uv(w2c, k, pts3d)
            for i in range(num_traj_pts):
                if valid[i]:
                    u, v = uv[i]
                    if 0 <= u < W and 0 <= v < H:
                        traj_uvs[i].append((int(round(u)), int(round(v))))

            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
            depth_vis = np.ascontiguousarray(depth_vis, dtype=np.uint8)

            for i in range(num_traj_pts):
                if len(traj_uvs[i]) >= 2:
                    bgr = (int(base_cols[i, 2]), int(base_cols[i, 1]), int(base_cols[i, 0]))
                    poly = np.array(traj_uvs[i], dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(rgb, [poly], False, bgr, 1, cv2.LINE_AA)
                    cv2.polylines(depth_vis, [poly], False, bgr, 1, cv2.LINE_AA)

            cv2.imwrite(rgb_png(t), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(depth_png(t), depth_u16)
            cv2.imwrite(depthvis_png(t), cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))

            if rgb_vw is None:
                rgb_vw = cv2.VideoWriter(rgb_mp4_path, fourcc, render_fps, (W, H), True)
            if depth_vw is None:
                depth_vw = cv2.VideoWriter(depth_vis_mp4_path, fourcc, render_fps, (W, H), True)
            rgb_vw.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            depth_vw.write(cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))

        if rgb_vw is not None: rgb_vw.release()
        if depth_vw is not None: depth_vw.release()

        print(f"  ✅ View {view_id} export complete.")
        print(f"     - RGB Video: {rgb_mp4_path}")
        print(f"     - Depth Video: {depth_vis_mp4_path}")

    print("\nAll rendering tasks finished.")