#!/usr/bin/env python
"""OmniPhysGS system identification on a PhysON scene (baselines addition; upstream main.py untouched).

Keeps OmniPhysGS's model — a physics-guided network that assigns every particle one expert
elasticity and one expert plasticity model (straight-through hard softmax) driving the pure-PyTorch
MPM — and replaces the video-diffusion SDS objective by supervision from the observed multi-view
frames of the scene. Additions over upstream: learnable expert parameters (E, nu, sigma_y, ...),
a learnable rigid initial velocity, the scene's declared external force field, a separable floor
and an explicit world<->sim scaling with all quantities converted (see README_baselines.md).

    python fit.py --config <pkg>/config.yaml --tag <name> [--gpu N] [--output outputs] [--eval_only] [k=v ...]

<pkg> is a package written by ../eval/convert_physon_to_omniphysgs.py, with gs_model/ from
recon_static.py. Extra `k=v` arguments are OmegaConf dot-list overrides of config.yaml
(e.g. `train.epochs=1 sim.force_mode=none material.init_E=8e4`). Outputs go to
<output>/<tag>/ (particles/<f>.ply in world metres, renders_test/, renders_test_alpha/, gt_test/,
material.ply, params.json, metrics.json, video_test.mp4, log.txt, checkpoints/, tensorboard).
The run is resumable: the latest checkpoints/epoch_XXXX.pth (or velocity.pth) is picked up on
restart, and a finished training goes straight to the final free rollout. Run from the OmniPhysGS
directory or anywhere (paths on the command line are resolved before the script chdirs here).
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.checkpoint import checkpoint

HERE = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="<pkg>/config.yaml (from eval/convert_physon_to_omniphysgs.py)")
    p.add_argument("--tag", default=None, help="run name; outputs go to <output>/<tag>/")
    p.add_argument("--gpu", type=int, default=None, help="index among the visible devices")
    p.add_argument("--output", default=None, help="output root (default: config train.export_path = ./outputs)")
    p.add_argument("--eval_only", action="store_true", help="skip training; free rollout with the latest checkpoint (or the initial parameters)")
    p.add_argument("--no_video", action="store_true")
    p.add_argument("overrides", nargs="*", help="OmegaConf dot-list overrides, e.g. train.epochs=1")
    return p.parse_args()


# ----------------------------------------------------------------------------------------------
# helpers

def log(msg, fh=None):
    print(msg, flush=True)
    if fh is not None:
        fh.write(msg + "\n")
        fh.flush()


def chamfer_sq_mm2(a, b, samples=8192, seed=0):
    """Bidirectional mean squared NN distance in 10^3 mm^2 (inputs in metres; same convention and
    subsampling as eval/metrics.py so numbers are comparable)."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    if len(a) > samples:
        a = a[rng.choice(len(a), samples, replace=False)]
    if len(b) > samples:
        b = b[rng.choice(len(b), samples, replace=False)]
    da = cKDTree(b).query(a, k=1)[0]
    db = cKDTree(a).query(b, k=1)[0]
    return float(((da ** 2).mean() + (db ** 2).mean()) * 1e6 / 1e3)


def chamfer_torch(a, b, samples=4096, gen=None):
    """Differentiable symmetric squared Chamfer distance on random subsamples (sim or world units)."""
    if a.shape[0] > samples:
        a = a[torch.randperm(a.shape[0], device=a.device, generator=gen)[:samples]]
    if b.shape[0] > samples:
        b = b[torch.randperm(b.shape[0], device=b.device, generator=gen)[:samples]]
    d = torch.cdist(a, b)
    return d.min(1).values.pow(2).mean() + d.min(0).values.pow(2).mean()


def psnr_t(a, b):
    return float(-10.0 * torch.log10(((a - b) ** 2).mean()))


def polar_rotation(F, iters=8):
    """Rotation factor R of F = R S by the Newton iteration X <- (X + X^-T)/2 (Higham), batched [N,3,3].
    Unlike an SVD-based U V^T this is smooth under autograd at F = I (repeated singular values), which
    is the state of every particle at frame 0 / at rest. det(F) > 0 assumed (F is clamped by the MPM)."""
    eye = torch.eye(3, device=F.device, dtype=F.dtype).expand_as(F)
    # particles whose F has collapsed (det ~ 0, e.g. an unstable expert) get R = I instead of a
    # singular inversion; their state is meaningless anyway and nan_to_num'd gradients handle the rest
    ok = (torch.isfinite(F).flatten(1).all(1) & (torch.linalg.det(F).abs() > 1e-6)).view(-1, 1, 1)
    X = torch.where(ok, F, eye)
    X = X * (math.sqrt(3.0) / X.norm(dim=(1, 2), keepdim=True).clamp_min(1e-6))
    for _ in range(iters):
        try:
            Xi = torch.linalg.inv(X)
        except RuntimeError:  # an iterate went exactly singular (blown-up particle): pseudo-inverse
            Xi = torch.linalg.pinv(X)
        X = 0.5 * (X + Xi.transpose(1, 2))
    good = ok & torch.isfinite(X).flatten(1).all(1).view(-1, 1, 1)
    return torch.where(good, X, eye)


PALETTE = np.array([[230, 40, 40], [40, 200, 220], [60, 180, 75], [240, 180, 30], [150, 60, 200],
                    [255, 120, 0], [0, 100, 200], [120, 120, 120]], dtype=np.uint8)


# ----------------------------------------------------------------------------------------------

class Fitter:
    def __init__(self, cfg, config_dir: Path, args):
        self.cfg = cfg
        self.args = args
        self.device = torch.device(f"cuda:{cfg.train.gpu}")
        torch.cuda.set_device(cfg.train.gpu)

        # ---- output dir / logging
        out_root = Path(cfg.train.export_path)
        tag = cfg.train.train_tag or time.strftime("%Y%m%d_%H_%M_%S")
        self.out = out_root / tag
        for d in ("checkpoints", "particles", "renders_test", "renders_test_alpha", "gt_test"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.logf = open(self.out / "log.txt", "a")
        with open(self.out / "config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(str(self.out / "tb"))
        self.timings = {}
        log(f"[fit] output {self.out}", self.logf)

        # ---- seeds
        seed = int(cfg.train.seed)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        self.gen = torch.Generator(device=self.device); self.gen.manual_seed(seed)

        # ---- warp / taichi (as upstream main.py)
        import warp as wp
        wp.init(); wp.ScopedTimer.enabled = False; wp.set_module_options({"fast_math": False})
        self.filling = cfg.preprocessing.get("particle_filling", None)
        if self.filling is not None:
            import taichi as ti
            ti.init(arch=ti.cuda, device_memory_GB=min(8.0, float(os.environ.get("TI_DEVICE_MEMORY_GB", 8))))

        # ---- package / scene
        from src.utils.physon_scene import PhysonPackage
        pkg_dir = (config_dir / str(cfg.scene.package)).resolve()
        if not (pkg_dir / "scene.json").exists():
            raise FileNotFoundError(f"{pkg_dir}/scene.json missing: run eval/convert_physon_to_omniphysgs.py first")
        self.pkg = PhysonPackage(pkg_dir)
        self.n_frames_total = int(self.pkg.meta.get("n_frames_total", self.pkg.n_frames))
        self.n_frames_train = min(int(cfg.sim.n_frames), self.n_frames_total)
        self.frame_dt = float(cfg.sim.frame_dt)
        self.test_cam_ids = [int(c) for c in cfg.eval.test_cam_ids] or self.pkg.test_cam_ids
        self.train_cam_ids = [c for c in sorted(self.pkg.cameras) if c not in self.test_cam_ids]
        self.rgb_exclude = set(int(c) for c in (cfg.train.get("rgb_exclude_cams", None) or self.pkg.meta.get("rgb_exclude_cams", []) or []))
        if self.rgb_exclude:
            log(f"[fit] train cams {sorted(self.rgb_exclude)} have a black object silhouette: alpha loss only (no RGB term)", self.logf)
        if self.pkg.meta.get("rigid_bodies"):
            log("[warn] scene has rigid bodies (rigid_bodies.json); OmniPhysGS has no rigid coupling — running without them", self.logf)

        t0 = time.time()
        self._load_gt()
        self._load_gaussians()
        self._preprocess()
        self._build_sim()
        self._build_material()
        self._init_velocity()
        self.timings["setup_s"] = time.time() - t0
        log(f"[fit] setup {self.timings['setup_s']:.1f}s", self.logf)

    # ------------------------------------------------------------------ gaussians -> particles
    def _load_gaussians(self):
        from src.utils.render_utils import load_gaussian_ckpt, PipelineParamsNoparse
        gs_dir = self.pkg.pkg / str(self.cfg.scene.gs_model)
        if not (gs_dir / "point_cloud").is_dir():
            raise FileNotFoundError(f"{gs_dir}/point_cloud missing: run recon_static.py --package {self.pkg.pkg} first")
        self.gaussians = load_gaussian_ckpt(str(gs_dir), iteration=int(self.cfg.scene.gs_iteration))
        self.pipeline = PipelineParamsNoparse()
        self.pipeline.compute_cov3D_python = True
        log(f"[gs] loaded {self.gaussians.get_xyz.shape[0]} gaussians from {gs_dir}", self.logf)

    def _preprocess(self):
        cfg, pc = self.cfg, self.gaussians
        pp = cfg.preprocessing
        dev = self.device
        with torch.no_grad():
            xyz = pc.get_xyz.detach().float()
            opacity = pc.get_opacity.detach().float()
            shs = pc.get_features.detach().float()          # [N, 16, 3]
            cov_w = pc.get_covariance().detach().float()      # [N, 6] world, upper triangle
        n_all = xyz.shape[0]
        # 1) visual-hull pruning: a Gaussian centre must project inside the (dilated) frame-0 alpha
        #    mask of at least hull_min_views train cameras, so floaters never become MPM particles
        n_views = self.hull_views(xyz, frame=0, dilate_px=int(pp.get("hull_dilate_px", 3)))
        min_views = pp.get("hull_min_views", None)
        min_views = len(self.train_cam_ids) - 1 if min_views is None else int(min_views)
        keep = n_views >= min_views
        log(f"[pre] visual hull (frame 0, {len(self.train_cam_ids)} train cams, >= {min_views} views): removed {int((~keep).sum())} of {n_all} gaussians", self.logf)
        # 2) opacity threshold, and the scene bbox prior (+ margin) / floor as a hard box
        keep &= opacity[:, 0] > float(pp.opacity_threshold)
        bb_lo = torch.tensor(cfg.sim.bbox_min, device=dev) - float(pp.get("crop_margin", 0.1))
        bb_hi = torch.tensor(cfg.sim.bbox_max, device=dev) + float(pp.get("crop_margin", 0.1))
        bb_lo[1] = max(float(bb_lo[1]), float(cfg.sim.floor_height) - 0.01)
        keep &= ((xyz > bb_lo) & (xyz < bb_hi)).all(1)
        idx = keep.nonzero().squeeze(1)
        max_g = pp.get("max_gaussians", None)
        if max_g is not None and idx.numel() > int(max_g):
            idx = idx[torch.randperm(idx.numel(), device=dev, generator=self.gen)[: int(max_g)]]
            idx = idx.sort().values
        xyz, opacity, shs, cov_w = xyz[idx], opacity[idx], shs[idx], cov_w[idx]
        self.n_gs = xyz.shape[0]
        log(f"[pre] gaussians {n_all} -> {self.n_gs} (opacity>{pp.opacity_threshold}, inside bbox prior +{pp.get('crop_margin', 0.1)} m, cap {max_g})", self.logf)
        if self.n_gs < 100:
            raise RuntimeError("too few gaussians survive preprocessing; check the static reconstruction")

        # ---- scene box -> sim cube: x_sim = (x - center) * s + 0.5
        lo = np.array(cfg.sim.bbox_min, dtype=np.float64)
        hi = np.array(cfg.sim.bbox_max, dtype=np.float64)
        if cfg.sim.get("force_bbox_min") is not None and cfg.sim.get("box_include_force", True):
            lo = np.minimum(lo, np.array(cfg.sim.force_bbox_min, dtype=np.float64))
            hi = np.maximum(hi, np.array(cfg.sim.force_bbox_max, dtype=np.float64))
        m = float(cfg.sim.motion_margin)
        lo, hi = lo - m, hi + m
        floor = float(cfg.sim.floor_height)
        lo[1] = min(lo[1], floor - float(cfg.sim.floor_margin))
        # gaussians must be inside the box too (they define the initial state)
        g_lo, g_hi = xyz.min(0).values.cpu().numpy(), xyz.max(0).values.cpu().numpy()
        lo, hi = np.minimum(lo, g_lo - 0.02), np.maximum(hi, g_hi + 0.02)
        center = (lo + hi) / 2
        extent = hi - lo
        s = float(cfg.sim.box_fill) / float(extent.max())
        self.s = s
        self.center = torch.tensor(center, dtype=torch.float32, device=dev)
        self.box_lo, self.box_hi = lo, hi
        self.floor_sim = (floor - center[1]) * s + 0.5
        dx = 1.0 / int(cfg.sim.num_grids)
        log(f"[box] world box {lo.round(3).tolist()} .. {hi.round(3).tolist()} (extent {extent.round(3).tolist()} m), "
            f"s = {s:.4f} sim/m, dx = {dx / s * 100:.2f} cm, floor at y_sim = {self.floor_sim:.4f} ({self.floor_sim / dx:.1f} cells)", self.logf)
        assert self.floor_sim >= 3 * dx, "floor plane too close to the cube boundary; increase sim.floor_margin"

        x_sim = self.world_to_sim(xyz)
        cov_sim = cov_w * (s * s)
        clip = float(cfg.sim.clip_bound) * dx
        assert bool((x_sim > clip).all() and (x_sim < 1 - clip).all()), "gaussians outside the sim cube"
        log(f"[box] gaussians in sim cube: {x_sim.min(0).values.tolist()} .. {x_sim.max(0).values.tolist()}", self.logf)

        # ---- internal filling (taichi, PhysGaussian style) + attribute transfer to filled particles
        if self.filling is not None:
            from src.utils.filling_utils import fill_particles, init_filled_particles, get_particle_volume
            fp = self.filling
            n_grid = int(fp.n_grid)
            pos = fill_particles(pos=x_sim, opacity=opacity, cov=cov_sim, grid_n=n_grid,
                                 max_samples=int(fp.max_particles_num), grid_dx=1.0 / n_grid,
                                 density_thres=float(fp.density_threshold), search_thres=float(fp.search_threshold),
                                 max_particles_per_cell=int(fp.max_particles_per_cell),
                                 search_exclude_dir=int(fp.search_exclude_direction), ray_cast_dir=int(fp.ray_cast_direction),
                                 boundary=None, smooth=bool(fp.smooth)).to(dev)
            n_fill = pos.shape[0] - self.n_gs
            if n_fill > 0:
                shs_all, opacity_all, cov_all = init_filled_particles(pos[: self.n_gs], shs, cov_sim, opacity, pos[self.n_gs:])
            else:
                shs_all, opacity_all, cov_all = shs, opacity, cov_sim
            vol = get_particle_volume(pos, n_grid, 1.0 / n_grid).to(dev)   # cell volume / particles in cell (sim)
            log(f"[fill] +{n_fill} internal particles on a {n_grid}^3 lattice -> {pos.shape[0]} particles; "
                f"volume/particle mean {vol.mean().item():.3e} sim^3 ({vol.mean().item() / s ** 3 * 1e6:.2f} cm^3), total {vol.sum().item() / s ** 3 * 1e3:.2f} L", self.logf)
        else:
            pos, shs_all, opacity_all, cov_all = x_sim, shs, opacity, cov_sim
            vol = torch.full((pos.shape[0],), (1.0 / int(cfg.sim.num_grids)) ** 3, device=dev)
        pos = pos.clamp(clip, 1 - clip)
        self.n_particles = pos.shape[0]

        # ---- network features for ALL particles (upstream load_params only covered the originals)
        from src.utils.transformation_utils import flatten_and_normalize
        n = self.n_particles
        if cfg.model.normalize_features:
            feats = torch.cat((pos, flatten_and_normalize(shs_all, n), flatten_and_normalize(cov_all, n),
                               flatten_and_normalize(opacity_all, n)), dim=1)
        else:
            feats = torch.cat((pos, shs_all.reshape(n, -1), cov_all.reshape(n, -1), opacity_all.reshape(n, -1)), dim=1)

        self.x0 = pos.detach().contiguous()
        self.features = feats.detach().contiguous()
        self.vol = vol.detach().contiguous()
        self.shs = shs.contiguous()            # render only the original gaussians (render_mask = first n_gs)
        self.opacity = opacity.contiguous()
        self.cov_w0 = cov_w.contiguous()       # world covariance at rest; render cov = F cov0 F^T
        self.ones_color = torch.ones(self.n_gs, 3, device=dev)
        self.bg_black = torch.zeros(3, device=dev)
        from src.utils.physon_scene import write_ply_xyz
        write_ply_xyz(self.out / "particles_init.ply", self.sim_to_world(self.x0).cpu().numpy())

    # ------------------------------------------------------------------ camera geometry helpers
    def cam_KRT(self, cam_id):
        c = self.pkg.cameras[cam_id]
        K = torch.tensor(c["K"], dtype=torch.float32, device=self.device)
        R = torch.tensor(c["R"], dtype=torch.float32, device=self.device)   # 3DGS convention: c2w rotation
        T = torch.tensor(c["T"], dtype=torch.float32, device=self.device)   # w2c translation
        return K, R, T

    def project(self, x_world, cam_id):
        """world points [N,3] -> pixel (u, v) [N], depth [N] in camera cam_id."""
        K, R, T = self.cam_KRT(cam_id)
        xc = x_world @ R + T                       # x_c = R^T x + T
        z = xc[:, 2].clamp_min(1e-6)
        u = K[0, 0] * xc[:, 0] / z + K[0, 2]
        v = K[1, 1] * xc[:, 1] / z + K[1, 2]
        return u, v, xc[:, 2]

    def gt_mask(self, cam_id, frame, thr=0.5, dilate_px=0):
        a = self.gt[self.gt_cam_index[cam_id], frame, :, :, 3].to(self.device).float() / 255.0
        m = (a > thr).float()
        if dilate_px > 0:
            k = 2 * dilate_px + 1
            m = torch.nn.functional.max_pool2d(m[None, None], k, stride=1, padding=dilate_px)[0, 0]
        return m > 0.5

    @torch.no_grad()
    def hull_views(self, x_world, frame=0, dilate_px=3):
        """number of train cameras whose (dilated) alpha mask contains each point's projection."""
        n = torch.zeros(x_world.shape[0], dtype=torch.long, device=self.device)
        for c in self.train_cam_ids:
            m = self.gt_mask(c, frame, dilate_px=dilate_px)
            u, v, z = self.project(x_world, c)
            ui, vi = u.round().long(), v.round().long()
            ok = (z > 0) & (ui >= 0) & (ui < self.W) & (vi >= 0) & (vi < self.H)
            hit = torch.zeros_like(ok)
            hit[ok] = m[vi[ok], ui[ok]]
            n += hit.long()
        return n

    @torch.no_grad()
    def triangulate_centroids(self, frames):
        """3D object centroid per frame: least-squares intersection of the camera rays through the
        2D silhouette centroids (alpha > 0.5) of the train cameras. Returns [len(frames), 3] world."""
        out = []
        for f in frames:
            A = torch.zeros(3, 3, device=self.device); b = torch.zeros(3, device=self.device)
            for c in self.train_cam_ids:
                m = self.gt_mask(c, f)
                if m.sum() < 10:
                    continue
                vv, uu = torch.nonzero(m, as_tuple=True)
                u0, v0 = uu.float().mean(), vv.float().mean()
                K, R, T = self.cam_KRT(c)
                d_cam = torch.stack([(u0 - K[0, 2]) / K[0, 0], (v0 - K[1, 2]) / K[1, 1], torch.ones((), device=self.device)])
                d = R @ d_cam; d = d / d.norm()
                C = -(R @ T)                        # camera centre in world coordinates
                P = torch.eye(3, device=self.device) - torch.outer(d, d)
                A += P; b += P @ C
            out.append(torch.linalg.solve(A, b))
        return torch.stack(out)

    def _init_velocity(self):
        """Initial rigid velocity. `train.v0_mode=closed_form` (default): the object centroid c(t) is
        triangulated from the train silhouettes for the first train.v0_frames frames (before floor
        contact) and c(t) = c0 + v0 t + a t^2/2 is fitted by linear least squares; v0 (world -> sim by
        *s) initialises the state, omega = 0. `learn`: init from init.v0/init.omega. Either way an
        optional short refinement follows (train.v0_refine) and v0/omega are frozen during the
        material phase unless train.v0_frozen_in_material_phase=false."""
        tr = self.cfg.train
        self.v0_info = dict(mode=str(tr.get("v0_mode", "closed_form")))
        if self.v0_info["mode"] == "closed_form":
            k = min(int(tr.get("v0_frames", 8)), self.n_frames_total)
            frames = list(range(k))
            c = self.triangulate_centroids(frames)                       # [k, 3]
            t = torch.tensor(frames, dtype=torch.float32, device=self.device) * self.frame_dt
            X = torch.stack([torch.ones_like(t), t, 0.5 * t * t], dim=1)  # c(t) = c0 + v0 t + a t^2/2
            sol = torch.linalg.lstsq(X, c).solution                      # [3, 3]
            c0, v0, a = sol[0], sol[1], sol[2]
            resid = float((X @ sol - c).norm(dim=1).mean())
            a_ref = torch.tensor(self.cfg.sim.gravity, device=self.device)
            if self.force_grid is not None:
                from src.utils.physon_force import ForceField  # noqa: F401 (documentation only)
                node = self.world_to_sim(c0[None])
                idx = ((node * self.mpm.num_grids).round().long().clamp(0, self.mpm.num_grids - 1))
                flat = int(idx[0, 0] * self.mpm.num_grids ** 2 + idx[0, 1] * self.mpm.num_grids + idx[0, 2])
                a_ref = a_ref + self.force_grid[0, flat] / self.s
            self.v0_info.update(frames=k, c0=c0.tolist(), v0=v0.tolist(), a=a.tolist(), fit_residual_m=resid,
                                a_expected=a_ref.tolist(), centroids=c.tolist())
            log(f"[v0] closed form over {k} frames: c0 {np.round(c0.tolist(), 4).tolist()} v0 {np.round(v0.tolist(), 4).tolist()} m/s "
                f"a {np.round(a.tolist(), 3).tolist()} m/s^2 (expected g+ext {np.round(a_ref.tolist(), 3).tolist()}), residual {resid * 1000:.2f} mm", self.logf)
            with torch.no_grad():
                self.v0.copy_(v0 * self.s); self.omega.zero_()
        elif self.v0_info["mode"] != "learn":
            raise ValueError("train.v0_mode must be closed_form|learn")
        self.v0_info["v0_init_world"] = (self.v0 / self.s).tolist()

    # ------------------------------------------------------------------ world <-> sim
    def world_to_sim(self, x):
        return (x - self.center) * self.s + 0.5

    def sim_to_world(self, x):
        return (x - 0.5) / self.s + self.center

    # ------------------------------------------------------------------ simulator
    def _build_sim(self):
        from src.mpm_core import MPMModel, set_boundary_conditions
        from src.utils.physon_force import ForceField
        cfg, dev = self.cfg, self.device
        steps = int(round(self.frame_dt / float(cfg.sim.dt)))
        dt = self.frame_dt / steps
        if abs(dt - float(cfg.sim.dt)) / float(cfg.sim.dt) > 0.01:
            log(f"[warn] dt adjusted from {float(cfg.sim.dt):.3e} to {dt:.3e} to fit frame_dt exactly", self.logf)
        self.steps_per_frame, self.dt = steps, dt
        g_world = np.array(cfg.sim.gravity, dtype=np.float64)
        g_sim = (g_world * self.s).tolist()
        sim_params = dict(num_grids=int(cfg.sim.num_grids), dt=dt, gravity=g_sim, clip_bound=float(cfg.sim.clip_bound),
                          damping=float(cfg.sim.damping), particle_vol=self.vol)
        self.mpm = MPMModel(sim_params, dict(rho=float(cfg.material.rho)), init_pos=self.x0, enable_train=True, device=dev)
        set_boundary_conditions(self.mpm, [dict(type="surface_collider", point=[0.5, float(self.floor_sim), 0.5], normal=[0.0, 1.0, 0.0],
                                                surface=str(cfg.sim.floor_surface), friction=0.0, start_time=0.0, end_time=1e9)])
        log(f"[sim] grid {cfg.sim.num_grids}^3, dt {dt:.4e} s x {steps} substeps = frame_dt {self.frame_dt:.5f} s ({1 / self.frame_dt:.1f} fps), "
            f"gravity sim {np.round(g_sim, 4).tolist()} (world {g_world.tolist()}), floor '{cfg.sim.floor_surface}' at y_sim {self.floor_sim:.4f}, "
            f"{self.n_particles} particles, mass total {float(self.mpm.p_mass.sum()) if torch.is_tensor(self.mpm.p_mass) else self.mpm.p_mass * self.n_particles:.4g} (sim)", self.logf)

        # ---- external force on the grid nodes, per frame, sim units (explicit tensor passed through checkpoint)
        self.force_grid = None
        mode = str(cfg.sim.force_mode)
        if mode == "oracle":
            npz = self.pkg.pkg / str(cfg.sim.force_npz) if cfg.sim.force_npz else None
            if npz is None or not npz.exists():
                log(f"[force] force_mode=oracle but no force file ({npz}); external acceleration = 0", self.logf)
            else:
                ff = ForceField(self.pkg.pkg / "source", cfg.sim.gravity, npz_path=npz,
                                json_path=(self.pkg.pkg / str(cfg.sim.force_json)) if cfg.sim.force_json else None,
                                h5_path=(self.pkg.pkg / str(cfg.sim.physics_h5)) if cfg.sim.physics_h5 else None, device=str(dev))
                node_world = self.sim_to_world(self.mpm.grid_x * self.mpm.dx)   # [G^3, 3] static
                T = self.n_frames_total
                fg = torch.empty((T, node_world.shape[0], 3), device=dev)
                stats = []
                x0w = self.sim_to_world(self.x0)
                with torch.no_grad():
                    for f in range(T):
                        fg[f] = ff.sample(node_world, f) * self.s
                        a = ff.sample(x0w, f).norm(dim=1)
                        stats.append((float(a.mean()), float(a.max())))
                self.force_grid = fg
                log(f"[force] {ff.describe()} -> grid tensor {tuple(fg.shape)} sim m/s^2 ({fg.numel() * 4 / 2 ** 20:.0f} MB)", self.logf)
                for f in (0, T // 4, T // 2, 3 * T // 4, T - 1):
                    log(f"[force] frame {f:2d}: |a_ext| at frame-0 particles mean {stats[f][0]:.3f} max {stats[f][1]:.3f} m/s^2 "
                        f"(sim {stats[f][0] * self.s:.3f} / {stats[f][1] * self.s:.3f}); |g| sim {abs(g_sim[1]):.3f}", self.logf)
        elif mode != "none":
            raise ValueError(f"sim.force_mode must be oracle|none, got {mode}")
        else:
            log("[force] force_mode=none: no external acceleration (as-is ablation)", self.logf)

    # ------------------------------------------------------------------ material + initial velocity
    def _build_material(self):
        from src.physics_guided_network import PhysicsNetwork
        from src.constitutive_models import get_elasticity, get_plasticity
        cfg, dev = self.cfg, self.device
        mat = cfg.material
        e_list = list(mat.elasticity_physicals)
        p_list = list(mat.plasticity_physicals)
        assert mat.elasticity == "neural" and mat.plasticity == "neural", "fit.py uses the neural (Gumbel) mixture over the expert lists"
        model_params = OmegaConf.to_container(cfg.model, resolve=True)
        model_params["num_groups"] = min(int(model_params["num_groups"]), self.n_particles)
        model_params = OmegaConf.create(model_params)
        self.material = PhysicsNetwork(elasticity_physicals=e_list, plasticity_physicals=p_list,
                                       in_channels=self.features.shape[1], params=model_params,
                                       n_particles=self.n_particles, export_path=str(self.out)).to(dev)
        lcfg = OmegaConf.to_container(mat, resolve=True)
        learnable = bool(mat.get("learnable_experts", True))
        self.elasticity = get_elasticity("neural", physicals=e_list, device=dev, learnable=learnable, learnable_cfg=lcfg).to(dev)
        self.plasticity = get_plasticity("neural", physicals=p_list, device=dev, learnable=learnable, learnable_cfg=lcfg).to(dev)
        self.experts = [m for m in list(self.elasticity.physicals) + list(self.plasticity.physicals)]
        if learnable:
            for m in self.experts:
                m.set_length_scale(self.s)
            # optional per-expert overrides {index: {E: .., nu: ..}} (e.g. GT reference rollouts)
            for key, lst in (("elasticity_overrides", self.elasticity.physicals), ("plasticity_overrides", self.plasticity.physicals)):
                for i, kv in (mat.get(key, None) or {}).items():
                    lst[int(i)].set_world_params(**OmegaConf.to_container(kv, resolve=True))
        else:
            log("[warn] material.learnable_experts=false: upstream fixed constants (E=2e6 in SIM units!)", self.logf)
        self.e_names, self.p_names = e_list, p_list
        self.fixed_e = mat.get("fixed_elasticity", None)
        self.fixed_p = mat.get("fixed_plasticity", None)

        # learnable rigid initial state (sim units; world v0 * s)
        v0 = torch.tensor(list(cfg.init.v0), dtype=torch.float32, device=dev) * self.s
        om = torch.tensor(list(cfg.init.omega), dtype=torch.float32, device=dev)
        self.v0 = nn.Parameter(v0)
        self.omega = nn.Parameter(om)
        self.centroid = self.x0.mean(0)

        n_net = sum(p.numel() for p in self.material.parameters())
        n_exp = len(self.expert_param_list())
        log(f"[material] elasticity {e_list}, plasticity {p_list}, network '{cfg.model.network}' ({n_net} params), "
            f"{n_exp} expert parameter tensors, learnable={learnable}; init {self.expert_params()}", self.logf)

    def expert_param_list(self):
        out = []
        for m in self.experts:
            out += [p for n, p in m.named_parameters() if n != "useless"]
        return out

    def expert_params(self):
        """Current expert parameters in WORLD units."""
        res = []
        for name, m in zip(self.e_names + self.p_names, self.experts):
            res.append(dict(name=name, **(m.world_params() if hasattr(m, "world_params") else {})))
        return res

    def categories(self):
        """Per-particle logits (e_cat, p_cat) from the network, or fixed one-hot experts."""
        e_cat, p_cat = self.material(self.x0, self.features)
        if self.fixed_e is not None:
            e_cat = torch.full_like(e_cat, -10.0); e_cat[:, int(self.fixed_e)] = 10.0
        if self.fixed_p is not None:
            p_cat = torch.full_like(p_cat, -10.0); p_cat[:, int(self.fixed_p)] = 10.0
        return e_cat, p_cat

    def initial_state(self):
        x = self.x0.clone()
        v = self.v0[None, :] + torch.cross(self.omega[None, :].expand_as(x), x - self.centroid, dim=1)
        C = torch.zeros((self.n_particles, 3, 3), device=self.device)
        F = torch.eye(3, device=self.device)[None].repeat(self.n_particles, 1, 1)
        return x, v, C, F

    # ------------------------------------------------------------------ GT
    def _load_gt(self):
        t0 = time.time()
        cams = self.train_cam_ids + [c for c in self.test_cam_ids if c not in self.train_cam_ids]
        self.gt_cam_index = {c: i for i, c in enumerate(cams)}
        self.gt = self.pkg.load_gt_stack(cams, list(range(self.n_frames_total)))   # uint8 [cams, T, H, W, 4] pinned
        self.H, self.W = self.gt.shape[2], self.gt.shape[3]
        self.cams = {c: self.pkg.gs_camera(c) for c in cams}
        log(f"[gt] {len(cams)} cams x {self.n_frames_total} frames ({self.H}x{self.W}) loaded in {time.time() - t0:.1f}s "
            f"({self.gt.numel() / 2 ** 20:.0f} MB, pinned); train cams {self.train_cam_ids}, test cams {self.test_cam_ids}", self.logf)
        self.gt_pcd = {}
        if self.pkg.has_gt_particles():
            from src.utils.physon_scene import read_ply_xyz
            for f in range(self.n_frames_total):
                p = self.pkg.gt_ply_path(f)
                if p.exists():
                    self.gt_pcd[f] = read_ply_xyz(p)
            log(f"[gt] {len(self.gt_pcd)} GT particle frames (evaluation only unless loss.w_chamfer > 0)", self.logf)

    def gt_view(self, cam, f):
        from src.utils.physon_scene import composite_white
        rgba = self.gt[self.gt_cam_index[cam], f].to(self.device, non_blocking=True)
        rgb, a = composite_white(rgba)
        return rgb, a

    # ------------------------------------------------------------------ rendering
    def frame_render_params(self, x, F):
        """Render inputs of the original Gaussians: world positions, deformed world covariance
        F cov0 F^T and R(F)^T for the view-dependent SH (upstream compute_R_from_F uses the warp SVD,
        whose backward is singular at F = I; the Newton polar iteration is used instead)."""
        from src.utils.transformation_utils import compute_cov_from_F
        n = self.n_gs
        pos_w = self.sim_to_world(x[:n])
        cov = compute_cov_from_F(self.cov_w0, F[:n])
        rot = polar_rotation(F[:n]).transpose(1, 2)
        return pos_w, cov, rot

    def render(self, cam_id, pos_w, cov, rot):
        """-> (rgb on white [3,H,W], alpha [1,H,W]); two rasterizations of the inria rasterizer
        (colour with black background, then ones-colour for the accumulated alpha)."""
        from src.utils.render_utils import initialize_resterize, convert_SH
        cam = self.cams[cam_id]
        rasterizer = initialize_resterize(cam, self.gaussians, self.pipeline, self.bg_black)
        colors = convert_SH(self.shs, cam, self.gaussians, pos_w, rot)
        screen = torch.zeros_like(pos_w, requires_grad=True)
        pre, _ = rasterizer(means3D=pos_w, means2D=screen, shs=None, colors_precomp=colors, opacities=self.opacity,
                            scales=None, rotations=None, cov3D_precomp=cov)
        alpha, _ = rasterizer(means3D=pos_w, means2D=screen, shs=None, colors_precomp=self.ones_color, opacities=self.opacity,
                              scales=None, rotations=None, cov3D_precomp=cov)
        alpha = alpha[:1].clamp(0, 1)
        rgb = (pre + (1.0 - alpha)).clamp(0, 1)
        return rgb, alpha

    def view_loss(self, rgb, alpha, gt_rgb, gt_a, use_rgb=True):
        """L1(rgb) + w_alpha L1(alpha) over the union bbox of predicted / GT alpha (+ margin).
        use_rgb=False (black-silhouette views, train.rgb_exclude_cams): alpha term only."""
        cfg = self.cfg.loss
        m = (alpha[0] > 0.01) | (gt_a[0] > 0.01)
        rows, cols = m.any(1).nonzero(), m.any(0).nonzero()
        if rows.numel() == 0:
            y0, y1, x0, x1 = 0, self.H, 0, self.W
        else:
            mg = int(cfg.crop_margin)
            y0, y1 = max(int(rows.min()) - mg, 0), min(int(rows.max()) + mg + 1, self.H)
            x0, x1 = max(int(cols.min()) - mg, 0), min(int(cols.max()) + mg + 1, self.W)
        l_rgb = (rgb[:, y0:y1, x0:x1] - gt_rgb[:, y0:y1, x0:x1]).abs().mean()
        l_a = (alpha[:, y0:y1, x0:x1] - gt_a[:, y0:y1, x0:x1]).abs().mean()
        w_rgb = float(cfg.w_rgb) if use_rgb else 0.0
        return w_rgb * l_rgb + float(cfg.w_alpha) * l_a, l_rgb.detach(), l_a.detach()

    # ------------------------------------------------------------------ simulation
    def simulate_frame(self, x, v, C, F, e_cat, p_cat, frame, use_ckpt):
        """Advance the state from frame `frame` to frame+1 (steps_per_frame substeps). The external
        acceleration of this frame is passed as an explicit tensor so checkpoint recomputation is exact."""
        a_ext = self.force_grid[min(frame, self.force_grid.shape[0] - 1)] if self.force_grid is not None else None
        for _ in range(self.steps_per_frame):
            if use_ckpt:
                stress = checkpoint(self.elasticity, F, e_cat, use_reentrant=True)
                x, v, C, F = checkpoint(self.mpm, x, v, C, F, stress, a_ext, use_reentrant=True)
                F = checkpoint(self.plasticity, F, p_cat, use_reentrant=True)
            else:
                stress = self.elasticity(F, e_cat)
                x, v, C, F = self.mpm(x, v, C, F, stress, a_ext)
                F = self.plasticity(F, p_cat)
        return x, v, C, F

    @staticmethod
    def finite(*ts):
        return all(bool(torch.isfinite(t).all()) for t in ts)

    def rollout_loss(self, state, frame0, n_frames, e_cat, p_cat, cams_per_frame):
        """Simulate n_frames from `state` (state at frame0), supervise frames frame0+1..frame0+n_frames.
        Returns (loss, parts, final_state) or (None, info, None) when the state became non-finite."""
        x, v, C, F = state
        tot, parts = 0.0, dict(rgb=0.0, alpha=0.0, chamfer=0.0, n=0)
        w_ch = float(self.cfg.loss.w_chamfer)
        for i in range(n_frames):
            f = frame0 + i
            x, v, C, F = self.simulate_frame(x, v, C, F, e_cat, p_cat, f, use_ckpt=True)
            if not self.finite(x, F):
                return None, dict(frame=f + 1), None
            fr = f + 1
            if fr >= self.n_frames_total:
                break
            pos_w, cov, rot = self.frame_render_params(x, F)
            cams = self.train_cam_ids if cams_per_frame >= len(self.train_cam_ids) else random.sample(self.train_cam_ids, cams_per_frame)
            for c in cams:
                rgb, alpha = self.render(c, pos_w, cov, rot)
                gt_rgb, gt_a = self.gt_view(c, fr)
                l, l_rgb, l_a = self.view_loss(rgb, alpha, gt_rgb, gt_a, use_rgb=c not in self.rgb_exclude)
                tot = tot + l / len(cams)
                parts["rgb"] += float(l_rgb) / len(cams); parts["alpha"] += float(l_a) / len(cams)
            if w_ch > 0 and fr in self.gt_pcd:
                gt = torch.as_tensor(self.gt_pcd[fr], device=self.device)
                lc = chamfer_torch(self.sim_to_world(x), gt, int(self.cfg.loss.chamfer_samples), self.gen)
                tot = tot + w_ch * lc
                parts["chamfer"] += float(lc)
            parts["n"] += 1
        n = max(parts["n"], 1)
        for k in ("rgb", "alpha", "chamfer"):
            parts[k] /= n
        return tot / n, parts, (x, v, C, F)

    # ------------------------------------------------------------------ checkpoints
    def ckpt_path(self, epoch):
        return self.out / "checkpoints" / f"epoch_{epoch:04d}.pth"

    def save_ckpt(self, path, epoch, opt=None, phase="train"):
        torch.save(dict(material=self.material.state_dict(), elasticity=self.elasticity.state_dict(),
                        plasticity=self.plasticity.state_dict(), v0=self.v0.detach().cpu(), omega=self.omega.detach().cpu(),
                        optimizer=opt.state_dict() if opt is not None else None, epoch=epoch, phase=phase), path)

    def load_ckpt(self, path, opt=None):
        ck = torch.load(path, map_location=self.device)
        self.material.load_state_dict(ck["material"])
        self.elasticity.load_state_dict(ck["elasticity"])
        self.plasticity.load_state_dict(ck["plasticity"])
        with torch.no_grad():
            self.v0.copy_(ck["v0"].to(self.device)); self.omega.copy_(ck["omega"].to(self.device))
        if opt is not None and ck.get("optimizer") is not None:
            try:
                opt.load_state_dict(ck["optimizer"])
            except Exception as e:  # e.g. a changed expert list
                log(f"[warn] optimizer state not restored: {e}", self.logf)
        log(f"[ckpt] loaded {path} (phase {ck.get('phase')}, epoch {ck.get('epoch')})", self.logf)
        return ck

    def latest_epoch_ckpt(self):
        cks = sorted((self.out / "checkpoints").glob("epoch_*.pth"))
        return cks[-1] if cks else None

    # ------------------------------------------------------------------ training
    def grad_norms(self):
        def gn(params):
            g = [p.grad.norm() ** 2 for p in params if p.grad is not None]
            return float(torch.stack(g).sum().sqrt()) if g else 0.0
        return dict(net=gn(self.material.parameters()), expert=gn(self.expert_param_list()), vel=gn([self.v0, self.omega]))

    def clean_grads(self, params):
        """nan_to_num + clip as upstream, but non-finite gradients are counted and the first
        occurrence is logged with the parameter groups affected (upstream hides them silently)."""
        bad = {}
        named = [("net", self.material.parameters()), ("expert", self.expert_param_list()), ("vel", [self.v0, self.omega])]
        for name, ps in named:
            n = sum(int(not torch.isfinite(p.grad).all()) for p in ps if p.grad is not None)
            if n:
                bad[name] = n
        if bad:
            self.n_nonfinite_grad_steps = getattr(self, "n_nonfinite_grad_steps", 0) + 1
            if self.n_nonfinite_grad_steps <= 3:
                log(f"[warn] non-finite gradients zeroed (tensors per group: {bad}); occurrence {self.n_nonfinite_grad_steps}", self.logf)
        for p in params:
            if p.grad is not None:
                torch.nan_to_num_(p.grad, 0.0, 0.0, 0.0)
        clip = float(self.cfg.train.grad_clip)
        if clip > 0:
            torch.nn.utils.clip_grad_norm_([p for p in params if p.grad is not None], clip)

    def train(self):
        cfg = self.cfg
        tr = cfg.train
        views = int(tr.views_per_frame) if int(tr.views_per_frame) > 0 else len(self.train_cam_ids)
        self.step = 0
        expert_params = self.expert_param_list()
        # v0/omega are frozen in the material phase by default (they absorb early errors otherwise)
        vel_params = [self.v0, self.omega] if not bool(tr.get("v0_frozen_in_material_phase", True)) else []
        opt = torch.optim.Adam([dict(params=list(self.material.parameters()), lr=float(tr.learning_rate)),
                                dict(params=expert_params, lr=float(tr.expert_lr))] +
                               ([dict(params=vel_params, lr=float(tr.vel_lr))] if vel_params else []))
        start_epoch = 0
        last = self.latest_epoch_ckpt()
        if last is not None:
            ck = self.load_ckpt(last, opt)
            start_epoch = int(ck["epoch"]) + 1
        elif (self.out / "checkpoints" / "velocity.pth").exists():
            self.load_ckpt(self.out / "checkpoints" / "velocity.pth")
        elif bool(tr.get("v0_refine", True)) and int(tr.vel_iters) > 0 and int(tr.vel_frames) > 0:
            self.train_velocity()
            self.save_ckpt(self.out / "checkpoints" / "velocity.pth", -1, None, phase="velocity")
        self.v0_info["v0_refined_world"] = (self.v0 / self.s).tolist()
        self.v0_info["omega_refined"] = self.omega.tolist()
        if not vel_params:
            self.v0.requires_grad_(False); self.omega.requires_grad_(False)
        epochs = int(tr.epochs)
        if start_epoch >= epochs:
            log(f"[train] all {epochs} epochs already done ({last}); skipping training", self.logf)
            return
        t0 = time.time()
        fps = int(cfg.sim.frames_per_stage)
        n_sim = self.n_frames_train - 1                    # frames 1..n-1 are simulated/supervised
        stages = [(f0, min(fps, n_sim - f0)) for f0 in range(0, n_sim, fps)]
        log(f"[train] epochs {start_epoch}..{epochs - 1}, {len(stages)} stages of {fps} frames over {self.n_frames_train} frames, "
            f"{tr.internal_epochs} steps/stage, {views} views/frame", self.logf)
        for p in self.material.parameters():
            p.requires_grad_(True)
        for p in expert_params:
            p.requires_grad_(True)
        for epoch in range(start_epoch, epochs):
            state = self.initial_state()
            state = tuple(t.detach() for t in state)
            for si, (f0, nf) in enumerate(stages):
                stage_state = None
                for it in range(int(tr.internal_epochs)):
                    ts = time.time()
                    opt.zero_grad(set_to_none=True)
                    if si == 0:
                        x, v, C, F = self.initial_state()            # v depends on v0/omega
                    else:
                        x, v, C, F = [t.detach().requires_grad_(True) for t in state]
                    self.mpm.reset()
                    e_cat, p_cat = self.categories()
                    loss, parts, final = self.rollout_loss((x, v, C, F), f0, nf, e_cat, p_cat, views)
                    if loss is None or not torch.isfinite(loss):
                        log(f"[train] epoch {epoch} stage {si} it {it}: non-finite state/loss ({parts}); step skipped", self.logf)
                        self.step += 1
                        continue
                    loss.backward()
                    gns = self.grad_norms()
                    self.clean_grads(list(self.material.parameters()) + expert_params + vel_params)
                    opt.step()
                    for m in self.experts:
                        if hasattr(m, "project_"):
                            m.project_()
                    stage_state = tuple(t.detach() for t in final)
                    dt_step = time.time() - ts
                    usage = self.expert_usage(e_cat, p_cat)
                    line = (f"epoch {epoch} stage {si} (frames {f0 + 1}-{f0 + nf}) it {it}: loss {float(loss):.5f} "
                            f"rgb {parts['rgb']:.5f} alpha {parts['alpha']:.5f} chamfer {parts['chamfer']:.4g} | grad net {gns['net']:.3e} "
                            f"expert {gns['expert']:.3e} vel {gns['vel']:.3e} | v0 {[round(float(a) / self.s, 4) for a in self.v0]} "
                            f"| {dt_step:.1f}s | E {[round(p.get('E', 0)) for p in self.expert_params()[:len(self.e_names)]]} "
                            f"nu {[round(p.get('nu', 0), 3) for p in self.expert_params()[:len(self.e_names)]]} | use {usage}")
                    log(line, self.logf)
                    self.writer.add_scalar("loss/total", float(loss), self.step)
                    self.writer.add_scalar("loss/rgb", parts["rgb"], self.step)
                    self.writer.add_scalar("loss/alpha", parts["alpha"], self.step)
                    self.writer.add_scalar("time/step_s", dt_step, self.step)
                    for k, vv in gns.items():
                        self.writer.add_scalar(f"grad/{k}", vv, self.step)
                    self.step += 1
                if stage_state is None or not self.finite(*stage_state):
                    with torch.no_grad():
                        e_cat, p_cat = self.categories()
                        x, v, C, F = self.initial_state() if si == 0 else state
                        for i in range(nf):
                            x, v, C, F = self.simulate_frame(x, v, C, F, e_cat, p_cat, f0 + i, use_ckpt=False)
                        stage_state = (x, v, C, F)
                    if not self.finite(*stage_state):
                        log(f"[train] epoch {epoch}: state non-finite after stage {si} even without grad; epoch aborted", self.logf)
                        break
                state = stage_state
            if (epoch + 1) % int(tr.ckpt_interval) == 0 or epoch == epochs - 1:
                self.save_ckpt(self.ckpt_path(epoch), epoch, opt)
                log(f"[ckpt] saved {self.ckpt_path(epoch)}", self.logf)
        self.timings["train_s"] = time.time() - t0
        log(f"[train] done in {self.timings['train_s']:.0f}s", self.logf)

    def train_velocity(self):
        """Phase 1: fit v0/omega on the first vel_frames frames with the material frozen."""
        cfg = self.cfg
        tr = cfg.train
        t0 = time.time()
        views = int(tr.views_per_frame) if int(tr.views_per_frame) > 0 else len(self.train_cam_ids)
        nf = min(int(tr.vel_frames), self.n_frames_train - 1)
        for p in list(self.material.parameters()) + self.expert_param_list():
            p.requires_grad_(False)
        self.v0.requires_grad_(True); self.omega.requires_grad_(True)
        opt = torch.optim.Adam([self.v0, self.omega], lr=float(tr.vel_lr))
        with torch.no_grad():
            e_cat, p_cat = self.categories()
        log(f"[vel] {tr.vel_iters} steps on v0/omega over frames 1-{nf}", self.logf)
        for it in range(int(tr.vel_iters)):
            ts = time.time()
            opt.zero_grad(set_to_none=True)
            self.mpm.reset()
            state = self.initial_state()
            loss, parts, _ = self.rollout_loss(state, 0, nf, e_cat, p_cat, views)
            if loss is None or not torch.isfinite(loss):
                log(f"[vel] it {it}: non-finite ({parts}); skipped", self.logf)
                continue
            loss.backward()
            gns = self.grad_norms()
            self.clean_grads([self.v0, self.omega])
            opt.step()
            log(f"[vel] it {it}: loss {float(loss):.5f} rgb {parts['rgb']:.5f} alpha {parts['alpha']:.5f} | grad vel {gns['vel']:.3e} | "
                f"v0 world {[round(float(a) / self.s, 4) for a in self.v0]} omega {[round(float(a), 4) for a in self.omega]} | {time.time() - ts:.1f}s", self.logf)
            self.writer.add_scalar("vel/loss", float(loss), it)
        for p in list(self.material.parameters()) + self.expert_param_list():
            p.requires_grad_(True)
        self.timings["velocity_s"] = time.time() - t0

    def expert_usage(self, e_cat, p_cat):
        with torch.no_grad():
            e = torch.bincount(e_cat.argmax(1), minlength=len(self.e_names)).float() / e_cat.shape[0]
            p = torch.bincount(p_cat.argmax(1), minlength=len(self.p_names)).float() / p_cat.shape[0]
        return dict(e=[round(float(a), 3) for a in e], p=[round(float(a), 3) for a in p])

    # ------------------------------------------------------------------ final evaluation
    @torch.no_grad()
    def evaluate(self):
        import imageio.v2 as imageio
        from src.utils.physon_scene import write_ply_xyz
        from utils.loss_utils import ssim as ssim_fn
        cfg = self.cfg
        t0 = time.time()
        e_cat, p_cat = self.categories()
        usage = self.expert_usage(e_cat, p_cat)
        # material visualisation
        x0w = self.sim_to_world(self.x0).cpu().numpy()
        write_ply_xyz(self.out / "material.ply", x0w, PALETTE[e_cat.argmax(1).cpu().numpy() % len(PALETTE)])
        write_ply_xyz(self.out / "material_plasticity.ply", x0w, PALETTE[p_cat.argmax(1).cpu().numpy() % len(PALETTE)])
        params = dict(experts_elasticity=[dict(p, usage=usage["e"][i]) for i, p in enumerate(self.expert_params()[: len(self.e_names)])],
                      experts_plasticity=[dict(p, usage=usage["p"][i]) for i, p in enumerate(self.expert_params()[len(self.e_names):])],
                      v0_world=(self.v0 / self.s).tolist(), omega_world=self.omega.tolist(), v0_estimation=self.v0_info,
                      world_to_sim=dict(s=self.s, box_center=self.center.tolist(), box_lo=self.box_lo.tolist(), box_hi=self.box_hi.tolist(),
                                        floor_sim=float(self.floor_sim)),
                      n_particles=self.n_particles, n_gaussians=self.n_gs, dt=self.dt, steps_per_frame=self.steps_per_frame,
                      force_mode=str(cfg.sim.force_mode), gt_regions=self.pkg.meta.get("regions", []))
        json.dump(params, open(self.out / "params.json", "w"), indent=1)
        log(f"[eval] params: {json.dumps(params['experts_elasticity'])} v0 {params['v0_world']}", self.logf)

        test_cam = self.test_cam_ids[0]
        rows, frames_video = [], []
        x, v, C, F = self.initial_state()
        self.mpm.reset()
        last_ok = (x, v, C, F)
        nan_from = None
        for f in range(self.n_frames_total):
            if f > 0:
                if nan_from is None:
                    x, v, C, F = self.simulate_frame(x, v, C, F, e_cat, p_cat, f - 1, use_ckpt=False)
                    if not self.finite(x, F):
                        nan_from = f
                        log(f"[eval] non-finite state at frame {f}; holding the last finite state", self.logf)
                        x, v, C, F = last_ok
                    else:
                        last_ok = (x, v, C, F)
            xw = self.sim_to_world(x)
            write_ply_xyz(self.out / "particles" / f"{f:03d}.ply", xw.cpu().numpy())
            pos_w, cov, rot = self.frame_render_params(x, F)
            rgb, alpha = self.render(test_cam, pos_w, cov, rot)
            gt_rgb, gt_a = self.gt_view(test_cam, f)
            row = dict(frame=f, psnr=psnr_t(rgb, gt_rgb), ssim=float(ssim_fn(rgb[None], gt_rgb[None])),
                       alpha_iou=float(((alpha > 0.5) & (gt_a > 0.5)).sum() / max(int(((alpha > 0.5) | (gt_a > 0.5)).sum()), 1)))
            if f in self.gt_pcd:
                row["cd"] = chamfer_sq_mm2(xw.cpu().numpy().astype(np.float64), self.gt_pcd[f].astype(np.float64))
            rows.append(row)
            rgb8 = (rgb.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            gt8 = (gt_rgb.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            a8 = (alpha[0].cpu().numpy() * 255).round().astype(np.uint8)
            imageio.imwrite(self.out / "renders_test" / f"{f:03d}.png", rgb8)
            imageio.imwrite(self.out / "renders_test_alpha" / f"{f:03d}.png", a8)
            imageio.imwrite(self.out / "gt_test" / f"{f:03d}.png", gt8)
            frames_video.append(np.concatenate([rgb8, gt8], axis=1))
            if f % 8 == 0 or f == self.n_frames_total - 1:
                log(f"[eval] frame {f:2d}: psnr {row['psnr']:.2f} ssim {row['ssim']:.3f} iou {row['alpha_iou']:.3f}" +
                    (f" cd {row['cd']:.3f}" if "cd" in row else "") + f" | |v| max {float(v.norm(dim=1).max()) / self.s:.3f} m/s", self.logf)
        if not self.args.no_video:
            try:
                imageio.mimsave(self.out / "video_test.mp4", frames_video, fps=int(cfg.eval.video_fps), macro_block_size=16)
            except Exception as e:
                log(f"[warn] video not written: {e}", self.logf)
        self.timings["eval_s"] = time.time() - t0
        mean = lambda k: float(np.mean([r[k] for r in rows if k in r])) if any(k in r for r in rows) else None
        metrics = dict(per_frame=rows, n_frames=self.n_frames_total, test_cam=test_cam,
                       mean=dict(psnr=mean("psnr"), ssim=mean("ssim"), alpha_iou=mean("alpha_iou"), cd=mean("cd")),
                       non_finite_from_frame=nan_from, timings=self.timings, steps=getattr(self, "step", 0))
        json.dump(metrics, open(self.out / "metrics.json", "w"), indent=1)
        log(f"[eval] mean psnr {metrics['mean']['psnr']:.2f} ssim {metrics['mean']['ssim']:.3f} iou {metrics['mean']['alpha_iou']:.3f}"
            + (f" cd {metrics['mean']['cd']:.3f} (10^3 mm^2)" if metrics['mean']['cd'] is not None else "") +
            f" | eval {self.timings['eval_s']:.0f}s | outputs {self.out}", self.logf)
        return metrics


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"config {config_path} not found (run eval/convert_physon_to_omniphysgs.py)")
    out_root = Path(args.output).resolve() if args.output else None
    os.chdir(HERE)   # upstream modules add cwd-relative paths
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE / "third_party" / "gaussian-splatting"))
    cfg = OmegaConf.load(config_path)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    if args.gpu is not None:
        cfg.train.gpu = args.gpu
    if args.tag is not None:
        cfg.train.train_tag = args.tag
    if out_root is not None:
        cfg.train.export_path = str(out_root)
    elif not Path(cfg.train.export_path).is_absolute():
        cfg.train.export_path = str((HERE / cfg.train.export_path).resolve())
    torch.backends.cudnn.benchmark = False
    fitter = Fitter(cfg, config_path.parent, args)
    if args.eval_only:
        last = fitter.latest_epoch_ckpt()
        if last is not None:
            fitter.load_ckpt(last)
        elif (fitter.out / "checkpoints" / "velocity.pth").exists():
            fitter.load_ckpt(fitter.out / "checkpoints" / "velocity.pth")
        else:
            log("[eval] no checkpoint: free rollout with the initial parameters", fitter.logf)
    else:
        fitter.train()
    fitter.evaluate()


if __name__ == "__main__":
    main()
