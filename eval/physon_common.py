#!/usr/bin/env python
"""Shared reader for PhysON scenes (HF cmu-robotics-institute/PhysON, "MASIV layout +
masiv_physctrl_force_field_v1").

A scene directory looks like::

    <scene>/all_data.json            PAC-NeRF style camera list (file_path r_<cam>_<frame>.png,
                                     time in [0,1], c2w 3x4|4x4, intrinsic 3x3)
    <scene>/transforms_{train,val,test}.json   Blender-style split lists (test = held-out camera)
    <scene>/data/{a,r,m}_<cam>_<frame>.png     a: RGBA (RGB + soft alpha), r: RGB with background,
                                     m: RGB on white (mask rule: any channel != 255)
    <scene>/point_clouds/<frame>.ply GT particles (world metres, right-handed y-up)
    <scene>/physics.h5               GT particle states x[T,N,3], v, per-particle material, timing/*,
                                     force_field/* (mirror of force_field.npz), gravity_accel
    <scene>/force_field.npz          applied_accel [T-1,N,3] (includes gravity) and, for the
                                     analytic single-object exports, grid [4096,3] + grid_accel
                                     [T,4096,3] (includes gravity) on a 16^3 lattice with bbox_min/max
    <scene>/metadata.json, spec.json, meta.json (subset dependent)

Only the observable inputs (images, cameras, timing, the declared force field and the scene bounds)
are meant to be consumed by a baseline at fit time; GT particles are for evaluation.

The module is dependency-light (numpy, plyfile, optional h5py/torch) so both converters and the
vendored baselines can use it or copy the pieces they need.
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:  # h5py is only needed for physics.h5 (force carriers / gravity); converters install it.
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


# ----------------------------------------------------------------------------- cameras

def parse_frame_name(file_path: str) -> Tuple[str, int, int]:
    """'./data/r_3_12.png' -> ('r', 3, 12)."""
    stem = Path(file_path).stem
    kind, cam, frame = stem.split("_")
    return kind, int(cam), int(frame)


def gs_camera_RT(c2w) -> Tuple[np.ndarray, np.ndarray]:
    """PAC-NeRF/GIC/MASIV `all_data.json` c2w  ->  3DGS (R, T) as MASIV's reader does it.

    3DGS stores R transposed (world_view_transform = [R^T | T]); this is exactly MASIV's
    readCamerasFromAllData formula, so cameras built with it match the reconstructions that ship
    with the dataset.
    """
    c2w = np.array(c2w, dtype=np.float64)
    if c2w.shape[0] == 3:
        c2w = np.vstack([c2w, [0.0, 0.0, 0.0, 1.0]])
    matrix = np.linalg.inv(c2w)
    R = -np.transpose(matrix[:3, :3])
    R[:, 0] = -R[:, 0]
    T = -matrix[:3, 3]
    return R, T


def blender_c2w_from_RT(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Inverse of the original 3DGS Blender loader (readCamerasFromTransforms): returns a 4x4
    camera-to-world such that the loader recovers exactly (R, T)."""
    w2c = np.eye(4)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = T
    c2w = np.linalg.inv(w2c)
    c2w[:3, 1:3] *= -1  # loader flips them back
    return c2w


def camera_center_from_RT(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    w2c = np.eye(4)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = T
    return np.linalg.inv(w2c)[:3, 3]


@dataclass
class PhysonCamera:
    cam_id: int
    R: np.ndarray            # 3x3, 3DGS convention (stored transposed)
    T: np.ndarray            # 3
    K: np.ndarray            # 3x3 intrinsic
    width: int
    height: int

    @property
    def fovx(self) -> float:
        return 2.0 * float(np.arctan(self.width / (2.0 * self.K[0, 0])))

    @property
    def fovy(self) -> float:
        return 2.0 * float(np.arctan(self.height / (2.0 * self.K[1, 1])))

    @property
    def center(self) -> np.ndarray:
        return camera_center_from_RT(self.R, self.T)

    def to_json(self) -> dict:
        return dict(cam_id=self.cam_id, R=self.R.tolist(), T=self.T.tolist(), K=self.K.tolist(),
                    width=self.width, height=self.height, fovx=self.fovx, fovy=self.fovy)

    @classmethod
    def from_json(cls, d: dict) -> "PhysonCamera":
        return cls(int(d["cam_id"]), np.array(d["R"]), np.array(d["T"]), np.array(d["K"]),
                   int(d["width"]), int(d["height"]))


# ----------------------------------------------------------------------------- scene

@dataclass
class PhysonScene:
    scene_dir: Path
    metadata: dict
    cameras: Dict[int, PhysonCamera]
    frame_ids: List[int]
    frame_times: Dict[int, float]         # normalized time from all_data.json (0..1)
    test_cam_ids: List[int]
    sim_fps: float                        # physical observation rate (frames per second)
    gravity: np.ndarray                   # world m/s^2, y-up
    floor_height: float
    particle_size: float
    bbox_min: np.ndarray                  # frame-0 GT particle bounds (scene bounding box prior)
    bbox_max: np.ndarray
    regions: list = field(default_factory=list)
    spec: Optional[dict] = None

    @property
    def n_frames(self) -> int:
        return len(self.frame_ids)

    @property
    def frame_dt(self) -> float:
        return 1.0 / self.sim_fps

    @property
    def train_cam_ids(self) -> List[int]:
        return [c for c in sorted(self.cameras) if c not in self.test_cam_ids]

    def image_path(self, cam: int, frame: int, kind: str = "a") -> Path:
        return self.scene_dir / "data" / f"{kind}_{cam}_{frame}.png"

    def gt_ply_path(self, frame: int) -> Path:
        return self.scene_dir / "point_clouds" / f"{frame}.ply"

    def has_force(self) -> bool:
        return (self.scene_dir / "force_field.npz").is_file()

    def summary(self) -> dict:
        return dict(scene=str(self.scene_dir), n_frames=self.n_frames, sim_fps=self.sim_fps,
                    cameras=sorted(self.cameras), test_cams=self.test_cam_ids,
                    gravity=self.gravity.tolist(), floor=self.floor_height,
                    particle_size=self.particle_size, bbox_min=self.bbox_min.tolist(),
                    bbox_max=self.bbox_max.tolist(), regions=self.regions, has_force=self.has_force())


def _read_h5_scalar(h5, key, default=None):
    if h5 is not None and key in h5:
        v = h5[key][()]
        return v
    return default


def read_ply_xyz(path) -> np.ndarray:
    from plyfile import PlyData
    v = PlyData.read(str(path))["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)


def write_ply_xyz(path, xyz: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    from plyfile import PlyData, PlyElement
    xyz = np.asarray(xyz, dtype=np.float32)
    if colors is None:
        arr = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    else:
        colors = np.asarray(colors)
        if colors.dtype != np.uint8:
            colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
        arr = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                         ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        arr["red"], arr["green"], arr["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def load_scene(scene_dir) -> PhysonScene:
    scene_dir = Path(scene_dir).resolve()
    meta = json.load(open(scene_dir / "metadata.json"))
    spec = json.load(open(scene_dir / "spec.json")) if (scene_dir / "spec.json").exists() else None

    # cameras + frames from all_data.json (background frame -1 skipped)
    cams: Dict[int, PhysonCamera] = {}
    frame_times: Dict[int, float] = {}
    for e in json.load(open(scene_dir / "all_data.json")):
        kind, cam, frame = parse_frame_name(e["file_path"])
        if frame < 0:
            continue
        frame_times[frame] = float(e["time"])
        if cam not in cams:
            R, T = gs_camera_RT(e["c2w"])
            K = np.array(e["intrinsic"], dtype=np.float64)
            res = meta.get("resolution", [800, 800])
            cams[cam] = PhysonCamera(cam, R, T, K, int(res[0]), int(res[1]))
        else:
            R, T = gs_camera_RT(e["c2w"])
            assert np.allclose(cams[cam].R, R, atol=1e-6) and np.allclose(cams[cam].T, T, atol=1e-6), \
                f"camera {cam} moves over time in {scene_dir}"
            assert np.allclose(cams[cam].K, np.array(e["intrinsic"]), atol=1e-3), \
                f"camera {cam} intrinsics vary over time in {scene_dir}"
    frame_ids = sorted(frame_times)
    assert frame_ids == list(range(len(frame_ids))), f"non-contiguous frames in {scene_dir}"

    # held-out camera(s)
    test_cam_ids: List[int] = []
    tf = scene_dir / "transforms_test.json"
    if tf.exists():
        test_cam_ids = sorted({parse_frame_name(f["file_path"])[1] for f in json.load(open(tf))["frames"]})

    # physical clock / gravity / floor from physics.h5 when present, metadata otherwise
    h5 = h5py.File(scene_dir / "physics.h5", "r") if (h5py is not None and (scene_dir / "physics.h5").exists()) else None
    sim_fps = _read_h5_scalar(h5, "timing/simulation_fps")
    if sim_fps is None:
        sim_fps = meta.get("simulation_fps", meta.get("fps"))
    sim_fps = float(sim_fps)
    grav = _read_h5_scalar(h5, "gravity_accel")
    if grav is None:
        grav = meta.get("gravity_accel")
    if grav is None:
        g = float(meta.get("gravity", -9.8))
        grav = [0.0, g, 0.0]
    gravity = np.array(grav, dtype=np.float32).reshape(3)
    floor = _read_h5_scalar(h5, "floor_height", 0.0)
    if h5 is not None:
        h5.close()
    floor = float(floor)

    particle_size = float(meta.get("particle_size", (spec or {}).get("geometry", {}).get("voxel_size", 0.02)))
    x0 = read_ply_xyz(scene_dir / "point_clouds" / "0.ply")
    bbox_min, bbox_max = x0.min(0), x0.max(0)

    regions = meta.get("material_regions") or meta.get("regions") or meta.get("objects") or []
    return PhysonScene(scene_dir, meta, cams, frame_ids, frame_times, test_cam_ids, sim_fps, gravity,
                       floor, particle_size, bbox_min, bbox_max, regions, spec)


def load_rgba(path) -> np.ndarray:
    """float32 HxWx4 in [0,1]; m_/r_ files get alpha from the white-background rule."""
    from PIL import Image
    im = Image.open(path)
    arr = np.asarray(im.convert("RGBA")).astype(np.float32) / 255.0
    if im.mode != "RGBA":
        arr[..., 3] = (np.asarray(im.convert("RGB")).astype(np.int32).sum(-1) != 255 * 3).astype(np.float32)
    return arr


# ----------------------------------------------------------------------------- force field

class ForceField:
    """Non-gravitational external acceleration a_ext(x, t) [m/s^2, world frame] of a PhysON scene.

    Sources, in order of preference:
      * grid mode:   force_field.npz grid [M,3] + grid_accel [T,M,3] (regular lattice, trilinear)
      * carrier mode: physics.h5 x[T,N,3] positions carrying external_accel [T,N,3]
                     (or applied_accel [T-1,N,3]) -> inverse-distance k-NN interpolation.
    Gravity is subtracted when the export says the arrays include it, so that simulators keep
    applying their own gravity. Sample t is the frame-start value that drives frame t -> t+1
    (dataset provenance note); frames past the last sample reuse the last sample.
    """

    def __init__(self, scene: PhysonScene, device="cpu", knn: int = 8):
        import torch
        self.device = device
        self.knn = knn
        self.gravity = scene.gravity
        self.mode = None
        self.n_samples = 0
        npz_path = scene.scene_dir / "force_field.npz"
        info = json.load(open(scene.scene_dir / "force_field.json")) if (scene.scene_dir / "force_field.json").exists() else {}
        self.preset = info.get("preset", "unknown")
        includes_g = bool(info.get("includes_gravity", True))
        if not npz_path.exists():
            return
        z = np.load(npz_path)
        g = torch.as_tensor(self.gravity, dtype=torch.float32, device=device)
        if "grid" in z.files and "grid_accel" in z.files:
            grid = z["grid"].astype(np.float32)
            acc = z["grid_accel"].astype(np.float32)  # [T, M, 3]
            axes = [np.unique(np.round(grid[:, i], 6)) for i in range(3)]
            n = [len(a) for a in axes]
            assert int(np.prod(n)) == grid.shape[0], "force grid is not a full regular lattice"
            idx = [np.searchsorted(axes[i], np.round(grid[:, i], 6)) for i in range(3)]
            vol = np.zeros((acc.shape[0], n[0], n[1], n[2], 3), dtype=np.float32)
            vol[:, idx[0], idx[1], idx[2]] = acc
            self.vol = torch.as_tensor(vol, device=device)
            if includes_g:
                self.vol = self.vol - g
            self.axis_min = torch.tensor([a[0] for a in axes], device=device)
            self.axis_max = torch.tensor([a[-1] for a in axes], device=device)
            self.n = torch.tensor(n, device=device)
            self.mode = "grid"
            self.n_samples = acc.shape[0]
        else:
            assert h5py is not None, "carrier-mode force sampling needs h5py"
            with h5py.File(scene.scene_dir / "physics.h5", "r") as h5:
                x = h5["x"][...].astype(np.float32)
            if "external_accel" in z.files:
                acc = z["external_accel"].astype(np.float32)
                sub_g = False
            else:
                acc = z["applied_accel"].astype(np.float32)
                sub_g = includes_g
            T = min(len(x), len(acc))
            self.carriers = torch.as_tensor(x[:T], device=device)
            a = torch.as_tensor(acc[:T], device=device)
            self.values = a - g if sub_g else a
            self.mode = "carriers"
            self.n_samples = T

    @property
    def available(self) -> bool:
        return self.mode is not None

    def _frame(self, frame: int) -> int:
        return int(min(max(frame, 0), self.n_samples - 1))

    def sample(self, points, frame: int):
        """points [M,3] world metres (torch or numpy) -> [M,3] torch external acceleration."""
        import torch
        p = torch.as_tensor(np.asarray(points) if not torch.is_tensor(points) else points,
                            dtype=torch.float32, device=self.device)
        if not self.available:
            return torch.zeros_like(p)
        f = self._frame(frame)
        if self.mode == "grid":
            return self._sample_grid(p, f)
        return self._sample_carriers(p, f)

    def _sample_grid(self, p, f):
        import torch
        vol = self.vol[f]  # [nx, ny, nz, 3]
        u = (p - self.axis_min) / (self.axis_max - self.axis_min) * (self.n - 1).float()
        u = u.clamp(torch.zeros(3, device=p.device), (self.n - 1).float())  # border padding
        i0 = u.floor().long()
        i1 = torch.minimum(i0 + 1, self.n - 1)
        w1 = (u - i0.float())
        w0 = 1.0 - w1
        out = torch.zeros_like(p)
        for dx, wx in ((i0[:, 0], w0[:, 0]), (i1[:, 0], w1[:, 0])):
            for dy, wy in ((i0[:, 1], w0[:, 1]), (i1[:, 1], w1[:, 1])):
                for dz, wz in ((i0[:, 2], w0[:, 2]), (i1[:, 2], w1[:, 2])):
                    out = out + (wx * wy * wz)[:, None] * vol[dx, dy, dz]
        return out

    def _sample_carriers(self, p, f, chunk: int = 8192):
        import torch
        c = self.carriers[f]  # [N,3]
        v = self.values[f]    # [N,3]
        k = min(self.knn, c.shape[0])
        out = torch.empty_like(p)
        for s in range(0, p.shape[0], chunk):
            q = p[s:s + chunk]
            d2 = torch.cdist(q, c).pow(2)  # [m, N]
            dk, ik = d2.topk(k, dim=1, largest=False)
            w = 1.0 / (dk + 1e-6)
            w = w / w.sum(1, keepdim=True)
            out[s:s + chunk] = (w[..., None] * v[ik]).sum(1)
        return out

    def describe(self) -> dict:
        return dict(mode=self.mode, preset=self.preset, n_samples=self.n_samples,
                    gravity=self.gravity.tolist())


if __name__ == "__main__":  # quick inspection: python physon_common.py <scene_dir>
    import sys
    s = load_scene(sys.argv[1])
    print(json.dumps(s.summary(), indent=1))
    ff = ForceField(s)
    print(ff.describe())
    if ff.available:
        x0 = read_ply_xyz(s.gt_ply_path(0))
        for f in (0, s.n_frames // 2, s.n_frames - 2):
            a = ff.sample(x0, f)
            print(f"frame {f}: mean |a_ext| {a.norm(dim=1).mean():.3f}  max {a.norm(dim=1).max():.3f}")
