"""PhysON package reader for the OmniPhysGS baseline (self-contained; no ../eval imports).

A package is written by eval/convert_physon_to_omniphysgs.py:
    <pkg>/scene.json      cameras (3DGS R, T, K, fov), timing, gravity, floor, bbox, force paths
    <pkg>/source/         -> the scene data (data/a_<cam>_<frame>.png RGBA, point_clouds/<f>.ply, ...)
    <pkg>/gs_model/       static frame-0 3DGS (recon_static.py)
    <pkg>/config.yaml     fit.py config
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[2]
GS_DIR = HERE / "third_party" / "gaussian-splatting"
if str(GS_DIR) not in sys.path:
    sys.path.append(str(GS_DIR))


def read_ply_xyz(path) -> np.ndarray:
    from plyfile import PlyData
    v = PlyData.read(str(path))["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)


def write_ply_xyz(path, xyz, colors=None) -> None:
    from plyfile import PlyData, PlyElement
    xyz = np.asarray(xyz, dtype=np.float32)
    if colors is None:
        arr = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    else:
        colors = np.asarray(colors)
        if colors.dtype != np.uint8:
            colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
        arr = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                         ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        arr["red"], arr["green"], arr["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def load_rgba_u8(path) -> np.ndarray:
    """HxWx4 uint8 (RGB + soft alpha); m_/r_ files get alpha from the white-background rule."""
    from PIL import Image
    im = Image.open(path)
    if im.mode == "RGBA":
        return np.asarray(im)
    rgb = np.asarray(im.convert("RGB"))
    a = ((rgb.astype(np.int32).sum(-1) != 255 * 3) * 255).astype(np.uint8)
    return np.concatenate([rgb, a[..., None]], -1)


class PhysonPackage:
    def __init__(self, pkg):
        self.pkg = Path(pkg).resolve()
        self.meta = json.load(open(self.pkg / "scene.json"))
        self.source = self.pkg / "source"
        self.cameras = {int(c["cam_id"]): c for c in self.meta["cameras"]}
        self.test_cam_ids = [int(c) for c in self.meta["test_cam_ids"]]
        self.train_cam_ids = [c for c in sorted(self.cameras) if c not in self.test_cam_ids]
        self.n_frames = int(self.meta["n_frames"])
        self.frame_dt = float(self.meta["frame_dt"])
        self.gravity = np.array(self.meta["gravity"], dtype=np.float32)
        self.floor_height = float(self.meta["floor_height"])
        self.bbox_min = np.array(self.meta["bbox_min"], dtype=np.float32)
        self.bbox_max = np.array(self.meta["bbox_max"], dtype=np.float32)
        self._gs_cams = {}

    # ---- cameras
    def gs_camera(self, cam_id: int):
        """Vendored 3DGS `Camera` (image is a dummy; only the matrices are used)."""
        if cam_id not in self._gs_cams:
            from scene.cameras import Camera
            c = self.cameras[cam_id]
            dummy = torch.zeros(3, int(c["height"]), int(c["width"]))
            self._gs_cams[cam_id] = Camera(colmap_id=cam_id, R=np.array(c["R"]), T=np.array(c["T"]),
                                           FoVx=float(c["fovx"]), FoVy=float(c["fovy"]), image=dummy,
                                           gt_alpha_mask=None, image_name=f"cam_{cam_id}", uid=cam_id,
                                           data_device="cuda")
        return self._gs_cams[cam_id]

    # ---- images
    def image_path(self, cam: int, frame: int, kind: str = "a") -> Path:
        return self.source / "data" / f"{kind}_{cam}_{frame}.png"

    def gt_rgba_u8(self, cam: int, frame: int) -> np.ndarray:
        return load_rgba_u8(self.image_path(cam, frame))

    def gt_rgb(self, cam: int, frame: int) -> torch.Tensor:
        """[3,H,W] float in [0,1], object composited on white."""
        return composite_white(torch.from_numpy(self.gt_rgba_u8(cam, frame).copy()))[0]

    def load_gt_stack(self, cam_ids, frames, workers: int = 16) -> torch.Tensor:
        """uint8 [n_cams, n_frames, H, W, 4] on the CPU (pinned when CUDA is available)."""
        jobs = [(ci, f) for ci in range(len(cam_ids)) for f in frames]
        with ThreadPoolExecutor(workers) as ex:
            imgs = list(ex.map(lambda j: self.gt_rgba_u8(cam_ids[j[0]], j[1]), jobs))
        H, W = imgs[0].shape[:2]
        out = torch.empty((len(cam_ids), len(frames), H, W, 4), dtype=torch.uint8)
        for (ci, fi), im in zip(((j[0], k % len(frames)) for k, j in enumerate(jobs)), imgs):
            out[ci, fi] = torch.from_numpy(np.ascontiguousarray(im))
        if torch.cuda.is_available():
            out = out.pin_memory()
        return out

    # ---- GT particles
    def gt_ply_path(self, frame: int) -> Path:
        return self.source / "point_clouds" / f"{frame}.ply"

    def has_gt_particles(self) -> bool:
        return self.gt_ply_path(0).exists()


def composite_white(rgba_u8: torch.Tensor):
    """uint8 [...,H,W,4] -> float [...,3,H,W] RGB on white, and [...,1,H,W] soft alpha."""
    x = rgba_u8.float() / 255.0
    rgb, a = x[..., :3], x[..., 3:4]
    rgb = rgb * a + (1.0 - a)
    return rgb.movedim(-1, -3), a.movedim(-1, -3)
