"""External (non-gravitational) acceleration field of a PhysON scene — standalone copy of
`eval/physon_common.ForceField` for the vendored MOSIV baseline (no ../eval import at runtime).

Grid mode (single-object exports: `grid` + `grid_accel`, gravity included) or carrier mode
(multi-object exports: per-GT-particle `external_accel` on `physics.h5` positions, gravity excluded).
`sample(points, frame)` returns world m/s^2 without gravity; sample t drives frame t -> t+1.
"""
import json
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


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

    def __init__(self, scene_dir, gravity, device="cpu", knn: int = 8, npz_path=None, json_path=None, h5_path=None):
        import torch
        self.device = device
        self.knn = knn
        self.gravity = np.asarray(gravity, dtype=np.float32).reshape(3)
        scene_dir = Path(scene_dir)
        self.mode = None
        self.n_samples = 0
        npz_path = Path(npz_path) if npz_path else scene_dir / "force_field.npz"
        json_path = Path(json_path) if json_path else scene_dir / "force_field.json"
        info = json.load(open(json_path)) if json_path.exists() else {}
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
            h5_path = Path(h5_path) if h5_path else scene_dir / "physics.h5"
            with h5py.File(h5_path, "r") as h5:
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


