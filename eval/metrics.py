"""Shared metrics for baseline evaluation.

Conventions (aligned with MASIV / MOSIV):
- CD: bidirectional mean *squared* nearest-neighbor distance. Scene units are
  meters; we convert to mm and report in 10^3 mm^2 (MASIV Table 1 units).
- EMD: exact assignment on `emd_samples` FPS-free random subsamples, mean
  distance in meters (MOSIV convention).
- PSNR/SSIM on uint8 RGB frames; LPIPS (AlexNet) optional.
Frame alignment is by index; the caller decides the observable/future split.
"""
import json
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment


def load_points(ply_path):
    from plyfile import PlyData
    v = PlyData.read(str(ply_path))["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)


def chamfer_sq_mm2(a, b):
    """Bidirectional mean squared NN distance, input meters -> 10^3 mm^2."""
    da = cKDTree(b).query(a, k=1)[0]
    db = cKDTree(a).query(b, k=1)[0]
    cd_m2 = (da ** 2).mean() + (db ** 2).mean()
    return cd_m2 * 1e6 / 1e3  # m^2 -> mm^2, then report in 10^3 mm^2


def emd_m(a, b, samples=1024, seed=0):
    """Exact-assignment EMD on subsampled clouds, meters."""
    rng = np.random.default_rng(seed)
    a = a[rng.choice(len(a), min(samples, len(a)), replace=False)]
    b = b[rng.choice(len(b), min(samples, len(b)), replace=False)]
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    cost = np.linalg.norm(a[:, None] - b[None], axis=-1)
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].mean())


def psnr_ssim(pred_img, gt_img):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    p = peak_signal_noise_ratio(gt_img, pred_img, data_range=255)
    s = structural_similarity(gt_img, pred_img, data_range=255, channel_axis=-1)
    return float(p), float(s)


_lpips_model = None


def lpips_fn(pred_img, gt_img, device="cuda"):
    global _lpips_model
    import torch, lpips
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net="alex").to(device)
    def to_t(x):
        t = torch.from_numpy(x.astype(np.float32) / 127.5 - 1.0)
        return t.permute(2, 0, 1)[None].to(device)
    with torch.no_grad():
        return float(_lpips_model(to_t(pred_img), to_t(gt_img)).item())


def eval_particle_sequence(pred_plys, gt_plys, split=None, emd_samples=1024):
    """pred_plys/gt_plys: index-aligned lists of ply paths.
    Returns per-frame and windowed means; `split` = first future frame index."""
    n = min(len(pred_plys), len(gt_plys))
    rows = []
    for i in range(n):
        a, b = load_points(pred_plys[i]), load_points(gt_plys[i])
        rows.append({"frame": i,
                     "cd": chamfer_sq_mm2(a, b),
                     "emd": emd_m(a, b, samples=emd_samples)})
    out = {"per_frame": rows, "n_frames": n}
    def mean_over(idx):
        return {k: float(np.mean([r[k] for r in idx])) for k in ("cd", "emd")} if idx else None
    if split is None:
        out["all"] = mean_over(rows)
    else:
        out["observable"] = mean_over([r for r in rows if r["frame"] < split])
        out["future"] = mean_over([r for r in rows if r["frame"] >= split])
    return out


def eval_image_sequence(pred_imgs, gt_imgs, split=None, use_lpips=True, device="cuda"):
    import imageio.v2 as imageio
    n = min(len(pred_imgs), len(gt_imgs))
    rows = []
    for i in range(n):
        p = imageio.imread(pred_imgs[i])[..., :3]
        g = imageio.imread(gt_imgs[i])[..., :3]
        if p.shape != g.shape:
            raise ValueError(f"shape mismatch at {i}: {p.shape} vs {g.shape}")
        ps, ss = psnr_ssim(p, g)
        row = {"frame": i, "psnr": ps, "ssim": ss}
        if use_lpips:
            row["lpips"] = lpips_fn(p, g, device)
        rows.append(row)
    keys = [k for k in ("psnr", "ssim", "lpips") if rows and k in rows[0]]
    out = {"per_frame": rows, "n_frames": n}
    def mean_over(idx):
        return {k: float(np.mean([r[k] for r in idx])) for k in keys} if idx else None
    if split is None:
        out["all"] = mean_over(rows)
    else:
        out["observable"] = mean_over([r for r in rows if r["frame"] < split])
        out["future"] = mean_over([r for r in rows if r["frame"] >= split])
    return out


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
