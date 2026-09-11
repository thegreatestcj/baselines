"""Shared machinery for the *_to_vid2sim benchmark converters.

Vid2Sim's Stage-II 3DGS refinement initializes VERBATIM from the LGM
prediction (gs/scene/__init__.py -> create_from_pcd_lgm), and LGM emits its
gaussians in a fixed canonical frame: object centered at the origin with
half-extent ~0.33 (measured on the finished GSO bus run), z-up, and the first
input view looking from -y. The stock GSO dataset world IS that frame (view 0
at (0,-1.8,-0.2), floor_axis 2, objects falling toward -z), which is why the
stock pipeline works without any alignment step.

A converted benchmark therefore maps its world into the same frame with a
similarity transform

    x_sim = s * R @ (x_world - c)

where c is the object's frame-0 center (visual hull), R rotates the
benchmark's y-up world to z-up with the chosen view-0 azimuth onto -y, and
s = TARGET_HALF / hull half-extent. Physics stays consistent by scaling time
(the generated config's delta_t = frame_dt * sqrt(s), keeping g = 9.8 as
hardcoded in LBSSimulator), which rescales Young's modulus: E_sim = s * E_gt
(nu unchanged). The transform is recorded in the case dir's
sim_transform.json so eval/vid2sim_future.py can map simulated point clouds
back to the benchmark world for particle metrics.
"""
import json
import math
import numpy as np
from pathlib import Path

# Half-extent of LGM's canonical object box (measured from GSO bus pred.ply).
TARGET_HALF = 0.33


# --------------------------------------------------------------- geometry

def cam_azel(pos, center):
    """Azimuth/elevation (deg) of a camera position in a y-up world,
    kiui/LGM convention: az 0 -> camera at +z, az 90 -> camera at +x."""
    d = np.asarray(pos, dtype=np.float64) - center
    az = math.degrees(math.atan2(d[0], d[2]))
    el = math.degrees(math.asin(d[1] / np.linalg.norm(d)))
    return az, el


def _wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


def pick_canonical_views(cam_pos, center, elev_weight=1.5):
    """Pick 4 cams closest to a front/right/back/left orbit (90 deg apart,
    low elevation) for LGM. cam_pos: {cam_id: xyz}. Returns [id0..id3]."""
    azel = {i: cam_azel(p, center) for i, p in cam_pos.items()}
    best, best_cost = None, None
    for anchor in azel:
        a0 = azel[anchor][0]
        chosen, cost, used = [], 0.0, set()
        for slot in range(4):
            target = a0 + 90.0 * slot
            cand = min((i for i in azel if i not in used),
                       key=lambda i: abs(_wrap(azel[i][0] - target)) + elev_weight * abs(azel[i][1]))
            chosen.append(cand)
            used.add(cand)
            cost += abs(_wrap(azel[cand][0] - target)) + elev_weight * abs(azel[cand][1])
        if best_cost is None or cost < best_cost:
            best, best_cost = chosen, cost
    for slot, i in enumerate(best):
        az, el = azel[i]
        print(f"[views] slot {slot} (target az {_wrap(azel[best[0]][0] + 90 * slot):+7.1f}): "
              f"cam {i} az {az:+7.1f} elev {el:+6.1f}")
    return best


def build_world_rotation(view0_pos, center):
    """R with x_sim = R @ x_world mapping y-up world -> Vid2Sim/LGM canonical
    (z-up, view-0 camera on -y looking toward +y)."""
    d0 = center - np.asarray(view0_pos, dtype=np.float64)
    d0[1] = 0.0
    d0 /= np.linalg.norm(d0)
    up = np.array([0.0, 1.0, 0.0])
    e1, e2 = d0, up                       # -> +y, +z
    e3 = np.cross(e1, e2)                 # -> +x
    return np.stack([e3, e1, e2], axis=0)


def visual_hull(cams, mask_fn, box_min, box_max, grid=96, margin=0.5, frac=0.99):
    """Frame-0 visual hull center / half-extent.

    cams: {cam_id: (c2w 4x4 OpenGL, K 3x3)}; mask_fn(cam_id) -> bool HxW.
    The box is expanded by `margin` before carving. Returns (center,
    half_extent, n_inside)."""
    box_min = np.asarray(box_min, dtype=np.float64)
    box_max = np.asarray(box_max, dtype=np.float64)
    c, h = (box_min + box_max) / 2, (box_max - box_min) / 2
    box_min, box_max = c - (1 + margin) * h, c + (1 + margin) * h

    axes = [np.linspace(box_min[k], box_max[k], grid) for k in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    inside = np.zeros(len(pts), dtype=np.int32)

    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    for cam_id, (c2w, K) in cams.items():
        w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float64) @ flip)
        Xc = pts @ w2c[:3, :3].T + w2c[:3, 3]
        z = Xc[:, 2]
        uv = Xc @ np.asarray(K, dtype=np.float64).T
        u = uv[:, 0] / np.clip(uv[:, 2], 1e-9, None)
        v = uv[:, 1] / np.clip(uv[:, 2], 1e-9, None)
        mask = mask_fn(cam_id)
        H, W = mask.shape
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        ok = (z > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        hit = np.zeros(len(pts), dtype=bool)
        hit[ok] = mask[vi[ok], ui[ok]]
        inside += hit

    for thr in (frac, 0.9, 0.75):
        keep = inside >= math.ceil(thr * len(cams))
        if keep.sum() >= 20:
            break
    assert keep.sum() >= 20, "visual hull empty -- check masks/cameras"
    hull = pts[keep]
    lo, hi = hull.min(0), hull.max(0)
    center, half = (lo + hi) / 2, float((hi - lo).max() / 2)
    print(f"[hull] {keep.sum()} / {len(pts)} grid points (thr {thr}), "
          f"center {center.round(3).tolist()}, half-extent {half:.3f}")
    return center, half, int(keep.sum())


# --------------------------------------------------------------- writers

def transform_c2w(c2w, R, s, c):
    """Apply x_sim = s*R@(x-c) to an OpenGL c2w (4x4)."""
    c2w = np.asarray(c2w, dtype=np.float64)
    out = np.eye(4)
    out[:3, :3] = R @ c2w[:3, :3]
    out[:3, 3] = s * (R @ (c2w[:3, 3] - c))
    return out


def write_transforms(case_dir: Path, camera_angle_x, view_c2ws):
    """GSO-style transforms_{train,test,val,simulation}.json: every dataset
    view at frame 0, file_path ./data/m_<view>_0. view_c2ws: list of 4x4
    (already in sim frame), index = new view id."""
    frames = [{"file_path": f"./data/m_{v}_0", "time": 0.0, "rotation": 0.0,
               "transform_matrix": [list(map(float, row)) for row in c2w]}
              for v, c2w in enumerate(view_c2ws)]
    content = {"camera_angle_x": float(camera_angle_x), "frames": frames}
    for name in ("transforms_train", "transforms_test", "transforms_val",
                 "transforms_simulation"):
        with open(case_dir / f"{name}.json", "w") as f:
            json.dump(content, f, indent=2)
    print(f"[transforms] {len(frames)} views, camera_angle_x={camera_angle_x:.6f}")


def write_points3d(case_dir: Path, num_pts=100_000, half=1.3, seed=0):
    """Random init cloud like GSO's points3d.ply (unused by the pipeline --
    refine_gs initializes from LGM -- but the Blender loader wants the file)."""
    from plyfile import PlyData, PlyElement
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-half, half, size=(num_pts, 3))
    rgb = rng.uniform(0, 1, size=(num_pts, 3)) * 255
    arr = np.empty(num_pts, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                   ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr["x"], arr["y"], arr["z"] = xyz.T.astype(np.float32)
    arr["nx"] = arr["ny"] = arr["nz"] = 0.0
    arr["red"], arr["green"], arr["blue"] = rgb.T.astype(np.uint8)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(case_dir / "points3d.ply"))


def write_sim_transform(case_dir: Path, s, c, R):
    with open(case_dir / "sim_transform.json", "w") as f:
        json.dump({"comment": "x_sim = scale * rotation @ (x_world - center)",
                   "scale": float(s),
                   "center": [float(v) for v in c],
                   "rotation": [[float(v) for v in row] for row in R]}, f, indent=2)


def write_config(template_path: Path, out_path: Path, **overrides):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(template_path)
    for k, v in overrides.items():
        cfg[k] = v
    out_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_path)
    print(f"[config] {out_path}: " + ", ".join(f"{k}={v}" for k, v in overrides.items()))


def write_gt_phys_params(case_dir: Path, yms, prs, s, note):
    """gt_phys_params.yaml like GSO's. yms/prs may be None (placeholder -1).
    yms_sim is the value comparable to the recovered best_params.yaml
    (E scales with the world: E_sim = s * E_gt)."""
    lines = [f"# {ln}" for ln in note.splitlines()]
    lines += [f"yms: {yms if yms is not None else -1}",
              f"prs: {prs if prs is not None else -1}",
              f"yms_sim: {yms * s if yms is not None else -1}",
              f"world_scale: {s}"]
    (case_dir / "gt_phys_params.yaml").write_text("\n".join(lines) + "\n")
