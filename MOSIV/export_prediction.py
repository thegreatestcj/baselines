#!/usr/bin/env python
"""Roll out the fitted MOSIV scene and export what the baselines evaluation needs (PhysON addition).

Reads `<model_path>/<id>-pred.json` (best per-object parameters + velocities written by
train_dynamic_MO.py) and `<model_path>/mpm/multi_object_0.ply` (the lifted particles with object
labels), re-simulates all frames with the same estimator/simulator (external force included) and
writes, under <model_path>:

  mpm/simulation_<f>.ply          all particles per frame, world metres  (eval/eval_scene.py --pred_plys)
  mpm/object<k>_<f>.ply           per-object particles
  img_render/<view>_<f>.png       silhouette render of the held-out camera (particles as small
  img_render/<view>_<f>_mask.png  isotropic Gaussians, colour per object) and its alpha
  prediction_metrics.json         per-frame Chamfer (10^3 mm^2, 8192 samples) vs the GT particles,
                                  overall and per object, silhouette IoU of the held-out camera

  python export_prediction.py -c config/physon/<subset>/<scene>.json -s data/PhysON_mosiv/<subset>/<scene> \
      -m output/physon/<subset>/<scene> --view_id 0
"""
import argparse
import json
import math
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def chamfer_sq_mm2(a, b, samples=8192, seed=0):
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    if len(a) > samples:
        a = a[rng.choice(len(a), samples, replace=False)]
    if len(b) > samples:
        b = b[rng.choice(len(b), samples, replace=False)]
    da = cKDTree(b).query(a, k=1)[0]
    db = cKDTree(a).query(b, k=1)[0]
    return float(((da ** 2).mean() + (db ** 2).mean()) * 1e6 / 1e3)


def read_xyz(path):
    from plyfile import PlyData
    v = PlyData.read(str(path))["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)


def write_xyz(path, xyz, colors=None):
    from plyfile import PlyData, PlyElement
    xyz = np.asarray(xyz, np.float32)
    if colors is None:
        arr = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    else:
        arr = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        arr["red"], arr["green"], arr["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def apply_pred(phys_args, pred):
    """Overwrite the config's sub_objects with the fitted values (same mapping as --resume_from_pred)."""
    for po in pred.get("sub_objects", []):
        for so in phys_args.sub_objects:
            if so["object_id"] != po["object_id"]:
                continue
            if "vel" in po:
                so["init_vel"] = po["vel"]
            mp = po.get("mat_params", {})
            m = mp.get("material", so["material"])
            so["material"] = m
            if "rho" in mp:
                so["rho"] = mp["rho"]
            if m in (10, 12, 13):
                if "E" in mp: so["init_E"] = mp["E"]
                if "nu" in mp: so["init_nu"] = mp["nu"]
            if m == 12 and "yield_stress" in mp:
                so["init_yield_stress"] = mp["yield_stress"]
            if m == 13 and "friction_alpha" in mp:
                so["init_friction_alpha"] = mp["friction_alpha"]
            if m in (11, 14):
                if "mu" in mp: so["mu"] = mp["mu"]
                if "kappa" in mp: so["kappa"] = mp["kappa"]
            if m == 14:
                if "plastic_viscosity" in mp: so["init_plastic_viscosity"] = mp["plastic_viscosity"]
                if "yield_stress" in mp: so["init_yield_stress"] = mp["yield_stress"]
    return phys_args


def build_camera(source, view_id):
    """3DGS Camera of camera `view_id` from all_data.json (PAC-NeRF convention, as MOSIV's reader)."""
    from scene.cameras import Camera
    from utils.graphics_utils import focal2fov
    entries = json.load(open(Path(source) / "all_data.json"))
    for e in entries:
        stem = Path(e["file_path"]).stem
        _, cam, frame = stem.split("_")
        if int(cam) == view_id and int(frame) >= 0:
            c2w = [list(map(float, r)) for r in e["c2w"]]
            if len(c2w) == 3:
                c2w.append([0.0, 0.0, 0.0, 1.0])
            matrix = np.linalg.inv(np.array(c2w))
            R = -np.transpose(matrix[:3, :3]); R[:, 0] = -R[:, 0]; T = -matrix[:3, 3]
            K = np.array(e["intrinsic"])
            from PIL import Image
            im = Image.open(Path(source) / e["file_path"].replace("./", ""))
            W, H = im.size
            fovx, fovy = focal2fov(K[0, 0], W), focal2fov(K[1, 1], H)
            dummy = torch.zeros(3, H, W)
            cam_obj = Camera(colmap_id=view_id, R=R, T=T, FoVx=fovy, FoVy=fovx, image=dummy,  # MOSIV swaps them; square images
                             gt_alpha_mask=np.ones((1, H, W), np.float32), image_name=f"cam_{view_id}", uid=view_id, fid=0.0)
            return cam_obj, H, W
    raise RuntimeError(f"camera {view_id} not found in all_data.json")


def particle_gaussians(xyz, labels, radius, colors):
    from scene.gaussian_model import GaussianModel
    from utils.graphics_utils import BasicPointCloud
    from utils.general_utils import inverse_sigmoid
    col = np.asarray([colors[int(l) - 1 if int(l) >= 1 else 0] for l in labels], np.float32)
    g = GaussianModel(0)
    g.create_from_pcd(BasicPointCloud(points=xyz, colors=col, normals=np.zeros_like(xyz)), 1.0)
    with torch.no_grad():
        g._scaling.data = torch.full_like(g._scaling, math.log(radius))
        g._opacity.data = inverse_sigmoid(torch.full_like(g._opacity, 0.95))
    return g


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config_path", required=True)
    p.add_argument("-s", "--source_path", required=True)
    p.add_argument("-m", "--model_path", required=True)
    p.add_argument("--view_id", type=int, default=None, help="held-out camera (default: physics.test_cam_ids[0] or 0)")
    p.add_argument("--frames", type=int, default=None, help="frames to roll out (default: all GT frames)")
    p.add_argument("--radius", type=float, default=0.008, help="render radius of a particle (m)")
    p.add_argument("--no_render", action="store_true")
    a = p.parse_args()
    os.chdir(HERE); sys.path.insert(0, str(HERE))
    import taichi as ti
    if os.environ.get("TI_DEVICE_MEMORY_GB"):
        ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_GB=float(os.environ["TI_DEVICE_MEMORY_GB"]))
    else:
        ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_fraction=0.5)
    from simulator.estimator_multi import Estimator
    from utils.system_utils import read_ply_with_labels

    cfg = json.load(open(a.config_path))
    phys = Namespace(**cfg["physics"])
    model = Path(a.model_path).resolve()
    pred_path = model / f"{phys.id}-pred.json"
    if not pred_path.exists():
        raise FileNotFoundError(f"{pred_path} missing: train_dynamic_MO.py has not finished")
    pred = json.load(open(pred_path))
    phys = apply_pred(phys, pred)
    print("[export] fitted parameters:", json.dumps(pred.get("sub_objects"), indent=None))

    xyz, labels = read_ply_with_labels(str(model / "mpm" / "multi_object_0.ply"))
    if labels is None:
        raise RuntimeError("multi_object_0.ply has no object_id labels")
    vol = torch.from_numpy(xyz.astype(np.float32)).cuda().contiguous()
    labels_t = torch.from_numpy(labels.astype(np.int32)).cuda()
    mats = torch.zeros(len(xyz), dtype=torch.int32, device="cuda")
    for so in phys.sub_objects:
        mats[labels_t == int(so["object_id"])] = int(so["material"])
    est = Estimator(phys, "float32", [], init_vol=vol, gts_per_object=None, surface_index=None, dynamic_scene=None,
                    image_scale=1.0, pipeline=None, image_op=None, particle_materials=mats, object_labels=labels_t)
    est.initialize()

    source = Path(a.source_path).resolve()
    gt_dirs = [source / "point_clouds" / str(k) for k in range(2)]
    n_gt = len(list(gt_dirs[0].glob("*.ply"))) if gt_dirs[0].exists() else int(getattr(phys, "n_frames", 30))
    frames = int(a.frames or n_gt)
    view_id = a.view_id if a.view_id is not None else int((getattr(phys, "test_cam_ids", None) or [0])[0])

    sim = est.simulator
    n_sub = int(sim.n_substeps[None])
    rows, seq = [], []
    for f in range(frames):
        if f > 0:
            sim.set_frame(f - 1)
            for i in range(n_sub * (f - 1), n_sub * f):
                if sim.cfl_satisfy[None]:
                    sim.substep(i, cache=False)
        pos = np.zeros((est.num_particles[None], 3), np.float32)
        sim.get_x(f, pos)
        if not sim.cfl_satisfy[None] or not np.isfinite(pos).all():
            print(f"[export] simulation failed at frame {f} (CFL/NaN); holding the last state")
            pos = seq[-1] if seq else pos
        seq.append(pos)
        write_xyz(model / "mpm" / f"simulation_{f}.ply", pos)
        row = dict(frame=f)
        for k in range(2):
            pk = pos[labels == k + 1]
            write_xyz(model / "mpm" / f"object{k}_{f}.ply", pk)
            gp = gt_dirs[k] / f"{f}.ply"
            if gp.exists() and len(pk):
                row[f"cd_obj{k}"] = chamfer_sq_mm2(pk.astype(np.float64), read_xyz(gp).astype(np.float64))
        if all(f"cd_obj{k}" in row for k in range(2)):
            gt_all = np.concatenate([read_xyz(gt_dirs[k] / f"{f}.ply") for k in range(2)], 0)
            row["cd"] = chamfer_sq_mm2(pos.astype(np.float64), gt_all.astype(np.float64))
        rows.append(row)
        if f % 8 == 0:
            print(f"[export] frame {f}: " + ", ".join(f"{k} {v:.3f}" for k, v in row.items() if k != "frame"), flush=True)
    print(f"[export] wrote {frames} frames of particles to {model / 'mpm'}")

    if not a.no_render:
        from gaussian_renderer import render
        from PIL import Image
        cam, H, W = build_camera(source, view_id)
        pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
        colors = [[1.0, 0.784, 0.157], [0.004, 0.267, 0.129]]
        g = particle_gaussians(seq[0], labels, a.radius, colors)
        bg = torch.tensor([1.0, 1.0, 1.0], device="cuda")
        (model / "img_render").mkdir(exist_ok=True)
        with torch.no_grad():
            for f, pos in enumerate(seq):
                d_xyz = torch.from_numpy(pos).cuda() - g.get_xyz
                res = render(cam, g, pipe, bg, d_xyz, 0.0, 0.0, False)
                rgb = (res["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                alpha = (res["alpha"][0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(rgb).save(model / "img_render" / f"{view_id}_{f:05d}.png")
                Image.fromarray(alpha).save(model / "img_render" / f"{view_id}_{f:05d}_mask.png")
                gt_a_path = source / "data" / f"a_{view_id}_{f}.png"
                if gt_a_path.exists():
                    gt_a = np.asarray(Image.open(gt_a_path).convert("RGBA"))[..., 3] > 127
                    pa = alpha > 127
                    rows[f]["silhouette_iou"] = float((gt_a & pa).sum() / max((gt_a | pa).sum(), 1))
        print(f"[export] rendered camera {view_id} to {model / 'img_render'}")

    mean = lambda k: float(np.mean([r[k] for r in rows if k in r])) if any(k in r for r in rows) else None
    metrics = dict(per_frame=rows, frames=frames, view_id=view_id, fitted=pred.get("sub_objects"),
                   mean=dict(cd=mean("cd"), cd_obj0=mean("cd_obj0"), cd_obj1=mean("cd_obj1"), silhouette_iou=mean("silhouette_iou")))
    json.dump(metrics, open(model / "prediction_metrics.json", "w"), indent=1)
    print("[export] mean:", metrics["mean"])


if __name__ == "__main__":
    main()
