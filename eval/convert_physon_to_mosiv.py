#!/usr/bin/env python
"""Convert a multi-object PhysON scene into MOSIV's GenesisMO input format and write its config.

MOSIV (Liu et al., ICLR 2026; vendored at MOSIV/) reads the "GenesisMO" layout:

  <out>/all_data.json                     -> symlink to the PhysON scene file (PAC-NeRF cameras)
  <out>/data/                             -> symlink to the PhysON data/ (m_<cam>_<frame>.png on white, r_*)
  <out>/masks/o_<cam>_<frame>.npy         uint8 [H,W,K] {0,255} instance masks, one channel per object
  <out>/point_clouds/<k>/<frame>.ply      per-object GT particles (evaluation), k = 0..K-1
  <out>/metadata.json                     GenesisMO-style scene description (bounds, obj1..objK, fps, ...)
  <config_out>                            MOSIV config json (data bounds, gs, physics.sub_objects, ...)

PhysON multi-object scenes ship a whole-scene foreground alpha (`a_*.png`) but no instance masks.
MOSIV's own benchmark provides simulator-rendered instance masks, so the equivalent oracle input is
generated here from the GT particles: every object's particles are splatted into each camera with a
z-buffer, pixels inside the foreground alpha take the label of the nearest covered pixel. The GT
particles are otherwise only used for the per-object bounding boxes (the PAC-NeRF/MOSIV scene-bbox
prior) and for evaluation. Material classes come from metadata (oracle class, as in MOSIV); the
initial parameter values are MOSIV's per-class defaults, never the GT values.

Both PhysON metadata schemas are understood: `objects[k].particle_range` (multiobject_heterogeneous)
and `objects[k]` + `region_offsets` (multiobject_heterogeneous_new). MOSIV's released code handles
exactly two objects; the vendored copy was generalised to K objects (ids 1..K, config order), so
any object count runs. Scenes without a force_field.npz run with force_mode none.

  python eval/convert_physon_to_mosiv.py --scene_data MOSIV/data/PhysON/multiobject_heterogeneous/0_0 \
      --out MOSIV/data/PhysON_mosiv/multiobject_heterogeneous/0_0 \
      --config_out MOSIV/config/physon/multiobject_heterogeneous/0_0.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from physon_common import load_scene, read_ply_xyz, write_ply_xyz  # noqa: E402

# PhysON material kind -> MOSIV/GIC material code and MOSIV's per-class initialisation
# (generate_configs.py in the MOSIV repo; values are actual scale, log10 is taken inside the estimator)
MATERIAL_CODE = {"elastic": 10, "plasticine": 12, "elastoplastic": 12, "sand": 13, "snow": 13,
                 "newtonian": 11, "fluid": 11, "liquid": 11, "non_newtonian": 14}
DEFAULT_PARAMS = {
    10: dict(init_E=1e5, init_nu=0.25, trainable=["Youngs modulus", "Poisson ratio"]),
    12: dict(init_E=1e4, init_nu=0.25, init_yield_stress=1e3, trainable=["Youngs modulus", "Poisson ratio", "Yield stress"]),
    13: dict(init_E=1e6, init_nu=0.3, init_friction_alpha=10.0, trainable=["friction angle"]),
    11: dict(mu=10.0, kappa=1e4, trainable=["kappa", "mu"]),
    14: dict(mu=10.0, kappa=1e4, init_plastic_viscosity=100.0, init_yield_stress=1e3,
             trainable=["kappa", "mu", "plastic viscosity", "Yield stress"]),
}
TRAINING_PARAMS = {
    "Youngs modulus": dict(lr_decay=True, init_lr=0.05, final_lr=0.01, max_steps=200),
    "Poisson ratio": dict(lr_decay=True, init_lr=0.01, final_lr=0.005, max_steps=200),
    "Yield stress": dict(lr_decay=True, init_lr=0.1, final_lr=0.01, max_steps=200),
    "friction angle": dict(lr_decay=False, init_lr=1.0, final_lr=0.1, max_steps=100),
    "plastic viscosity": dict(lr_decay=False, init_lr=0.01, final_lr=0.005, max_steps=100),
    "mu": dict(lr_decay=False, init_lr=0.15, final_lr=0.01, max_steps=100),
    "kappa": dict(lr_decay=False, init_lr=0.2, final_lr=0.1, max_steps=100),
}
MOSIV_MATERIAL_NAME = {10: "elastic", 12: "elastoplastic", 13: "sand", 11: "fluid", 14: "non_newtonian"}
# same list as MOSIV/utils/object_palette.py (kept in sync by hand; eval/ must not import MOSIV/)
OBJ_COLORS = [[1.0, 0.784, 0.157], [0.004, 0.267, 0.129], [0.122, 0.467, 0.706], [0.839, 0.153, 0.157],
              [0.580, 0.404, 0.741], [0.549, 0.337, 0.294], [0.890, 0.467, 0.761], [0.090, 0.745, 0.812]]


def obj_color(k):  # k = 0-based object index
    return OBJ_COLORS[k % len(OBJ_COLORS)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene_data", required=True, help="PhysON scene dir")
    p.add_argument("--out", required=True, help="GenesisMO-format output dir (MOSIV/data/PhysON_mosiv/<subset>/<scene>)")
    p.add_argument("--config_out", required=True, help="MOSIV config json to write")
    p.add_argument("--n_frames", type=int, default=None, help="observed frames used for fitting (default: all)")
    p.add_argument("--iter_cnt", type=int, default=80, help="physical-parameter iterations (upstream default 300 ~ 15 h/scene; 80 ~ 5 h, validated)")
    p.add_argument("--vel_iter_cnt", type=int, default=80)
    p.add_argument("--gs_iterations", type=int, default=40000)
    p.add_argument("--bc_style", type=int, default=2, help="ground collider: 0 sticky, 1 slip, 2 separate (PhysON floors separate)")
    p.add_argument("--force_mode", default="oracle", choices=["oracle", "none"],
                   help="oracle applies the scene's declared force field; scenes without one always run as none")
    p.add_argument("--traj_save_interval", type=int, default=10)
    p.add_argument("--mask_radius_px", type=int, default=6)
    p.add_argument("--bbox_margin", type=float, default=0.05, help="m around the per-object frame-0 bbox (init points)")
    p.add_argument("--max_particles", type=int, default=200000,
                   help="lifted-particle budget; the density grid / MPM voxel are coarsened for bigger objects")
    p.add_argument("--force", action="store_true", help="regenerate masks even if present")
    return p.parse_args()


def relative_symlink(link: Path, target: Path):
    rel = os.path.relpath(target.resolve(), link.parent.resolve())
    if link.is_symlink():
        if os.readlink(link) == rel:
            return
        link.unlink()
    elif link.exists():
        raise RuntimeError(f"{link} exists and is not a symlink")
    link.symlink_to(rel)


def scene_objects(meta):
    """[(name, kind, material dict, (start, end))] per object, in metadata order (both schemas)."""
    objs = meta.get("objects") or []
    offsets = meta.get("region_offsets")
    out = []
    for k, o in enumerate(objs):
        mat = o.get("material") or o.get("material_parameters") or {}
        if o.get("particle_range") is not None:
            s, e = int(o["particle_range"][0]), int(o["particle_range"][1])
        elif offsets is not None and len(offsets) == len(objs) + 1:
            s, e = int(offsets[k]), int(offsets[k + 1])
        else:
            raise SystemExit(f"cannot locate the particles of object {k}: no particle_range and no region_offsets")
        name = o.get("asset") or o.get("role") or o.get("name") or f"obj{k + 1}"
        out.append((str(name), str(mat.get("kind", "elastic")), mat, (s, e)))
    if not out:
        raise SystemExit("metadata.json lists no objects")
    return out


def instance_masks(scene, x_objs, cam_id, frame, radius_px):
    """[H,W,K] uint8 {0,255}: z-buffered splats of every object's GT particles, filled inside the
    foreground alpha (a_<cam>_<frame>.png) with the label of the nearest covered pixel."""
    from PIL import Image
    from scipy import ndimage
    c = scene.cameras[cam_id]
    H, W = c.height, c.width
    alpha = np.asarray(Image.open(scene.image_path(cam_id, frame, "a")).convert("RGBA"))[..., 3] > 127
    depth = np.full((len(x_objs), H * W), np.inf, dtype=np.float32)
    dy, dx = np.mgrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    disk = (dy ** 2 + dx ** 2) <= radius_px ** 2
    dy, dx = dy[disk], dx[disk]
    for k, x in enumerate(x_objs):
        if len(x) == 0:
            continue
        xc = x @ c.R + c.T                       # camera coords (3DGS: R stored transposed)
        z = xc[:, 2]
        ok = z > 1e-4
        u = np.round(c.K[0, 0] * xc[ok, 0] / z[ok] + c.K[0, 2]).astype(int)
        v = np.round(c.K[1, 1] * xc[ok, 1] / z[ok] + c.K[1, 2]).astype(int)
        zz = z[ok]
        uu = (u[:, None] + dx[None]).ravel(); vv = (v[:, None] + dy[None]).ravel()
        zk = np.repeat(zz, len(dx))
        inb = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        np.minimum.at(depth[k], vv[inb] * W + uu[inb], zk[inb])
    depth = depth.reshape(len(x_objs), H, W)
    covered = np.isfinite(depth).any(0)
    label = depth.argmin(0)                      # nearest object per covered pixel
    fg = alpha | covered                          # covered pixels count as foreground too
    if covered.any():
        _, (iy, ix) = ndimage.distance_transform_edt(~covered, return_indices=True)
        label = label[iy, ix]
    out = np.zeros((H, W, len(x_objs)), dtype=np.uint8)
    for k in range(len(x_objs)):
        out[..., k] = ((label == k) & fg) * 255
    return out


def main():
    a = parse_args()
    src = Path(a.scene_data).resolve()
    scene = load_scene(src)
    meta = scene.metadata
    objs = scene_objects(meta)
    K = len(objs)
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    relative_symlink(out / "all_data.json", src / "all_data.json")
    relative_symlink(out / "data", src / "data")
    print(f"[scene] {src.parent.name}/{src.name}: {K} objects " +
          ", ".join(f"{n} ({kind}, {e - s} particles)" for n, kind, _, (s, e) in objs) +
          f"; {scene.n_frames} frames @ {scene.sim_fps:g} fps, test cams {scene.test_cam_ids}, floor {scene.floor_height}")

    # ---- per-object GT particles (evaluation) and bounds
    n_frames_total = scene.n_frames
    xs = [read_ply_xyz(scene.gt_ply_path(f)) for f in range(n_frames_total)]
    per_obj = [[x[s:e] for x in xs] for (_, _, _, (s, e)) in objs]
    for k in range(K):
        d = out / "point_clouds" / str(k); d.mkdir(parents=True, exist_ok=True)
        for f in range(n_frames_total):
            if not (d / f"{f}.ply").exists() or a.force:
                write_ply_xyz(d / f"{f}.ply", per_obj[k][f])
    obj_bbox = [(per_obj[k][0].min(0) - a.bbox_margin, per_obj[k][0].max(0) + a.bbox_margin) for k in range(K)]
    for k in range(K):
        obj_bbox[k][0][1] = max(obj_bbox[k][0][1], scene.floor_height)
    all_x = np.concatenate([np.concatenate(o, 0) for o in per_obj], 0)
    lo, hi = all_x.min(0) - 0.15, all_x.max(0) + 0.15
    lo[1] = min(lo[1], scene.floor_height - 0.05)

    # ---- instance masks from the GT particles (oracle segmentation, cf. MOSIV's simulator masks)
    mdir = out / "masks"; mdir.mkdir(exist_ok=True)
    todo = [(c, f) for c in sorted(scene.cameras) for f in range(n_frames_total)
            if a.force or not (mdir / f"o_{c}_{f}.npy").exists()]
    if todo:
        print(f"[masks] generating {len(todo)} instance masks (radius {a.mask_radius_px} px, {K} channels)")
        cov = []
        for i, (c, f) in enumerate(todo):
            m = instance_masks(scene, [per_obj[k][f] for k in range(K)], c, f, a.mask_radius_px)
            np.save(mdir / f"o_{c}_{f}.npy", m)
            if f == 0:
                from PIL import Image
                alpha = np.asarray(Image.open(scene.image_path(c, f, "a")).convert("RGBA"))[..., 3] > 127
                cov.append(((m.max(-1) > 0) & alpha).sum() / max(alpha.sum(), 1))
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
        print(f"[masks] frame-0 foreground coverage per camera: {np.round(cov, 3).tolist()}")
    (out / "masks_vis").mkdir(exist_ok=True)
    try:
        from PIL import Image
        for c in scene.test_cam_ids + scene.train_cam_ids[:1]:
            m = np.load(mdir / f"o_{c}_0.npy")
            vis = np.zeros((*m.shape[:2], 3), np.uint8)
            for k in range(m.shape[2]):
                col = np.asarray(obj_color(k)) * 255
                vis[m[..., k] > 0] = col.astype(np.uint8)
            Image.fromarray(vis).save(out / "masks_vis" / f"vis_{c}_0.png")
    except Exception as e:  # visualisation only
        print(f"[masks] vis skipped: {e}")

    # ---- metadata.json (GenesisMO style; PhysON fields kept under 'physon')
    fps = float(scene.sim_fps)
    mpm_iter = 200
    md = dict(
        source="PhysON " + str(src.parent.name) + "/" + src.name, format="GenesisMO (MOSIV) export of a PhysON scene",
        n_objects=K, gravity=float(scene.gravity[1]), dt=1.0 / (fps * mpm_iter), fps=fps, video_fps=meta.get("video_fps"),
        mpm_lower_bound=lo.tolist(), mpm_upper_bound=hi.tolist(), particle_size=float(scene.particle_size),
        resolution=meta.get("resolution", [800, 800]), ground_friction=0.0, floor_height=float(scene.floor_height),
        collide_time=0.0, collide_loc=[0.0, 0.0, 0.0],
        colors=[obj_color(k) for k in range(K)], camera_ids=sorted(scene.cameras), test_cam_ids=scene.test_cam_ids,
        physon=dict(objects=meta.get("objects"), region_offsets=meta.get("region_offsets"), regions=scene.regions,
                    external_force=meta.get("external_force")),
    )
    for k, (name, kind, mat, (s, e)) in enumerate(objs):
        code = MATERIAL_CODE.get(kind)
        if code is None:
            raise SystemExit(f"unsupported PhysON material kind {kind!r} for MOSIV")
        md[f"obj{k + 1}"] = dict(material=MOSIV_MATERIAL_NAME[code], material_kind_physon=kind, geometry=name,
                                 rho=float(mat.get("rho", 1000.0)), surface_color=obj_color(k),
                                 gt_material_parameters=mat, n_particles=int(e - s), particle_range=[s, e])
    json.dump(md, open(out / "metadata.json", "w"), indent=1)

    # ---- MOSIV config json (mirrors MOSIV/generate_configs.py; timing from the PhysON physical clock)
    sub_objects, trainable = [], []
    for k, (name, kind, mat, _) in enumerate(objs):
        code = MATERIAL_CODE[kind]
        dp = dict(DEFAULT_PARAMS[code]); trainable += dp.pop("trainable")
        sub_objects.append(dict(name=name, object_id=k + 1, material=code, **dp,
                                rho=float(mat.get("rho", 1000.0)), init_vel=[0.0, 0.0, 0.0], color=obj_color(k)))
    codes = [s["material"] for s in sub_objects]
    if 13 in codes:
        voxel, dgs, dmin, dmax = 0.015, 0.12, 0.67, 0.8
    elif 11 in codes or 14 in codes:
        voxel, dgs, dmin, dmax = 0.02, 0.12, 0.67, 0.9
    elif 12 in codes:
        voxel, dgs, dmin, dmax = 0.02, 0.1, 0.7, 0.9
    else:
        voxel, dgs, dmin, dmax = 0.02, 0.1, 0.5, 0.7
    # particle budget: MOSIV lifts one particle per (density_grid_size/16)^3 of object volume; a big
    # object (e.g. the cushion of multiobject_heterogeneous/0_0: 0.1 m^3 -> 400k particles) would
    # need far more taichi memory/time than the ~150k of MOSIV's own scenes, so both grid sizes are
    # scaled up together until the estimate fits --max_particles
    volume = float(scene.particle_size) ** 3 * sum(e - s for (_, _, _, (s, e)) in objs)
    n_est = volume / (dgs / 16.0) ** 3
    if n_est > a.max_particles:
        f = (n_est / a.max_particles) ** (1.0 / 3.0)
        print(f"[budget] ~{n_est:.0f} lifted particles at density_grid_size {dgs} > {a.max_particles}: "
              f"scaling density_grid_size/voxel_size by {f:.2f}")
        dgs, voxel = dgs * f, voxel * f
    n_frames = int(a.n_frames or n_frames_total)
    force_mode = a.force_mode
    if force_mode == "oracle" and not (src / "force_field.npz").exists():
        print("[force] scene has no force_field.npz (gravity only): force_mode none")
        force_mode = "none"
    # paths in the config are relative to the MOSIV dir (train_dynamic_MO.py / export_prediction.py run there)
    mosiv_root = Path(__file__).resolve().parent.parent / "MOSIV"
    relm = lambda p: os.path.relpath(Path(p).resolve(), mosiv_root)
    data = dict(xyz_min=lo.tolist(), xyz_max=hi.tolist())
    for k in range(K):
        data[f"obj{k + 1}_xyz_min"] = obj_bbox[k][0].tolist()
        data[f"obj{k + 1}_xyz_max"] = obj_bbox[k][1].tolist()
    cfg = dict(
        data=data,
        gs=dict(eval=True, is_blender=True, timenet=True, test_iterations=[5000, 6000, 7000],
                save_iterations=sorted({i for i in (7000, 10000, 20000, 30000, a.gs_iterations) if i <= a.gs_iterations}), quiet=False,
                iterations=a.gs_iterations, enable_mask_training=True, mask_loss_weight=0.5),
        physics=dict(
            id=src.name, fps=fps, dt=1.0 / (fps * mpm_iter), gravity=scene.gravity.tolist(), ground_friction=0.0,
            n_objects=K, sub_objects=sub_objects, voxel_size=voxel, mpm_iter_cnt=mpm_iter,
            bc=dict(ground=[[0.0, float(scene.floor_height), 0.0], [0.0, 1.0, 0.0], int(a.bc_style)]),
            density_grid_size=dgs, density_min_th=dmin, density_max_th=dmax, opacity_threshold=0.01, random_sample=False,
            img_loss=True, geo_loss=True, w_img=0.0, w_alp=1.0, w_geo=1.0,
            params={p: TRAINING_PARAMS[p] for p in sorted(set(trainable))},
            iter_cnt=int(a.iter_cnt), vel_iter_cnt=int(a.vel_iter_cnt), vel_estimation_frames=4, vel_lr=0.05,
            init_vel=[0.0, 0.0, 0.0], collide_time=0.0, collide_loc=[0.0, 0.0, 0.0], n_frames=n_frames,
            traj_save_interval=int(a.traj_save_interval),
            force_mode=force_mode, force_npz=relm(src / "force_field.npz"), force_json=relm(src / "force_field.json"),
            force_h5=relm(src / "physics.h5"),
            physon_scene=relm(src), test_cam_ids=scene.test_cam_ids,
        ),
    )
    Path(a.config_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(a.config_out, "w"), indent=2)
    print(f"[done] data {out}\n[done] config {a.config_out}: objects {[(s['name'], s['material']) for s in sub_objects]}, "
          f"fps {fps}, n_frames {n_frames}, iter_cnt {a.iter_cnt}, force {force_mode}, bounds {np.round(lo, 3).tolist()}..{np.round(hi, 3).tolist()}")


if __name__ == "__main__":
    main()
