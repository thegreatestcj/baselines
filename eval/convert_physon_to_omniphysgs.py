#!/usr/bin/env python
"""Convert a PhysON scene into an OmniPhysGS "package".

Writes, under the vendored OmniPhysGS repo (all of it gitignored):

  data/PhysON/<subset>/<scene>/
      source -> <scene_data>        relative symlink; images / GT / force files are read through it
      gs_dataset/                   static frame-0 3DGS dataset for third_party/gaussian-splatting
          transforms_train.json     train cameras, frame 0 (file_path ./data/a_<cam>_0, Blender c2w)
          transforms_test.json      the held-out camera(s)
          data -> ../source/data
          points3d.ply              ~100k random points inside the frame-0 visual hull carved from the
                                    train cameras' alpha masks (--no_hull: uniform in the bbox prior)
      scene.json                    cameras (3DGS R/T/K/fov), timing, gravity, floor, bbox, force paths
      config.yaml                   from --template with the scene values filled in

When the package dir *is* the scene dir (the default `data/PhysON/<subset>/<scene>` layout, where
data/PhysON is the symlink to the data store) the `source` link simply points to `.`.

Run with the baselines interpreter from the OmniPhysGS dir:
  "$BASELINES_PY" ../eval/convert_physon_to_omniphysgs.py \
      --scene_data data/PhysON/singleobject_heterogeneous_new/0_0 \
      --out_name singleobject_heterogeneous_new/0_0
Then:
  "$OMNIPHYSGS_PY" recon_static.py --package data/PhysON/singleobject_heterogeneous_new/0_0
  "$OMNIPHYSGS_PY" fit.py --config data/PhysON/singleobject_heterogeneous_new/0_0/config.yaml --tag ...
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from physon_common import blender_c2w_from_RT, load_scene  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene_data", required=True, help="PhysON scene dir (all_data.json, data/, ...)")
    p.add_argument("--out_name", required=True, help="<subset>/<scene>; package = <omniphysgs_root>/data/PhysON/<out_name>")
    p.add_argument("--omniphysgs_root", default=str(REPO_ROOT / "OmniPhysGS"))
    p.add_argument("--template", default=str(Path(__file__).resolve().parent / "omniphysgs_physon_template.yaml"))
    p.add_argument("--n_init_points", type=int, default=100_000)
    p.add_argument("--init_margin", type=float, default=0.05, help="m around the frame-0 bbox for the init points")
    p.add_argument("--no_hull", action="store_true",
                   help="init points uniformly in the bbox prior instead of inside the frame-0 visual hull")
    p.add_argument("--train_frames", type=int, default=None, help="limit the simulated/supervised frames")
    p.add_argument("--keep_black_views", action="store_true",
                   help="do not exclude black-silhouette views (object luminance ~0) from 3DGS/RGB supervision")
    p.add_argument("--force", action="store_true", help="rewrite files that already exist")
    return p.parse_args()


def relative_symlink(link: Path, target: Path, force: bool):
    """`link -> target` with a path relative to the link's directory ('.' when they coincide)."""
    rel = os.path.relpath(target.resolve(), link.parent.resolve())
    if link.is_symlink() or link.exists():
        if link.is_symlink() and os.readlink(link) == rel and not force:
            return
        if link.is_symlink():
            link.unlink()
        else:
            raise RuntimeError(f"{link} exists and is not a symlink")
    link.symlink_to(rel)


def visual_hull_points(scene, cam_ids, lo, hi, n, voxel=0.01, alpha_thr=0.3, seed=0):
    """Sample n points inside the frame-0 visual hull carved from the alpha masks of `cam_ids`
    (world metres; box lo..hi voxelised at `voxel`). Only observed masks are used, so this is a
    legitimate initialisation for the static 3DGS. Returns None when the hull is empty."""
    from PIL import Image
    axes = [np.arange(lo[k], hi[k], voxel) + voxel / 2 for k in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    inside = np.ones(len(pts), dtype=bool)
    for cid in cam_ids:
        c = scene.cameras[cid]
        im = Image.open(scene.image_path(cid, 0))
        a = np.asarray(im.convert("RGBA"))[..., 3].astype(np.float32) / 255.0
        if im.mode != "RGBA":  # m_/r_ style files: white-background rule
            a = (np.asarray(im.convert("RGB")).astype(np.int32).sum(-1) != 255 * 3).astype(np.float32)
        mask = a > alpha_thr
        xc = pts @ c.R + c.T                      # R is stored transposed (3DGS): w2c rotation = R^T
        z = np.clip(xc[:, 2], 1e-9, None)
        u = np.round(c.K[0, 0] * xc[:, 0] / z + c.K[0, 2]).astype(int)
        v = np.round(c.K[1, 1] * xc[:, 1] / z + c.K[1, 2]).astype(int)
        ok = (xc[:, 2] > 0) & (u >= 0) & (u < mask.shape[1]) & (v >= 0) & (v < mask.shape[0])
        hit = np.zeros(len(pts), dtype=bool)
        hit[ok] = mask[v[ok], u[ok]]
        inside &= hit
    if inside.sum() < 10:
        return None
    hull = pts[inside]
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(hull), size=n)
    xyz = hull[pick] + rng.uniform(-voxel / 2, voxel / 2, size=(n, 3))
    print(f"[hull] {inside.sum()} of {len(pts)} voxels ({voxel} m) inside all {len(cam_ids)} train masks; "
          f"hull bbox {hull.min(0).round(3).tolist()}..{hull.max(0).round(3).tolist()}")
    return xyz.astype(np.float32)


def black_views(scene, cam_ids, frame=0, lum_thr=0.05, alpha_thr=0.5):
    """Cameras whose frame-`frame` object pixels (alpha > alpha_thr) are essentially black while the
    object is bright in the other views (e.g. PhysON 0_x cam 4 looks straight down at a black disc).
    Such a view carries no appearance information and would poison the 3DGS colours; its alpha
    mask is still valid (hull, alpha loss). Returns (excluded ids, {cam: luminance})."""
    from PIL import Image
    lum = {}
    for cid in cam_ids:
        a = np.asarray(Image.open(scene.image_path(cid, frame)).convert("RGBA")).astype(np.float32) / 255.0
        m = a[..., 3] > alpha_thr
        lum[cid] = float(a[m][:, :3].mean()) if m.any() else 0.0
    med = float(np.median(list(lum.values())))
    bad = [c for c in cam_ids if lum[c] < lum_thr and med > 4 * lum_thr]
    return bad, lum


def write_points3d(path: Path, lo, hi, n, seed=0, xyz=None):
    from plyfile import PlyData, PlyElement
    rng = np.random.default_rng(seed)
    if xyz is None:
        xyz = rng.uniform(lo, hi, size=(n, 3)).astype(np.float32)
    n = len(xyz)
    rgb = rng.uniform(0.3, 0.7, size=(n, 3))  # neutral grey init colours
    arr = np.empty(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                             ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["nx"] = arr["ny"] = arr["nz"] = 0.0
    arr["red"], arr["green"], arr["blue"] = (rgb * 255).astype(np.uint8).T
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def fmt_list(v):
    return "[" + ", ".join(f"{float(x):.6g}" for x in v) + "]"


def main():
    args = parse_args()
    scene_dir = Path(args.scene_data).resolve()
    scene = load_scene(scene_dir)
    root = Path(args.omniphysgs_root).resolve()
    pkg = root / "data" / "PhysON" / args.out_name
    pkg.mkdir(parents=True, exist_ok=True)
    same_dir = pkg.resolve() == scene_dir
    print(f"[scene] {scene_dir}: {len(scene.cameras)} cams (test {scene.test_cam_ids}), {scene.n_frames} frames @ "
          f"{scene.sim_fps:g} fps, gravity {scene.gravity.tolist()}, floor {scene.floor_height}")
    print(f"[package] {pkg}" + (" (package dir == scene dir)" if same_dir else ""))

    # ---- source link
    relative_symlink(pkg / "source", scene_dir, args.force)

    # ---- static 3DGS dataset (frame 0)
    cams = scene.cameras
    fx = np.array([c.K[0, 0] for c in cams.values()])
    fy = np.array([c.K[1, 1] for c in cams.values()])
    assert np.ptp(fx) < 1e-3 * fx.mean() and np.ptp(fy) < 1e-3 * fy.mean(), \
        "per-camera intrinsics differ; the vendored Blender loader reads a single camera_angle_x"
    assert all(c.width == c.height for c in cams.values()) or np.allclose(fx, fy, rtol=1e-3), \
        "non-square pixels/images: fovy would be derived from fovx by the Blender loader"
    gs = pkg / "gs_dataset"
    gs.mkdir(exist_ok=True)
    relative_symlink(gs / "data", pkg / "source" / "data", args.force)
    rgb_exclude, lum = ([], {}) if args.keep_black_views else black_views(scene, scene.train_cam_ids)
    if rgb_exclude:
        print(f"[views] black-silhouette train cams {rgb_exclude} (object luminance "
              f"{ {c: round(l, 3) for c, l in lum.items()} }): excluded from 3DGS training and from the RGB loss; "
              "their alpha masks are still used")
    gs_train_cams = [c for c in scene.train_cam_ids if c not in rgb_exclude]

    def transforms(cam_ids):
        frames = []
        for cid in cam_ids:
            c = cams[cid]
            frames.append(dict(file_path=f"./data/a_{cid}_0", rotation=0.0,
                               transform_matrix=blender_c2w_from_RT(c.R, c.T).tolist()))
        return dict(camera_angle_x=float(next(iter(cams.values())).fovx), frames=frames)

    json.dump(transforms(gs_train_cams), open(gs / "transforms_train.json", "w"), indent=1)
    json.dump(transforms(scene.test_cam_ids), open(gs / "transforms_test.json", "w"), indent=1)
    lo = scene.bbox_min - args.init_margin
    hi = scene.bbox_max + args.init_margin
    lo[1] = max(lo[1], scene.floor_height)
    ply = gs / "points3d.ply"
    if args.force or not ply.exists():
        xyz = None
        if not args.no_hull:
            xyz = visual_hull_points(scene, scene.train_cam_ids, lo, hi, args.n_init_points)
            if xyz is None:
                print("[warn] empty visual hull (bad masks?); falling back to random points in the bbox prior")
        write_points3d(ply, lo, hi, args.n_init_points, xyz=xyz)
    print(f"[gs_dataset] {len(gs_train_cams)} train / {len(scene.test_cam_ids)} test cams, "
          f"init points in {lo.round(3).tolist()}..{hi.round(3).tolist()}")

    # ---- force files (paths relative to the package, through source/)
    force_npz = "source/force_field.npz" if (scene_dir / "force_field.npz").exists() else ""
    force_json = "source/force_field.json" if (scene_dir / "force_field.json").exists() else ""
    physics_h5 = "source/physics.h5" if (scene_dir / "physics.h5").exists() else ""
    force_bbox_min = force_bbox_max = None
    if force_npz:
        z = np.load(scene_dir / "force_field.npz")
        if "bbox_min" in z.files and "bbox_max" in z.files:
            force_bbox_min, force_bbox_max = z["bbox_min"].tolist(), z["bbox_max"].tolist()
        print(f"[force] {force_npz} fields {z.files}; lattice bbox {force_bbox_min} .. {force_bbox_max}")
    else:
        print("[force] no force_field.npz (force_mode=oracle will apply zero external acceleration)")

    n_frames = scene.n_frames if args.train_frames is None else min(args.train_frames, scene.n_frames)

    # ---- scene.json
    scene_json = dict(
        name=args.out_name, scene_dir=os.path.relpath(scene_dir, pkg) if not same_dir else ".",
        cameras=[cams[c].to_json() for c in sorted(cams)],
        train_cam_ids=scene.train_cam_ids, test_cam_ids=scene.test_cam_ids,
        rgb_exclude_cams=rgb_exclude, object_luminance={str(c): l for c, l in lum.items()},
        n_frames=n_frames, n_frames_total=scene.n_frames, sim_fps=scene.sim_fps, frame_dt=scene.frame_dt,
        frame_times=[scene.frame_times[f] for f in scene.frame_ids],
        gravity=scene.gravity.tolist(), floor_height=scene.floor_height, particle_size=scene.particle_size,
        bbox_min=scene.bbox_min.tolist(), bbox_max=scene.bbox_max.tolist(),
        force_npz=force_npz, force_json=force_json, physics_h5=physics_h5,
        force_bbox_min=force_bbox_min, force_bbox_max=force_bbox_max,
        has_point_clouds=(scene_dir / "point_clouds").is_dir(),
        regions=scene.regions,  # GT annotations, for reporting only
        rigid_bodies=(scene_dir / "rigid_bodies.json").exists(),
    )
    json.dump(scene_json, open(pkg / "scene.json", "w"), indent=1)

    # ---- config.yaml from the template
    text = open(args.template).read()
    fill = {
        "scene_name": args.out_name,
        "bbox_min": fmt_list(scene.bbox_min), "bbox_max": fmt_list(scene.bbox_max),
        "force_bbox_min": fmt_list(force_bbox_min) if force_bbox_min else "null",
        "force_bbox_max": fmt_list(force_bbox_max) if force_bbox_max else "null",
        "frame_dt": f"{scene.frame_dt:.10g}", "sim_fps": f"{scene.sim_fps:g}", "n_frames": str(n_frames),
        "gravity": fmt_list(scene.gravity), "floor_height": f"{scene.floor_height:g}",
        "force_npz": force_npz, "force_json": force_json, "physics_h5": physics_h5,
        "test_cam_ids": "[" + ", ".join(str(c) for c in scene.test_cam_ids) + "]",
        "rgb_exclude_cams": "[" + ", ".join(str(c) for c in rgb_exclude) + "]",
    }
    for k, v in fill.items():
        text = text.replace("{{" + k + "}}", v)
    assert "{{" not in text, "unfilled template token: " + text[text.index("{{"):][:40]
    header = ("# generated by eval/convert_physon_to_omniphysgs.py — edit eval/omniphysgs_physon_template.yaml instead\n"
              f"# scene {args.out_name}; GT material regions (not consumed): {json.dumps(scene.regions)}\n")
    open(pkg / "config.yaml", "w").write(header + text)
    if scene_json["rigid_bodies"]:
        print("[warn] scene has rigid bodies (rigid_bodies.json); OmniPhysGS has no rigid coupling — run as-is (best effort)")
    print(f"[done] {pkg / 'config.yaml'}")


if __name__ == "__main__":
    main()
