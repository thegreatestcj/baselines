#!/usr/bin/env python
"""Convert a Spring-Gaus mpm_synthetic scene into a Vid2Sim GSO-style case.

Produces, under the vendored Vid2Sim repo:
  dataset/SpringGaus/<name>/data/    a_<v>_<f>.png (the original RGBA frame),
                                     m_<v>_<f>.png and r_<v>_<f>.png (object
                                     composited on white; SG frames have no
                                     background), r_<v>_-1.png (plain white),
                                     512x512, re-indexed to the new view order
  dataset/SpringGaus/<name>/         transforms_{train,test,val,simulation}.json,
                                     points3d.ply, gt_phys_params.yaml,
                                     sim_transform.json
  config/springgaus/<name>.yaml      generated from config/gso.yaml

Frame convention: SG scenes have 30 frames x 10 views; SG observes the first
20 (N_FRAME) and holds out the last 10. Mapped to Vid2Sim's step convention
that is: the stock pipeline still trains on steps 0-15 (its hardcoded window,
a subset of the observed 20), and the generated config sets predict_steps: 29
so eval/vid2sim_future.py rolls out the full sequence -- score it with
--split 20 --total_frames 30 (replay = observed 0..19, future = 20..29). No
frame padding is needed (30 >= the 24 GT frames LBSSimulator hardcodes).

Views 0-3 are the 4 SG cams closest to a front/right/back/left orbit for LGM,
and the world is mapped into Vid2Sim/LGM's canonical frame exactly like the
PAC-NeRF converter (see eval/vid2sim_adapter_common.py): frame-0 visual hull
-> x_sim = s*R@(x-c), delta_t = DT*sqrt(s), GT E compares as s*E.

Run with the Vid2Sim interpreter, e.g. from the Vid2Sim repo dir:
  "$VID2SIM_PY" ../eval/convert_springgaus_to_vid2sim.py \
      --scene_data ../Spring-Gaus/data/mpm_synthetic/render/torus \
      --sg_cfg ../Spring-Gaus/config/mpm_synthetic/torus.yaml \
      --out_name sg_torus

Then:
  "$VID2SIM_PY" run_pipeline.py --config config/springgaus/sg_torus.yaml \
      --dataset_dir dataset/SpringGaus --data_name sg_torus
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vid2sim_adapter_common import (TARGET_HALF, build_world_rotation,
                                    pick_canonical_views, transform_c2w,
                                    visual_hull, write_config,
                                    write_gt_phys_params, write_points3d,
                                    write_sim_transform, write_transforms)

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene_data", required=True,
                   help="SG scene render dir (camera.json + camera_XXX/FFF.png)")
    p.add_argument("--sg_cfg", required=True, help="SG scene yaml (BC, DT, XYZ bounds)")
    p.add_argument("--out_name", required=True, help="case name under dataset/SpringGaus and outputs/")
    p.add_argument("--vid2sim_root", default=str(REPO_ROOT / "Vid2Sim"))
    p.add_argument("--gt_yms", type=float, default=None, help="GT Young's modulus, if known")
    p.add_argument("--gt_prs", type=float, default=None, help="GT Poisson's ratio, if known")
    return p.parse_args()


def main():
    args = parse_args()
    scene = Path(args.scene_data)
    vid2sim = Path(args.vid2sim_root)
    cfg = yaml.safe_load(open(args.sg_cfg))["DATA"]
    n_frames, n_obs = int(cfg["FRAME_ALL"]), int(cfg["N_FRAME"])
    frame_dt = float(cfg["DT"])

    cam_list = json.load(open(scene / "camera.json"))
    cams = {i: (np.asarray(e["c2w"], dtype=np.float64), np.asarray(e["K"], dtype=np.float64))
            for i, e in enumerate(cam_list)}
    cam_names = {i: e["camera"] for i, e in enumerate(cam_list)}
    print(f"[scene] {scene}: {len(cams)} cams, {n_frames} frames "
          f"({n_obs} observed / {n_frames - n_obs} future), frame_dt={frame_dt}")

    W = int(cfg["W"])
    fxs = [K[0, 0] for _, K in cams.values()]
    assert max(fxs) - min(fxs) < 1e-3 * np.mean(fxs), "per-camera focal lengths differ"
    camera_angle_x = 2 * math.atan(W / (2 * float(np.mean(fxs))))

    def frame_path(cam_id, f):
        return scene / cam_names[cam_id] / f"{f:03d}.png"

    def mask_fn(cam_id):
        return np.asarray(Image.open(frame_path(cam_id, 0)))[..., 3] > 0

    center, half, _ = visual_hull(cams, mask_fn, cfg["XYZ_MIN"], cfg["XYZ_MAX"])
    s = TARGET_HALF / half

    cam_pos = {i: c2w[:3, 3] for i, (c2w, _) in cams.items()}
    order = pick_canonical_views(cam_pos, center)
    order += [i for i in sorted(cams) if i not in order]
    R = build_world_rotation(cam_pos[order[0]], center)

    floor_y = float(cfg["BC"][0][0][1])  # BC: [[point, normal], ...], ground plane
    floor_level = float(s * (floor_y - center[1]))
    delta_t = frame_dt * math.sqrt(s)
    print(f"[world] scale {s:.4f}, floor_level {floor_level:.4f} (world y={floor_y}), "
          f"delta_t {delta_t:.5f}")

    # ---- case dir
    case_dir = vid2sim / "dataset" / "SpringGaus" / args.out_name
    data_dir = case_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    white = Image.new("RGB", (W, W), (255, 255, 255))
    for v, o in enumerate(order):
        white.save(data_dir / f"r_{v}_-1.png")
        for f in range(n_frames):
            rgba = Image.open(frame_path(o, f)).convert("RGBA")
            rgba.save(data_dir / f"a_{v}_{f}.png")
            comp = Image.new("RGB", rgba.size, (255, 255, 255))
            comp.paste(rgba, mask=rgba.getchannel("A"))
            comp.save(data_dir / f"m_{v}_{f}.png")
            comp.save(data_dir / f"r_{v}_{f}.png")
    print(f"[images] {len(order)} views x {n_frames} frames (white-composited)")

    view_c2ws = [transform_c2w(cams[o][0], R, s, center) for o in order]
    write_transforms(case_dir, camera_angle_x, view_c2ws)
    write_points3d(case_dir)
    write_sim_transform(case_dir, s, center, R)
    write_gt_phys_params(
        case_dir, args.gt_yms, args.gt_prs, s,
        "Spring-Gaus mpm_synthetic ships GT particle trajectories\n"
        "(data/mpm_synthetic/simulation/<scene>) but no GT E/nu; -1 =\n"
        "placeholder. Compare recovered best_params.yaml yms against\n"
        "yms_sim = yms * world_scale when GT is filled in.")

    write_config(vid2sim / "config" / "gso.yaml",
                 vid2sim / "config" / "springgaus" / f"{args.out_name}.yaml",
                 view_samples=len(order), floor_level=round(floor_level, 6),
                 floor_axis=2, delta_t=round(delta_t, 6),
                 predict_steps=n_frames - 1)
    print(f"[done] case {args.out_name}: dataset {case_dir}")


if __name__ == "__main__":
    main()
