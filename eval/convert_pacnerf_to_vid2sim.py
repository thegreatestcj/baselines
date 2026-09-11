#!/usr/bin/env python
"""Convert a PAC-NeRF benchmark scene into a Vid2Sim GSO-style case.

Produces, under the vendored Vid2Sim repo:
  dataset/PACNeRF/<name>/data/       a_<v>_<f>.png (RGBA), m_<v>_<f>.png (object
                                     on white), r_<v>_<f>.png (full render),
                                     r_<v>_-1.png (background) -- straight
                                     copies of the PAC-NeRF images (400x400 is
                                     fine: every Vid2Sim stage takes the size
                                     from the images and resizes internally),
                                     re-indexed to the new view order and
                                     padded to --pad_frames by repeating the
                                     last real frame (PAC-NeRF has 14 frames;
                                     LBSSimulator hardcodes 24 GT frames and
                                     the predictor wants 16 from view 0)
  dataset/PACNeRF/<name>/            transforms_{train,test,val,simulation}.json,
                                     points3d.ply, gt_phys_params.yaml,
                                     sim_transform.json
  config/pacnerf/<name>.yaml         generated from config/gso.yaml

Views 0-3 are the 4 PAC-NeRF cams closest to a front/right/back/left orbit
(LGM's canonical input); the world is mapped into Vid2Sim/LGM's canonical
frame (see eval/vid2sim_adapter_common.py): x_sim = s*R@(x - c) with c/extent
from the frame-0 visual hull, so the LGM initialization that Stage-II 3DGS
refinement starts from lands on the object. delta_t = frame_dt*sqrt(s) keeps
gravity at the hardcoded 9.8; GT Young's modulus then compares as s*E.

Run with the Vid2Sim interpreter, e.g. from the Vid2Sim repo dir:
  "$VID2SIM_PY" ../eval/convert_pacnerf_to_vid2sim.py \
      --scene_data ../GIC/data/pacnerf/elastic/0 \
      --gic_cfg ../GIC/config/pacnerf/elastic/default.json \
      --out_name elastic_0

Then:
  "$VID2SIM_PY" run_pipeline.py --config config/pacnerf/elastic_0.yaml \
      --dataset_dir dataset/PACNeRF --data_name elastic_0
"""
import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
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
    p.add_argument("--scene_data", required=True, help="PAC-NeRF scene dir (all_data.json + data/)")
    p.add_argument("--gic_cfg", required=True, help="GIC scene config json (bounds, floor, fps)")
    p.add_argument("--out_name", required=True, help="case name under dataset/PACNeRF and outputs/")
    p.add_argument("--vid2sim_root", default=str(REPO_ROOT / "Vid2Sim"))
    p.add_argument("--pad_frames", type=int, default=24,
                   help="pad each view to this many frames by repeating the last one")
    p.add_argument("--predict_steps", type=int, default=15,
                   help="written into the generated config (PAC-NeRF GT particle plys cover 0..15)")
    p.add_argument("--gt_yms", type=float, default=None, help="GT Young's modulus, if known")
    p.add_argument("--gt_prs", type=float, default=None, help="GT Poisson's ratio, if known")
    return p.parse_args()


def load_all_data(scene_dir: Path):
    entries = json.load(open(scene_dir / "all_data.json"))
    cams, frames = {}, set()
    for e in entries:
        stem = Path(e["file_path"]).stem  # r_<cam>_<frame>
        _, cam, frame = stem.split("_")
        cam, frame = int(cam), int(frame)
        c2w = np.asarray(e["c2w"], dtype=np.float64)
        if c2w.shape == (3, 4):
            c2w = np.vstack([c2w, [0, 0, 0, 1]])
        if cam not in cams:
            cams[cam] = (c2w, np.asarray(e["intrinsic"], dtype=np.float64))
        else:
            assert np.allclose(cams[cam][0], c2w), f"c2w varies over time for cam {cam}"
        if frame >= 0:
            frames.add(frame)
    return cams, sorted(frames), sorted({e["time"] for e in entries if e["time"] >= 0})


def main():
    args = parse_args()
    scene = Path(args.scene_data)
    vid2sim = Path(args.vid2sim_root)
    gic_cfg = json.load(open(args.gic_cfg))

    cams, frames, times = load_all_data(scene)
    n_frames = len(frames)
    frame_dt = float(np.mean(np.diff(times))) if len(times) > 1 else 1 / 24
    print(f"[scene] {scene}: {len(cams)} cams, {n_frames} frames, frame_dt={frame_dt:.5f}")

    # single shared fov (Blender-style json has one camera_angle_x)
    W = Image.open(scene / "data" / f"m_0_{frames[0]}.png").size[0]
    fxs = [K[0, 0] for _, K in cams.values()]
    assert max(fxs) - min(fxs) < 1e-3 * np.mean(fxs), "per-camera focal lengths differ"
    camera_angle_x = 2 * math.atan(W / (2 * float(np.mean(fxs))))

    # object frame-0 visual hull (alpha of the a_ images), inside GIC's scene box
    def mask_fn(cam_id):
        a = np.asarray(Image.open(scene / "data" / f"a_{cam_id}_{frames[0]}.png"))
        return a[..., 3] > 0
    center, half, _ = visual_hull(cams, mask_fn,
                                  gic_cfg["data"]["xyz_min"], gic_cfg["data"]["xyz_max"])
    s = TARGET_HALF / half

    # canonical LGM view order, then the remaining cams
    cam_pos = {i: c2w[:3, 3] for i, (c2w, _) in cams.items()}
    order = pick_canonical_views(cam_pos, center)
    order += [i for i in sorted(cams) if i not in order]
    R = build_world_rotation(cam_pos[order[0]], center)

    # floor: GIC config bc.ground = [point, normal, ...]; PAC-NeRF ground is y=0
    ground = gic_cfg.get("physics", {}).get("bc", {}).get("ground", [[0, 0, 0]])
    floor_y = float(ground[0][1])
    floor_level = float(s * (floor_y - center[1]))
    delta_t = frame_dt * math.sqrt(s)
    print(f"[world] scale {s:.4f}, floor_level {floor_level:.4f} (world y={floor_y}), "
          f"delta_t {delta_t:.5f}")

    # ---- case dir
    case_dir = vid2sim / "dataset" / "PACNeRF" / args.out_name
    data_dir = case_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    last = frames[-1]
    for v, o in enumerate(order):
        shutil.copyfile(scene / "data" / f"r_{o}_-1.png", data_dir / f"r_{v}_-1.png")
        for f in range(args.pad_frames):
            src = min(f, last)
            for kind in ("a", "m", "r"):
                shutil.copyfile(scene / "data" / f"{kind}_{o}_{src}.png",
                                data_dir / f"{kind}_{v}_{f}.png")
    print(f"[images] {len(order)} views x {args.pad_frames} frames "
          f"(frames {n_frames}..{args.pad_frames - 1} repeat frame {last})")

    view_c2ws = [transform_c2w(cams[o][0], R, s, center) for o in order]
    write_transforms(case_dir, camera_angle_x, view_c2ws)
    write_points3d(case_dir)
    write_sim_transform(case_dir, s, center, R)
    write_gt_phys_params(
        case_dir, args.gt_yms, args.gt_prs, s,
        "GT E/nu are not shipped with the repo's PAC-NeRF data (the GIC config\n"
        "only carries optimizer inits); -1 = placeholder. Compare recovered\n"
        "best_params.yaml yms against yms_sim = yms * world_scale.")

    write_config(vid2sim / "config" / "gso.yaml",
                 vid2sim / "config" / "pacnerf" / f"{args.out_name}.yaml",
                 view_samples=len(order), floor_level=round(floor_level, 6),
                 floor_axis=2, delta_t=round(delta_t, 6),
                 predict_steps=args.predict_steps)
    print(f"[done] case {args.out_name}: dataset {case_dir}")


if __name__ == "__main__":
    main()
