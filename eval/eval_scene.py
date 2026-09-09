#!/usr/bin/env python
"""Evaluate one scene's predicted sequence against GT.

Particle metrics (CD in 10^3 mm^2, EMD in meters) from ply dirs; image
metrics (PSNR/SSIM/LPIPS) from frame dirs when given. --split marks the
first FUTURE frame (frames before it are the observable window).

Examples:
  python eval_scene.py --pred_plys 'PAC-NeRF/checkpoint/torus/simulation/*.ply' \
      --gt_plys '/data/pacnerf_gt/simulation_data/torus/*.ply' \
      --split 14 --out results/pacnerf/torus.json
"""
import argparse
import glob
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from metrics import eval_particle_sequence, eval_image_sequence, save_json


def numeric_sort(paths):
    def key(p):
        m = re.findall(r"(\d+)", Path(p).stem)
        return int(m[-1]) if m else 0
    return sorted(paths, key=key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_plys", help="glob for predicted per-frame plys")
    ap.add_argument("--gt_plys", help="glob for GT per-frame plys")
    ap.add_argument("--pred_frames", help="glob for predicted rendered frames")
    ap.add_argument("--gt_frames", help="glob for GT frames (same view)")
    ap.add_argument("--split", type=int, default=None,
                    help="index of first future frame; omit for single-window eval")
    ap.add_argument("--emd_samples", type=int, default=2048)
    ap.add_argument("--no_lpips", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True, help="output json path")
    args = ap.parse_args()

    result = {"args": vars(args)}
    if args.pred_plys:
        pred = numeric_sort(glob.glob(args.pred_plys))
        gt = numeric_sort(glob.glob(args.gt_plys))
        assert pred and gt, f"empty glob: {args.pred_plys} / {args.gt_plys}"
        result["particles"] = eval_particle_sequence(
            pred, gt, split=args.split, emd_samples=args.emd_samples)
    if args.pred_frames:
        pred = numeric_sort(glob.glob(args.pred_frames))
        gt = numeric_sort(glob.glob(args.gt_frames))
        assert pred and gt, f"empty glob: {args.pred_frames} / {args.gt_frames}"
        result["images"] = eval_image_sequence(
            pred, gt, split=args.split, use_lpips=not args.no_lpips,
            device=args.device)
    save_json(result, args.out)
    for group in ("particles", "images"):
        if group in result:
            for win in ("all", "observable", "future"):
                if result[group].get(win):
                    print(group, win, result[group][win])


if __name__ == "__main__":
    main()
