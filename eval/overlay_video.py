#!/usr/bin/env python
"""Silhouette-overlay comparison video: GT image | overlay (GT red, prediction cyan, overlap white).

This is the standard way test results are presented in this repo. Per frame the right panel
overlays the GT silhouette (red 230,40,40) and the predicted silhouette (cyan 40,200,220); pixels
covered by both are white, so offsets and missing motion are visible at a glance. Both panels are
cropped to the union of all silhouettes and upscaled, the header shows the frame index, whether the
frame is observed or FUTURE, and the per-frame silhouette IoU.

Inputs are per-frame image globs, sorted by the last integer in the file stem (frame index):
  --gt_rgba   RGBA frames of the held-out camera (e.g. PhysON data/a_<cam>_*.png); the GT silhouette is
              alpha > 0.5 and the left panel is RGB composited on black. Alternatively give --gt_rgb
              plus --gt_mask (grayscale masks) for datasets without RGBA.
  --pred_mask grayscale predicted alpha/mask frames of the same camera (silhouette = value > 0.5).
              Alternatively --pred_rgb frames rendered on a WHITE background (silhouette = any channel
              != 255), which is the rule MASIV/GIC readers use.

Outputs: <out>.mp4 (fps 3), <out>_frames.png (a vertical strip of sample frames) and <out>_iou.json
({'iou_per_frame': [...], 'iou_mean', 'iou_observed', 'iou_future', 'observed': N}).

Example:
  python eval/overlay_video.py --gt_rgba 'MASIV/data/PhysON/multiobject_heterogeneous_new/0_0/data/a_0_*.png' \
      --pred_mask 'MASIV/output/physon/multiobject_heterogeneous_new/0_0/img_render/*_mask.png' \
      --title 'MASIV' --subtitle 'multiobject 0_0, oracle force' --out results/physon_multiobject/masiv/0_0_overlay
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

RED, CYAN, WHITE = (230, 40, 40), (40, 200, 220), (255, 255, 255)


def numeric_sort(paths):
    def key(p):
        m = re.findall(r"(\d+)", Path(p).stem)
        return int(m[-1]) if m else 0
    return sorted(paths, key=key)


def load_font(size, bold=False):
    for cand in ([f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
                  "/usr/share/fonts/dejavu/DejaVuSans.ttf"]):
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    return ImageFont.load_default()


def read_gt(rgba_path=None, rgb_path=None, mask_path=None):
    """-> (rgb uint8 HxWx3 composited on black, silhouette bool HxW)"""
    if rgba_path is not None:
        im = Image.open(rgba_path)
        arr = np.asarray(im.convert("RGBA")).astype(np.float32) / 255.0
        if im.mode != "RGBA":  # e.g. an RGB-on-white frame: white-background rule
            arr[..., 3] = (np.asarray(im.convert("RGB")).astype(np.int32).sum(-1) != 255 * 3).astype(np.float32)
        rgb = (arr[..., :3] * arr[..., 3:4] * 255).astype(np.uint8)
        return rgb, arr[..., 3] > 0.5
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    mask = np.asarray(Image.open(mask_path).convert("L")).astype(np.float32) / 255.0 > 0.5
    return (rgb * mask[..., None]).astype(np.uint8), mask


def read_pred(mask_path=None, rgb_path=None):
    if mask_path is not None:
        return np.asarray(Image.open(mask_path).convert("L")).astype(np.float32) / 255.0 > 0.5
    rgb = np.asarray(Image.open(rgb_path).convert("RGB")).astype(np.int32)
    return rgb.sum(-1) != 255 * 3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt_rgba", help="glob of RGBA GT frames (held-out camera)")
    ap.add_argument("--gt_rgb", help="glob of RGB GT frames (with --gt_mask)")
    ap.add_argument("--gt_mask", help="glob of GT mask frames (with --gt_rgb)")
    ap.add_argument("--pred_mask", help="glob of predicted alpha/mask frames")
    ap.add_argument("--pred_rgb", help="glob of predicted RGB frames rendered on white")
    ap.add_argument("--title", default="prediction", help="right-panel title, e.g. the method name")
    ap.add_argument("--subtitle", default="", help="right-panel second line (scene, setting)")
    ap.add_argument("--observed", type=int, default=None,
                    help="index of the first FUTURE frame (frames before it are labelled observed); "
                         "default: all frames observed")
    ap.add_argument("--out", required=True, help="output prefix (writes .mp4, _frames.png, _iou.json)")
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--size", type=int, default=800, help="panel side length after cropping")
    ap.add_argument("--margin", type=int, default=40, help="crop margin in source pixels")
    ap.add_argument("--grid_frames", default="", help="comma-separated frame indices for the strip "
                    "(default: 6 evenly spaced)")
    ap.add_argument("--max_frames", type=int, default=None, help="use only the first N frames")
    args = ap.parse_args()

    if args.gt_rgba:
        gt_files = numeric_sort(glob.glob(args.gt_rgba))
        gt_reader = lambda i: read_gt(rgba_path=gt_files[i])
    else:
        assert args.gt_rgb and args.gt_mask, "give --gt_rgba or --gt_rgb + --gt_mask"
        gt_files = numeric_sort(glob.glob(args.gt_rgb))
        gt_masks = numeric_sort(glob.glob(args.gt_mask))
        assert len(gt_masks) == len(gt_files), "gt rgb/mask count mismatch"
        gt_reader = lambda i: read_gt(rgb_path=gt_files[i], mask_path=gt_masks[i])
    if args.pred_mask:
        pred_files = numeric_sort(glob.glob(args.pred_mask))
        pred_reader = lambda i: read_pred(mask_path=pred_files[i])
    else:
        assert args.pred_rgb, "give --pred_mask or --pred_rgb"
        pred_files = numeric_sort(glob.glob(args.pred_rgb))
        pred_reader = lambda i: read_pred(rgb_path=pred_files[i])
    assert gt_files and pred_files, f"empty glob: gt={len(gt_files)} pred={len(pred_files)}"
    n = min(len(gt_files), len(pred_files))
    if len(gt_files) != len(pred_files):
        print(f"[overlay] frame count differs (gt {len(gt_files)}, pred {len(pred_files)}); using first {n}")
    if args.max_frames:
        n = min(n, args.max_frames)
    observed = n if args.observed is None else args.observed

    gts, gt_sil, pred_sil = [], [], []
    for i in range(n):
        rgb, sil = gt_reader(i)
        p = pred_reader(i)
        if p.shape != sil.shape:
            p = np.asarray(Image.fromarray(p.astype(np.uint8) * 255).resize(sil.shape[::-1], Image.NEAREST)) > 127
        gts.append(rgb); gt_sil.append(sil); pred_sil.append(p)

    union = np.zeros_like(gt_sil[0])
    for a, b in zip(gt_sil, pred_sil):
        union |= a | b
    H, W = union.shape
    if union.any():
        ys, xs = np.where(union)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    else:
        y0, y1, x0, x1 = 0, H - 1, 0, W - 1
    side = min(max(y1 - y0, x1 - x0) + 2 * args.margin, max(H, W))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0 = max(0, min(H - side, cy - side // 2)); x0 = max(0, min(W - side, cx - side // 2))

    font, small = load_font(26, bold=True), load_font(20)

    def crop(a):
        return np.asarray(Image.fromarray(a[y0:y0 + side, x0:x0 + side]).resize((args.size, args.size), Image.BILINEAR))

    def label(a, t, s):
        im = Image.fromarray(a); dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, im.width, 60], fill=WHITE)
        dr.text((10, 4), t, fill=(0, 0, 0), font=font); dr.text((10, 34), s, fill=(70, 70, 70), font=small)
        return np.asarray(im)

    frames, ious = [], []
    for f in range(n):
        ga, pa = gt_sil[f], pred_sil[f]
        ov = np.zeros((H, W, 3), np.uint8); ov[ga] = RED; ov[pa] = CYAN; ov[ga & pa] = WHITE
        iou = float((ga & pa).sum() / max((ga | pa).sum(), 1)); ious.append(iou)
        left = label(crop(gts[f]), "GT image", f"frame {f}  {'observed' if f < observed else 'FUTURE'}")
        right = label(crop(ov), f"red = GT, cyan = {args.title}", f"{args.subtitle} | silhouette IoU {iou:.2f}".strip(" |"))
        frames.append(np.concatenate([left, right], 1))

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    import imageio
    with imageio.get_writer(str(out) + ".mp4", fps=args.fps, codec="libx264", quality=8, macro_block_size=1) as wr:
        for a in frames:
            wr.append_data(np.ascontiguousarray(a))
    idx = [int(v) for v in args.grid_frames.split(",") if v.strip()] or \
          sorted({int(round(v)) for v in np.linspace(0, n - 1, min(6, n))})
    Image.fromarray(np.concatenate([frames[i] for i in idx if i < n], 0)).save(str(out) + "_frames.png")
    summary = {"iou_per_frame": ious, "iou_mean": float(np.mean(ious)), "observed": observed,
               "iou_observed": float(np.mean(ious[:observed])) if observed > 0 else None,
               "iou_future": float(np.mean(ious[observed:])) if observed < n else None,
               "n_frames": n, "title": args.title, "subtitle": args.subtitle}
    json.dump(summary, open(str(out) + "_iou.json", "w"), indent=1)
    print(f"[overlay] {out}.mp4  IoU mean {summary['iou_mean']:.3f}  per frame:", [round(v, 2) for v in ious])


if __name__ == "__main__":
    main()
