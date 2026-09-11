#!/usr/bin/env python
"""Future-window evaluation driver for a finished Vid2Sim run.

Vid2Sim's stock final_simulation only replays the observed window (steps
0..15). This driver loads the finished run's best_params + refined models (no
retraining), simulates on to --predict_steps, renders every dataset view for
the future steps, and reports PSNR/SSIM per window against the dataset GT
frames:

  replay window  = frames [0, --split)   scored from the stock render_best/
                   images on disk (falls back to the re-simulated frames if a
                   file is missing -- the sim is deterministic either way)
  future window  = frames [--split, min(--predict_steps, --total_frames - 1)]
                   scored from the newly rendered frames

Like the stock pipeline, metrics compare the white-background object renders
(m_*.png) with the dataset's m_{view}_{frame}.png; the renders written to
render_future/ use the same m_/r_ naming as render_best/.

It also dumps the simulated Gaussian-center point cloud of EVERY step
(0..predict_steps) as outputs/<case>/sim_points/NNN.ply so eval_scene.py's CD
path can consume them. If the dataset case dir contains sim_transform.json
({"scale": s, "center": c, "rotation": R}, meaning x_sim = s * R @ (x_world -
c), written by the benchmark converters), the inverse transform is applied so
the plys land in the original benchmark world frame.

Run with the Vid2Sim interpreter from anywhere, e.g.:
  source env.sh
  "$VID2SIM_PY" eval/vid2sim_future.py --data_name bus

All relative --config/--dataset_dir/--output_dir paths are resolved against
the vendored Vid2Sim repo root (the script chdirs there; Vid2Sim's own code
requires it).
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VID2SIM_ROOT = REPO_ROOT / "Vid2Sim"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config/gso.yaml")
    p.add_argument("--dataset_dir", default="dataset/GSO")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--data_name", required=True)
    p.add_argument("--predict_steps", type=int, default=None,
                   help="last simulated step (default: config predict_steps, else 23)")
    p.add_argument("--split", type=int, default=None,
                   help="first future frame index (default: config frame_steps + 1 = 16)")
    p.add_argument("--total_frames", type=int, default=24,
                   help="GT frames available per view in the dataset")
    p.add_argument("--vid2sim_root", default=str(VID2SIM_ROOT))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(args.vid2sim_root)  # Vid2Sim uses cwd-relative sys.path entries
    sys.path.insert(0, args.vid2sim_root)

    import torch
    from omegaconf import OmegaConf
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor
    from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

    from utils.seeding import seed_everything
    from simulators.lbs_simulator import LBSSimulator
    from simulators.sim_utils.loading import load_gts

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    case_out = Path(args.output_dir) / args.data_name
    case_data = Path(args.dataset_dir) / args.data_name

    sim_args = OmegaConf.load(args.config)
    best_params = OmegaConf.load(case_out / "best_params.yaml")
    sim_args.tag = "best"
    sim_args.yms = best_params.yms
    sim_args.prs = best_params.prs

    predict_steps = args.predict_steps
    if predict_steps is None:
        predict_steps = int(sim_args.get("predict_steps", 23))
    split = args.split
    if split is None:
        split = int(sim_args.get("frame_steps", 15)) + 1
    last_eval = min(predict_steps, args.total_frames - 1)

    simulator = LBSSimulator(sim_args, args.dataset_dir, args.output_dir, args.data_name)
    simulator.set_material()
    simulator.load_lbs()
    simulator.initialize_simulator()

    # LBSSimulator hardcodes 24 GT frames; reload with the real frame count.
    simulator.ref = load_gts(args.dataset_dir, args.data_name,
                             int(sim_args.view_samples), args.total_frames)

    # ---- simulate to predict_steps, rendering all views + recording points
    n_views = len(simulator.gs_context["gs_views"])
    views = [simulator.gs_context["gs_views"][i] for i in range(n_views)]
    points_flat = simulator.points.flatten().unsqueeze(-1)
    positions = []
    with torch.no_grad():
        positions.append(simulator.points.detach().cpu().numpy())
        simulator.render_frame(0, views)
        for i in range(predict_steps):
            simulator.simulate_step()
            x = (simulator.B_full @ simulator.z + points_flat).reshape(-1, 3)
            positions.append(x.detach().cpu().numpy())
            simulator.render_frame(i + 1, views)
        simulator.current_step = predict_steps

    # renders (m_/r_ naming like the stock output) for every simulated frame
    simulator.save_images("future")

    # ---- per-step gaussian-center plys (benchmark world frame if transformed)
    import numpy as np
    from plyfile import PlyData, PlyElement
    tf_path = case_data / "sim_transform.json"
    scale, center, rot = 1.0, np.zeros(3), np.eye(3)
    if tf_path.exists():
        tf = json.load(open(tf_path))
        scale, center = float(tf["scale"]), np.asarray(tf["center"], dtype=np.float64)
        rot = np.asarray(tf.get("rotation", np.eye(3)), dtype=np.float64)
        print(f"[sim_points] applying inverse world transform (scale {scale})")
    ply_dir = case_out / "sim_points"
    ply_dir.mkdir(parents=True, exist_ok=True)
    for f, pts in enumerate(positions):
        pts = (pts.astype(np.float64) / scale) @ rot + center  # R^T via right-multiply
        arr = np.empty(len(pts), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        arr["x"], arr["y"], arr["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
        PlyData([PlyElement.describe(arr, "vertex")]).write(str(ply_dir / f"{f:03d}.ply"))
    print(f"[sim_points] wrote {len(positions)} plys -> {ply_dir}")

    # ---- metrics
    psnr_m = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    pred_stack = torch.stack(simulator.save_list, dim=1)  # [views, frames, 3, H, W]

    def frame_pred(view, frame):
        """Prefer the stock render_best image for replay frames; else in-memory."""
        best_png = case_out / "render_best" / f"m_{view}_{frame}.png"
        if frame < split and best_png.exists():
            img = pil_to_tensor(Image.open(best_png)).float()[:3].to(device) / 255.0
            return img
        return pred_stack[view, frame]

    def window_metrics(f0, f1):  # inclusive frame range
        ps, ss = [], []
        for v in range(n_views):
            for f in range(f0, f1 + 1):
                pred = frame_pred(v, f).clamp(0, 1).unsqueeze(0)
                gt = simulator.ref[v, f].clamp(0, 1).unsqueeze(0)
                ps.append(psnr_m(pred, gt).item())
                ss.append(ssim_m(pred, gt).item())
        return float(np.mean(ps)), float(np.mean(ss))

    result = {
        "data_name": args.data_name,
        "yms": float(best_params.yms), "prs": float(best_params.prs),
        "predict_steps": predict_steps, "split": split,
        "total_frames": args.total_frames, "views": n_views,
    }
    rp, rs = window_metrics(0, split - 1)
    result["replay"] = {"frames": [0, split - 1], "psnr": rp, "ssim": rs}
    print(f"[metrics] replay  frames 0..{split - 1}: PSNR={rp:.4f} SSIM={rs:.4f}")
    if last_eval >= split:
        fp, fs = window_metrics(split, last_eval)
        result["future"] = {"frames": [split, last_eval], "psnr": fp, "ssim": fs}
        print(f"[metrics] future  frames {split}..{last_eval}: PSNR={fp:.4f} SSIM={fs:.4f}")
    else:
        result["future"] = None
        print(f"[metrics] no GT frames beyond split={split} (total_frames={args.total_frames}); "
              f"future renders/plys still written")

    out_json = case_out / "future_metrics.json"
    json.dump(result, open(out_json, "w"), indent=2)
    print(f"[metrics] -> {out_json}")


if __name__ == "__main__":
    main()
