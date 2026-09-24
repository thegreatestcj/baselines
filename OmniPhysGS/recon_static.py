#!/usr/bin/env python
"""Static frame-0 3D Gaussian Splatting reconstruction of a PhysON package.

Trains the vendored `third_party/gaussian-splatting/train.py` (run as a subprocess from its own
directory, exactly as upstream expects) on `<pkg>/gs_dataset` and writes `<pkg>/gs_model`. Skipped
when `gs_model/point_cloud/iteration_<N>/point_cloud.ply` already exists (resumable task queue).
Afterwards `gs_model/recon_report.json` records the Gaussian count and the PSNR of frame 0 of the
held-out camera(s) rendered with the vendored renderer, plus `gs_model/test_<cam>.png`.

  python recon_static.py --package data/PhysON/<subset>/<scene> [--iterations 15000] [--gpu 0]

Run from the OmniPhysGS directory (the vendored 3DGS code is imported relative to it).
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GS_DIR = HERE / "third_party" / "gaussian-splatting"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--package", required=True, help="package dir written by eval/convert_physon_to_omniphysgs.py")
    p.add_argument("--iterations", type=int, default=15000)
    p.add_argument("--gpu", type=int, default=0, help="index among the visible devices")
    p.add_argument("--port", type=int, default=None, help="network-GUI port (random free-ish port by default)")
    p.add_argument("--extra", default="", help="extra args appended to train.py (quoted string)")
    p.add_argument("--force", action="store_true", help="retrain even if the checkpoint exists")
    return p.parse_args()


def train(pkg: Path, model: Path, iterations: int, gpu: int, port, extra: str):
    port = port or random.randint(20000, 60000)
    cmd = [sys.executable, "train.py", "-s", str(pkg / "gs_dataset"), "-m", str(model), "-w", "--eval",
           "--iterations", str(iterations), "--save_iterations", str(iterations),
           "--test_iterations", str(iterations), "--ip", "127.0.0.1", "--port", str(port)]
    if extra:
        cmd += extra.split()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu) if "CUDA_VISIBLE_DEVICES" not in env else \
        env["CUDA_VISIBLE_DEVICES"].split(",")[gpu]
    print("[recon] " + " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=str(GS_DIR), env=env, check=True)
    return time.time() - t0


def report(pkg: Path, model: Path, iterations: int, train_time):
    """Render frame 0 of the held-out camera(s) with the vendored renderer and compute PSNR."""
    import torch
    sys.path.insert(0, str(GS_DIR))
    from scene.gaussian_model import GaussianModel  # noqa: E402
    from gaussian_renderer import render  # noqa: E402
    from src.utils.physon_scene import PhysonPackage  # noqa: E402

    class Pipe:
        convert_SHs_python = False
        compute_cov3D_python = False
        debug = False

    scene = PhysonPackage(pkg)
    gaussians = GaussianModel(3)
    gaussians.load_ply(str(model / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"))
    bg = torch.tensor([1.0, 1.0, 1.0], device="cuda")
    out = dict(n_gaussians=int(gaussians.get_xyz.shape[0]), iterations=iterations, train_time_s=train_time,
               opacity_gt_0p02=int((gaussians.get_opacity[:, 0] > 0.02).sum()), test=[])
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    out["gaussian_bbox_min"], out["gaussian_bbox_max"] = xyz.min(0).tolist(), xyz.max(0).tolist()
    import imageio.v2 as imageio
    with torch.no_grad():
        for cid in scene.test_cam_ids:
            cam = scene.gs_camera(cid)
            img = render(cam, gaussians, Pipe(), bg)["render"].clamp(0, 1)
            gt = scene.gt_rgb(cid, 0).cuda()
            psnr = float(-10 * torch.log10(((img - gt) ** 2).mean()))
            imageio.imwrite(model / f"test_{cid}.png", (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            out["test"].append(dict(cam=cid, frame=0, psnr=psnr))
            print(f"[recon] test cam {cid} frame 0: PSNR {psnr:.2f} dB")
    json.dump(out, open(model / "recon_report.json", "w"), indent=1)
    print(f"[recon] {out['n_gaussians']} gaussians; report {model / 'recon_report.json'}")


def main():
    args = parse_args()
    pkg = Path(args.package).resolve()
    model = pkg / "gs_model"
    ply = model / "point_cloud" / f"iteration_{args.iterations}" / "point_cloud.ply"
    train_time = None
    if ply.exists() and not args.force:
        print(f"[recon] {ply} exists, skipping training")
    else:
        train_time = train(pkg, model, args.iterations, args.gpu, args.port, args.extra)
        assert ply.exists(), f"training finished but {ply} is missing"
    if not (model / "recon_report.json").exists() or train_time is not None or args.force:
        os.chdir(HERE)
        report(pkg, model, args.iterations, train_time)


if __name__ == "__main__":
    main()
