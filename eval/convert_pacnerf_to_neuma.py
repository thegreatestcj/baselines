#!/usr/bin/env python
"""Convert a PAC-NeRF benchmark scene (+ finished GIC outputs) into NeuMA's input format.

Produces, under the NeuMA repo:
  data/PACNeRF/<name>/                data_dynamic{,.json}, eval_dynamic{,.json}, data_static{,.json}
                                      (RGBA 800x800 frames; RGB from the matted m_ images, alpha
                                       from the white-background segmentation, exactly GIC's rule)
  experiments/assets/<name>/          kernels.ply (3DGS, isotropic scale expanded to 3 axes),
                                      particles_src.ply (downsampled static_0), and -- unless
                                      --skip_bindings -- particles.ply, bindings.pt, particles.npz
                                      built with NeuMA's own prepare_simulation_data (GPU needed).
  experiments/configs/pacnerf/        <name>.yaml (full) and <name>-smoke.yaml, generated from
                                      eval/neuma_pacnerf_template.yaml with scene constants filled.

Run with the NeuMA venv interpreter, e.g. from the baselines repo root:
  NeuMA/.venv/bin/python eval/convert_pacnerf_to_neuma.py \
      --scene_data GIC/data/pacnerf/torus \
      --gic_out   GIC/output/pacnerf/torus \
      --gic_cfg   GIC/config/pacnerf/torus.json \
      --out_name  torus

Conventions handled (verified empirically on torus):
  * PAC-NeRF c2w is 3x4 OpenGL-style (camera looks down -Z), identical to what NeuMA's
    readNeuMASyntheticCameras expects; we only pad it to 4x4. No scaling is applied
    (the 0.41-scaled c2w in NeuMA-Synthetic/BouncyBall is a quirk of that dataset's
    generator, not a requirement of the loader).
  * Intrinsics are passed through; NeuMA only uses fx/fy + image size for the FoV.
  * World -> sim mapping mirrors BouncyBall: a cube with the ground plane at y=0 mapped
    to the unit cube. If the cube side L != 1, gravity is divided by L (lengths shrink
    by 1/L, time is unchanged).
  * Timing: each video frame (1/fps s) is `substeps` MPM steps, dt = 1/(fps*substeps);
    substeps is chosen so dt is ~1e-3 s like NeuMA's own configs.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

REPO_ROOT = Path(__file__).resolve().parent.parent

N_F_REST = 45  # SH degree 3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene_data", required=True, help="PAC-NeRF scene dir (all_data.json + data/)")
    p.add_argument("--gic_out", required=True, help="GIC output dir for the scene")
    p.add_argument("--gic_cfg", required=True, help="GIC scene config json")
    p.add_argument("--out_name", required=True, help="Name under NeuMA/data/PACNeRF and experiments/assets")
    p.add_argument("--neuma_root", default=str(REPO_ROOT / "NeuMA"))
    p.add_argument("--template", default=str(Path(__file__).resolve().parent / "neuma_pacnerf_template.yaml"))
    p.add_argument("--eval_cams", default="5,10", help="Comma-separated camera ids held out for eval")
    p.add_argument("--max_kernels", type=int, default=30000,
                   help="Cap on Gaussian kernels (binding matrix is dense kernels x particles)")
    p.add_argument("--max_particles", type=int, default=25000, help="Cap on MPM particles")
    p.add_argument("--no_kernel_scale_boost", action="store_true",
                   help="Do not inflate kernel scales by (N_orig/N_kept)^(1/3) after subsampling")
    p.add_argument("--gic_iteration", type=int, default=None,
                   help="GIC point_cloud iteration to use (default: largest available)")
    p.add_argument("--substeps", type=int, default=0, help="MPM substeps per frame (0 = auto, dt ~ 1e-3)")
    p.add_argument("--vel_frames", type=int, default=4, help="Frames used for initial-velocity optimization")
    p.add_argument("--vel_epochs", type=int, default=100)
    p.add_argument("--const_epochs", type=int, default=1000)
    p.add_argument("--smoke_vel_epochs", type=int, default=20)
    p.add_argument("--smoke_const_epochs", type=int, default=3)
    p.add_argument("--pretrained_ckpt", default="experiments/base_models/jelly_0300.pt")
    p.add_argument("--skip_bindings", action="store_true",
                   help="Skip GPU binding precompute (finetune.py will then build them on first run, "
                        "but particles.npz cannot be written consistently)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------- cameras / images

def load_all_data(scene_dir: Path):
    """Return {cam_id: (c2w 4x4 list, intrinsic)} and the sorted list of frame ids (>= 0)."""
    entries = json.load(open(scene_dir / "all_data.json"))
    cams, frames = {}, set()
    for e in entries:
        stem = Path(e["file_path"]).stem            # r_<cam>_<frame>
        _, cam, frame = stem.split("_")
        cam, frame = int(cam), int(frame)
        c2w = [list(map(float, row)) for row in e["c2w"]]
        if len(c2w) == 3:
            c2w.append([0.0, 0.0, 0.0, 1.0])
        if cam in cams:
            assert np.allclose(cams[cam][0], c2w), f"c2w varies over time for cam {cam}"
        else:
            cams[cam] = (c2w, e["intrinsic"])
        if frame >= 0:
            frames.add(frame)
    return cams, sorted(frames)


def make_rgba(scene_dir: Path, cam: int, frame: int) -> Image.Image:
    """RGBA frame: RGB from the matted m_ image, alpha = non-white (GIC's exact mask rule)."""
    m = np.asarray(Image.open(scene_dir / "data" / f"m_{cam}_{frame}.png").convert("RGB"))
    alpha = (m.astype(int).sum(-1) != 255 * 3).astype(np.uint8) * 255
    return Image.fromarray(np.dstack([m, alpha]), "RGBA")


def write_video_data(scene_dir: Path, out_dir: Path, cams, frames, eval_cams):
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {"data_dynamic": [], "eval_dynamic": [], "data_static": []}
    for sub in splits:
        (out_dir / sub).mkdir(exist_ok=True)

    for cam, (c2w, intr) in sorted(cams.items()):
        sub = "eval_dynamic" if cam in eval_cams else "data_dynamic"
        for frame in frames:
            rgba = make_rgba(scene_dir, cam, frame)
            fname = f"r_{cam}_{frame:03d}.png"
            rgba.save(out_dir / sub / fname)
            splits[sub].append({"file_path": f"./{sub}/{fname}", "c2w": c2w, "intrinsic": intr})
            if frame == frames[0]:
                sname = f"s_{cam}_{frames[0]:03d}.png"
                rgba.save(out_dir / "data_static" / sname)
                splits["data_static"].append(
                    {"file_path": f"./data_static/{sname}", "c2w": c2w, "intrinsic": intr})

    for sub, entries in splits.items():
        with open(out_dir / f"{sub}.json", "w") as f:
            json.dump(entries, f, indent=2)
        print(f"[data] {sub}: {len(entries)} entries, images in {out_dir / sub}")


# ---------------------------------------------------------------- assets

def load_xyz(ply_path: Path) -> np.ndarray:
    v = PlyData.read(str(ply_path))["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)


def convert_kernels(src_ply: Path, dst_ply: Path, max_kernels: int, boost: bool,
                    opacity_thres: float = 0.02, seed: int = 42):
    """GIC 3DGS ply -> standard NeuMA-readable 3DGS ply.

    GIC stores an isotropic scale (single scale_0); NeuMA's GaussianModel and the
    binding code expect 3 scale columns. Prunes by opacity, subsamples to max_kernels,
    and optionally inflates scales by (N/kept)^(1/3) to preserve visual coverage.
    """
    v = PlyData.read(str(src_ply))["vertex"]
    n = v.count
    names = [pr.name for pr in v.properties]

    opacity = np.asarray(v["opacity"], dtype=np.float64)
    keep = 1.0 / (1.0 + np.exp(-opacity)) > opacity_thres
    idx = np.nonzero(keep)[0]
    rng = np.random.default_rng(seed)
    if len(idx) > max_kernels:
        idx = np.sort(rng.choice(idx, size=max_kernels, replace=False))
    kept = len(idx)

    scale_names = sorted([nm for nm in names if nm.startswith("scale_")],
                         key=lambda s: int(s.split("_")[-1]))
    scales = np.stack([np.asarray(v[nm], dtype=np.float64) for nm in scale_names], axis=1)
    if scales.shape[1] == 1:
        scales = np.repeat(scales, 3, axis=1)
    if boost and kept < n:
        scales = scales + np.log((n / kept) ** (1.0 / 3.0))  # log-space isotropic inflate

    f_rest_names = sorted([nm for nm in names if nm.startswith("f_rest_")],
                          key=lambda s: int(s.split("_")[-1]))
    assert len(f_rest_names) == N_F_REST, f"expected SH3 ({N_F_REST} f_rest), got {len(f_rest_names)}"

    out_names = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
                 + f_rest_names + ["opacity", "scale_0", "scale_1", "scale_2",
                                   "rot_0", "rot_1", "rot_2", "rot_3"])
    arr = np.empty(kept, dtype=[(nm, "f4") for nm in out_names])
    for nm in out_names:
        if nm.startswith("scale_"):
            arr[nm] = scales[idx, int(nm.split("_")[-1])]
        elif nm in ("nx", "ny", "nz") and nm not in names:
            arr[nm] = 0.0
        else:
            arr[nm] = np.asarray(v[nm])[idx]

    dst_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(dst_ply))
    print(f"[assets] kernels: {n} -> {kept} (opacity>{opacity_thres}, cap {max_kernels}, "
          f"scale boost {'on' if boost and kept < n else 'off'}) -> {dst_ply}")
    return kept


def write_particle_src(static_ply: Path, dst_ply: Path, max_particles: int, seed: int = 42):
    pts = load_xyz(static_ply)
    rng = np.random.default_rng(seed)
    if len(pts) > max_particles:
        sel = np.sort(rng.choice(len(pts), size=max_particles, replace=False))
        sub = pts[sel]
    else:
        sub = pts
    arr = np.empty(len(sub), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    arr["x"], arr["y"], arr["z"] = sub[:, 0], sub[:, 1], sub[:, 2]
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(dst_ply))
    print(f"[assets] particles: {len(pts)} -> {len(sub)} -> {dst_ply}")
    return pts, sub


def voxel_volume(pts: np.ndarray, voxel_size: float) -> float:
    """Total object volume estimated by voxel occupancy of the (full) particle cloud."""
    ijk = np.floor((pts - pts.min(0)) / voxel_size).astype(np.int64)
    occ = len(np.unique(ijk, axis=0))
    return occ * voxel_size ** 3


def build_bindings_and_npz(neuma_root: Path, asset_dir: Path, vol_total: float, seed: int):
    """Run NeuMA's own prepare_simulation_data (needs GPU), then write particles.npz
    consistent with the final particles.ply / bindings.pt ordering."""
    sys.path.insert(0, str(neuma_root))
    import torch
    import warp as wp
    from modules.tune.utils import prepare_simulation_data  # noqa: E402

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    wp.init()

    prepare_simulation_data(
        save_dir=asset_dir,
        kernels_path=asset_dir / "kernels.ply",
        particles_path=asset_dir / "particles_src.ply",
        sh_degree=3,
        opacity_thres=0.02,
        particles_downsample_factor=1,
        confidence=0.95,
        max_particles=10,
    )

    p_x = load_xyz(asset_dir / "particles.ply")
    vol = vol_total / len(p_x)
    np.savez(asset_dir / "particles.npz", p_x=p_x, vol=vol)
    print(f"[assets] particles.npz: {len(p_x)} particles, per-particle vol {vol:.3e} "
          f"(total {vol_total:.4e})")


# ---------------------------------------------------------------- config

def sim_cube(phys_cfg: dict, particles: np.ndarray):
    """World-space cube (ground plane y=0 at the bottom) mapped onto the unit sim cube.

    Mirrors BouncyBall ([-0.5,0,-0.5]..[0.5,1,0.5]) whenever the scene fits; grows the
    cube (and thus scales gravity by 1/L) otherwise.
    """
    xz_c = np.array([0.0, 0.0])
    lo, hi = particles.min(0), particles.max(0)
    margin = 0.02
    L = 1.0
    need = max((abs(lo[0] - xz_c[0]) + margin) * 2, (abs(hi[0] - xz_c[0]) + margin) * 2,
               (abs(lo[2] - xz_c[1]) + margin) * 2, (abs(hi[2] - xz_c[1]) + margin) * 2,
               hi[1] + margin)
    if need > L:
        L = float(np.ceil(need * 10) / 10)
        print(f"[cfg] scene does not fit the unit cube, using L={L} (gravity scaled by 1/L)")
    assert lo[1] > -1e-4, f"particles below the ground plane: min y = {lo[1]}"
    ori_min = [xz_c[0] - L / 2, 0.0, xz_c[1] - L / 2]
    ori_max = [xz_c[0] + L / 2, L, xz_c[1] + L / 2]
    return ori_min, ori_max, L


def bc_string(phys_cfg: dict) -> str:
    bc = phys_cfg.get("bc", {})
    ground = bc.get("ground")
    if ground is not None and len(ground) >= 3 and float(ground[2]) == 0.0:
        return "freeslip"
    return "noslip"


def write_config(template_path: Path, out_path: Path, mapping: dict):
    text = Path(template_path).read_text().format(**mapping)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(f"[cfg] wrote {out_path}")


def main():
    args = parse_args()
    scene_dir = Path(args.scene_data).resolve()
    gic_out = Path(args.gic_out).resolve()
    neuma_root = Path(args.neuma_root).resolve()
    name = args.out_name

    phys = json.load(open(args.gic_cfg))["physics"]
    n_frames = int(phys["n_frames"])
    fps = float(phys["fps"])
    gravity = [float(g) for g in phys["gravity"]]
    rho = float(phys.get("rho", 1000.0))
    voxel_size = float(phys.get("voxel_size", 0.02))

    # ---- video data
    cams, frames = load_all_data(scene_dir)
    frames = frames[:n_frames]
    eval_cams = {int(c) for c in args.eval_cams.split(",") if c != ""}
    unknown = eval_cams - set(cams)
    assert not unknown, f"eval cams {unknown} not in scene cams {sorted(cams)}"
    train_cams = [c for c in sorted(cams) if c not in eval_cams]
    print(f"[data] {len(cams)} cams ({len(train_cams)} train / {sorted(eval_cams)} eval), "
          f"{len(frames)} frames")
    data_out = neuma_root / "data" / "PACNeRF" / name
    write_video_data(scene_dir, data_out, cams, frames, eval_cams)

    # ---- assets
    asset_dir = neuma_root / "experiments" / "assets" / name
    pc_root = gic_out / "point_cloud"
    if args.gic_iteration is None:
        iters = sorted(int(p.name.split("_")[-1]) for p in pc_root.glob("iteration_*"))
        assert iters, f"no point_cloud/iteration_* under {gic_out}"
        it = iters[-1]
    else:
        it = args.gic_iteration
    kernels_src = pc_root / f"iteration_{it}" / "point_cloud.ply"
    static_src = gic_out / "mpm" / "static_0.ply"

    convert_kernels(kernels_src, asset_dir / "kernels.ply", args.max_kernels,
                    boost=not args.no_kernel_scale_boost, seed=args.seed)
    pts_full, _ = write_particle_src(static_src, asset_dir / "particles_src.ply",
                                     args.max_particles, seed=args.seed)
    vol_total = voxel_volume(pts_full, voxel_size)

    if args.skip_bindings:
        print("[assets] --skip_bindings: bindings.pt / particles.ply / particles.npz not built")
    else:
        build_bindings_and_npz(neuma_root, asset_dir, vol_total, args.seed)

    # ---- config
    substeps = args.substeps or max(1, round(1.0 / (fps * 1e-3)))
    dt = 1.0 / (fps * substeps)
    ori_min, ori_max, L = sim_cube(phys, pts_full)
    grav = [g / L for g in gravity]
    num_frames = len(frames) - 1                       # frame 0 is the initial state
    common = dict(
        name=name,
        debug_view=f"r_{train_cams[0]}",
        pretrained_ckpt=args.pretrained_ckpt,
        gravity_x=grav[0], gravity_y=grav[1], gravity_z=grav[2],
        bc=bc_string(phys),
        dt=f"{dt:.8f}",
        ori_min_x=f"{ori_min[0]:.4f}", ori_min_y=f"{ori_min[1]:.4f}", ori_min_z=f"{ori_min[2]:.4f}",
        ori_max_x=f"{ori_max[0]:.4f}", ori_max_y=f"{ori_max[1]:.4f}", ori_max_z=f"{ori_max[2]:.4f}",
        rho=rho,
        substeps=substeps,
        num_frames=num_frames,
        vel_frames=min(args.vel_frames, num_frames),
    )
    cfg_dir = neuma_root / "experiments" / "configs" / "pacnerf"
    write_config(args.template, cfg_dir / f"{name}.yaml",
                 dict(common, exp_name=f"{name}-v1", resume="false",
                      vel_epochs=args.vel_epochs, const_epochs=args.const_epochs))
    write_config(args.template, cfg_dir / f"{name}-smoke.yaml",
                 dict(common, exp_name=f"{name}-smoke", resume="true",
                      vel_epochs=args.smoke_vel_epochs, const_epochs=args.smoke_const_epochs))

    print(f"\nDone. From {neuma_root} run e.g.:\n"
          f"  PYTHONPATH=. .venv/bin/python experiments/finetune.py -c experiments/configs/pacnerf/{name}-smoke.yaml\n"
          f"  PYTHONPATH=. .venv/bin/python experiments/render.py -c experiments/configs/pacnerf/{name}-smoke.yaml \\\n"
          f"      -s {num_frames * substeps} -f {substeps} --init_frame 0 -l <NNNN>_lora.pt "
          f"-vn smoke -dv r_{train_cams[0]}")


if __name__ == "__main__":
    main()
