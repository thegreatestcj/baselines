# Vendored upstream repos

Third-party baseline code is vendored directly in this repo (their `.git` removed).
Local modifications are committed on top; diff against the commits below to see them.

| Dir | Upstream | Pinned commit |
|---|---|---|
| PAC-NeRF/ | https://github.com/xuan-li/PAC-NeRF.git | b613048557d0648e885697ececbef80297defac0 |
| GIC/ | https://github.com/Jukgei/gic.git | b523851ed4343109ff67b0ea123e0a154af2a40c |
| Spring-Gaus/ | https://github.com/Colmar-zlicheng/Spring-Gaus.git (+submodules) | 62a1bb5dbe83fe4396efa7048d3226754ac8fe1d |
| MASIV/ | https://github.com/Skaldak/MASIV.git | f05ff17 (vendored with velocity/constitution smoke validated) |
| NeuMA/ | https://github.com/XJay18/NeuMA.git | (cloned 2026-09-02; extern rasterizer pinned 59f5f77e) |
| Vid2Sim/ | https://github.com/CzzzzH/Vid2Sim.git | 8fbb26d7e16ca03b5322e490c7a8d422f70c2e57 |
| MOSIV/ | https://github.com/yizhou42/MOSIV.git (private; authors' code of Liu et al., ICLR 2026) | 4d0077db5279ac946ba6fa4f6ec345126be5a9e0 (2025-11-25); GIC-based |
| OmniPhysGS/ | https://github.com/wgsxm/OmniPhysGS.git | 5ab915b014467b911581b0c6df351713d3114a08 (2025-03-26) |
| OmniPhysGS/third_party/gaussian-splatting/ | https://github.com/graphdeco-inria/gaussian-splatting.git | 472689c0dc70417448fb451bf529ae532d32c095 (2024-04-05); vendored without its submodules and `SIBR_viewers` — the rasterizer (`diff_gaussian_rasterization`, original inria API returning `(color, radii)`) is built from Vid2Sim's vendored copy `Vid2Sim/gs/submodules/diff-gaussian-rasterization`, `simple_knn` comes from `env/third_party/simple-knn` via the masiv env |

## Local modifications so far

- PAC-NeRF/lib/pac_nerf.py — `TI_DEVICE_MEMORY_GB` env var overrides taichi
  `device_memory_fraction` (H200s are shared; fraction-based preallocation OOMs).
- GIC/train_dynamic.py — same `TI_DEVICE_MEMORY_GB` override.
- Spring-Gaus/lib/models/gaus/render.py — rasterizer returns 3+ values in our
  installed diff_gaussian_rasterization; unpack with `*_`.
- Spring-Gaus/train.py — `torch.backends.cuda.preferred_linalg_library("magma")`
  (cusolverDnCreate fails on some shared GPUs).
- Vid2Sim/gs/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h
  — added `#include <cstdint>` / `#include <cstdio>` (newer gcc fix).

### OmniPhysGS (PhysON heterogeneous single-object support; upstream `main.py` untouched)

- OmniPhysGS/recon_static.py (new) — frame-0 static 3DGS via the vendored
  `third_party/gaussian-splatting/train.py` + `recon_report.json`.
- OmniPhysGS/fit.py (new) — multi-view supervised fit (photometric + alpha loss
  replaces SDS), learnable expert parameters, learnable rigid initial velocity,
  known external force, free rollout + evaluation outputs.
- OmniPhysGS/src/constitutive_models/physical_constitutive_models/learnable.py
  (new) — expert models with `nn.Parameter` material constants.
- OmniPhysGS/src/mpm_core/ — `ext_accel` argument in `MPMModel.__call__` /
  `grid_update`; `separate` surface-collider type (zero only inward normal
  velocity).
- OmniPhysGS/src/utils/physon_force.py (new) — standalone copy of `ForceField`.
- OmniPhysGS/src/constitutive_models/{__init__,neural_constitutive_models}.py —
  `learnable` expert construction; `_MaskedGrad` per-expert gradient masking in the
  hard-softmax mixture (non-finite SVD adjoints of unassigned experts no longer
  poison the shared F gradient).
- OmniPhysGS/src/utils/physon_scene.py (new) — package/camera/GT reader.
- OmniPhysGS/third_party/gaussian-splatting/train.py — network-GUI bind failure
  tolerated (several reconstructions in parallel on one machine).

### MOSIV (PhysON multi-object support)

- MOSIV/simulator/mpm_simulator.py — external acceleration field on the taichi grid
  (`set_external_acceleration`, `set_frame`; dense box per observed frame, added to
  gravity in `grid_op`; `advance`/`advance_grad` select the frame's sample).
- MOSIV/simulator/estimator_multi.py — `_setup_external_force` (config keys
  `force_mode`, `force_npz`, `force_json`, `force_h5`, `force_box_margin`).
- MOSIV/utils/physon_force.py (new) — standalone copy of `eval/physon_common.ForceField`.
- MOSIV/train_dynamic_MO.py — `traj_save_interval` (upstream re-simulated the
  trajectory every iteration), `TI_DEVICE_MEMORY_GB` override for taichi.
- MOSIV/export_prediction.py (new) — rollout with the fitted parameters, per-frame /
  per-object plys, held-out-camera silhouettes, per-object Chamfer.
- MOSIV/README_baselines.md (new) — adaptation notes.
