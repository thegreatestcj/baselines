# OmniPhysGS on PhysON — adaptation notes

Upstream: https://github.com/wgsxm/OmniPhysGS (ICLR'25), vendored at commit
`5ab915b` with its 3DGS dependency `third_party/gaussian-splatting` @ `472689c`
(vendored without submodules / SIBR; see `../VERSIONS.md`). Upstream's own
README is `README.md`; this file documents what the baselines repo adds and
why. Upstream `main.py` (video-diffusion SDS training) is untouched and not
used; nothing under `src/video_distillation` is imported by the new code, so
`tinycudann`, `diffusers`, `transformers` are not installed.

## What OmniPhysGS is (the part we keep)

A per-particle *constitutive mixture* model: a physics-guided network
(`src/physics_guided_network`, KNN-transformer over Gaussian features)
outputs per-particle logits over expert elasticity models (`Corotated`,
`StVK`, `Sigma`, `Volume`, `Fluid`) and expert plasticity models
(`Identity`, `DruckerPrager`, `VonMises`, `Sigma`), combined by a
straight-through hard softmax (`GumbelElasticity` / `GumbelPlasticity`), and a
pure-PyTorch MPM (`src/mpm_core/mpm_model.py`, unit cube, gravity in
`grid_update`, boundary conditions via `set_boundary_conditions`).

## What differs from upstream and why

| Upstream | Here | Why |
|---|---|---|
| Objective: SDS against a text-conditioned video diffusion model | Supervised loss against the observed multi-view PhysON frames: per view L1(RGB on white) + `loss.w_alpha` · L1(alpha) over the union bounding box of pred/GT alpha (+50 px), averaged over `train.views_per_frame` train cameras; optional Chamfer to GT particles (`loss.w_chamfer`, default 0 = video only) | PhysON is a system-identification benchmark: the videos are the observation, not a prompt |
| Expert material constants are fixed buffers (`E=2e6, nu=0.4, sigma_y=1e3, friction=25°, cohesion=0`) | `material.learnable_experts: true` → `src/constitutive_models/physical_constitutive_models/learnable.py`: `log_E`, `nu` (parametrised into (0, 0.49)), `sigma_y`, `friction_angle`, `cohesion` are `nn.Parameter`s (E init 1e5 world, clamp [1e3, 1e7]); expert lists may repeat (`Corotated ×2`, `Identity ×2` by default) so a two-region object can get two independent stiffnesses | The heterogeneous scenes differ in *parameters*, not only in expert *type*; upstream's mixture alone cannot express E=8e4 vs 1.8e6 |
| Initial state at rest | Learnable rigid initial velocity `v0` (3) + angular velocity `omega` (3): `v = v0 + omega × (x − centroid)`; phase 1 fits them on the first `train.vel_frames` frames with the material frozen | PhysON objects may start moving; matches MASIV's velocity-estimation stage |
| No external force | `sim.force_mode: oracle|none`: the scene's `force_field.npz` (non-gravitational part, via the standalone copy `src/utils/physon_force.py`) is sampled once at the grid nodes per frame, scaled to sim units and passed to `MPMModel.__call__(..., ext_accel=)` as an explicit tensor through `torch.utils.checkpoint` | The force is a declared scene condition (known input); `none` is the as-is ablation |
| Floor: `slip` collider removes the normal velocity in both directions | New surface type `separate`: zero only the inward normal component (`v·n < 0`), keep tangential | PhysON's `ground_separate` BC (no penetration, free lift-off, no friction) |
| Gaussians from the user's own 3DGS | `recon_static.py`: frame-0 static 3DGS trained with the vendored `third_party/gaussian-splatting/train.py` (subprocess, `-w` white background, `--eval`), skipped when the checkpoint exists; `recon_report.json` (n_gaussians, PSNR on the held-out camera) | Benchmark inputs are only the videos |
| Internal filling breaks `load_params`' feature computation (features only for the original Gaussians) | Features for all particles (filled particles copy attributes from the nearest Gaussian), `render_mask` renders only the originals, all are simulated; particle volume = filling cell volume, `p_mass = rho · vol` | Filling is on by default (PhysGaussian-style) and must not crash |
| World→sim: `transform2origin` about the bbox centre | Explicit scene box (frame-0 bbox ∪ force bbox + motion margin, `y_min = floor − margin`): `x_sim = (x_world − box_center)·s + 0.5`, `s = sim.box_fill / max(extent)`; floor plane ≥ 3 grid cells above 0; gravity `[0, −9.8·s, 0]`; `E_sim = E_world·s²` (lengths scale by `s`, time and density unchanged) | The floor and the whole motion must stay inside the unit cube; reported parameters are converted back to world units |
| One `dt`, no time base | `dt` ≈ 1.5e-4 s adjusted so `steps_per_frame = round(frame_dt/dt)` reproduces the physical frame rate (96 fps → 69 substeps) | PhysON frame times are physical |

Training schedule follows upstream's truncated BPTT: `train.epochs` × stages
of `sim.frames_per_stage` frames with state carried across stages,
`train.internal_epochs` Adam steps per stage (net lr 5e-5 as upstream,
expert-parameter lr 1e-2, `v0` lr 1e-2), grad `nan_to_num` + clip and
per-substep `torch.utils.checkpoint` as upstream. A non-finite state
mid-rollout skips the step (logged) instead of crashing. Checkpoints
(`checkpoints/epoch_XXXX.pth`: network, experts, `v0`/`omega`, optimizer) are
written per epoch and the latest is resumed on restart, so the task queue can
kill and rerun a job.

## Files

- `recon_static.py --package <pkg> [--iterations N]` — static 3DGS of frame 0.
- `fit.py --config <pkg>/config.yaml --tag <name> [--output outputs] [--eval_only] [k=v ...]`
  — the fit + final free rollout; `k=v` are OmegaConf overrides
  (`sim.force_mode=none`, `train.epochs=10`, `preprocessing.max_gaussians=60000`, ...).
- `src/utils/physon_force.py` — standalone `ForceField` (copied, not imported, from `eval/`).
- `src/constitutive_models/physical_constitutive_models/learnable.py` — learnable experts.
- `src/mpm_core/` — `ext_accel` argument, `separate` collider.
- Package layout written by `../eval/convert_physon_to_omniphysgs.py` under
  `data/PhysON/<subset>/<scene>/` (gitignored): `source -> <scene dir>`,
  `gs_dataset/` (frame-0 Blender-style dataset for the vendored 3DGS),
  `gs_model/` (after `recon_static.py`), `scene.json` (cameras, timing, gravity,
  floor, bbox, force paths, regions), `config.yaml` (from
  `../eval/omniphysgs_physon_template.yaml`, documented inline).

## Config keys (see `../eval/omniphysgs_physon_template.yaml` for the full, commented schema)

- `preprocessing.hull_min_views` / `hull_dilate_px` (visual-hull pruning of floaters before anything else),
  `preprocessing.opacity_threshold`, `preprocessing.max_gaussians`,
  `preprocessing.particle_filling` (+ PhysGaussian-style filling params)
- `sim.num_grids` (96), `sim.dt` (1.5e-4), `sim.box_fill`, `sim.frames_per_stage` (8), `sim.force_mode`
- `model.network` (`knn|mlp|naive`), `material.learnable_experts`, `material.rho`,
  `material.elasticity_physicals` / `material.plasticity_physicals` (expert lists, entries may repeat),
  `material.init_E` / `E_range` / `init_nu` / `init_sigma_y` / `init_friction_angle` / `init_cohesion`,
  `material.fixed_elasticity` / `fixed_plasticity` / `*_overrides` (reference rollouts, ablations)
- `train.v0_mode` (`closed_form` default: the object centroid is triangulated from the train silhouettes
  over `train.v0_frames` frames and c(t) = c0 + v0 t + a t²/2 is fitted, a ≈ g + external force as a check),
  `train.v0_refine` / `vel_frames` (5) / `vel_iters` (30) (short refinement with the material frozen),
  `train.v0_frozen_in_material_phase` (true: v0 is never co-optimised with the material),
  `train.epochs` (10), `train.internal_epochs` (10), `train.views_per_frame` (4), learning rates
- `loss.w_rgb`, `loss.w_alpha`, `loss.crop_margin`, `loss.w_chamfer` (0)
- `train.rgb_exclude_cams` (filled by the converter: train cameras whose object silhouette is black,
  e.g. the top-down view of the bread scenes, get the alpha term only)

## Cost and budget (H200, scene 0_0: 14k Gaussians + 1.6k filled particles, 96³ grid, 69 substeps/frame)

One substep forward ≈ 11 ms (4 elasticity experts 4.2 ms, MPM 4.2 ms, 5 plasticity experts 2.6 ms);
forward + backward through the per-substep checkpoints ≈ 50 ms/substep, i.e. ≈ 3.5 s per frame plus
≈ 0.4 s per rendered view. With the defaults (8-frame stages, 4 views/frame) one Adam step is ≈ 25 s,
one epoch (6 stages × 10 steps) ≈ 25 min, so the default 10 epochs are ≈ 4-5 h per scene; peak GPU
memory ≈ 23 GB. `train.views_per_frame=-1` (all 10 views) roughly doubles the step time.

## Numerical notes

- The mixture evaluates every expert on every particle (upstream); an expert's SVD adjoint on particles
  that are *not* assigned to it can be non-finite (0 × inf) and used to poison the shared gradient of F.
  `_MaskedGrad` in `src/constitutive_models/neural_constitutive_models.py` gives each expert its own
  masked view of F (gradient only from its assigned particles, non-finite entries dropped); finite
  differences confirm the masked gradient (upstream's did not match).
- The rotation used for the view-dependent SH is a Newton polar iteration (`fit.py:polar_rotation`),
  smooth at F = I where the SVD adjoint is singular (every particle at frame 0); collapsed F (det ≈ 0)
  falls back to the identity.
## Outputs (`outputs/<tag>/`)

- `particles/<f>.ply` — all simulated particles per frame, world coordinates
  (→ `eval/eval_scene.py --pred_plys`).
- `renders_test/<f>.png`, `renders_test_alpha/<f>.png`, `gt_test/<f>.png` —
  held-out camera renders, their alpha, and GT composited on white
  (→ PSNR/SSIM/LPIPS and `eval/overlay_video.py`).
- `material.ply` / `material_plasticity.ply` — particles coloured by argmax expert.
- `params.json` — learned expert parameters in world units, `v0`/`omega` in
  world units, expert usage fractions.
- `metrics.json` — per-frame Chamfer vs GT particles, PSNR/SSIM on the test
  camera, timings. `video_test.mp4`, `log.txt`, tensorboard, `checkpoints/`.

## Running

Through the task queue (`bash gen_tasks.sh omniphysgs_physon_het`, see the
top-level README), or by hand from this directory with `source ../env.sh`:

```bash
$BASELINES_PY ../eval/convert_physon_to_omniphysgs.py --scene_data data/PhysON/singleobject_heterogeneous_new/0_0 --out_name singleobject_heterogeneous_new/0_0
$OMNIPHYSGS_PY recon_static.py --package data/PhysON/singleobject_heterogeneous_new/0_0
$OMNIPHYSGS_PY fit.py --config data/PhysON/singleobject_heterogeneous_new/0_0/config.yaml --tag PhysON/singleobject_heterogeneous_new/0_0
```

Scenes `2_0` (two-phase fluid in a rigid tank) and `3_0` (sand + kinematic
pusher) contain rigid bodies that OmniPhysGS has no coupling for; they run
without the rigid bodies and are reported as such.

Touched upstream files (see `../VERSIONS.md`): `src/mpm_core/mpm_model.py` (`ext_accel`, particle
volumes), `src/mpm_core/set_boundary_conditions.py` (`separate`), `src/constitutive_models/__init__.py`
+ `neural_constitutive_models.py` (learnable experts, masked-gradient mixture),
`third_party/gaussian-splatting/train.py` (network-GUI bind tolerated to fail so several recons can
run at once). New: `recon_static.py`, `fit.py`, `src/utils/physon_scene.py`, `src/utils/physon_force.py`,
`src/constitutive_models/physical_constitutive_models/learnable.py`.
