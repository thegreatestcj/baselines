# MOSIV on PhysON — adaptation notes

Upstream: MOSIV, "Multi-Object System Identification from Videos" (Liu et al., ICLR 2026,
https://arxiv.org/abs/2603.06022; dataset https://huggingface.co/datasets/Hanibel/MOSIV). The
authors' code (private GitHub repo `yizhou42/MOSIV`, commit `4d0077d`, 2025-11-25) is vendored
here; it is the GIC (NeurIPS 2024) codebase extended with object-aware dynamic Gaussians
(per-Gaussian object logits supervised by instance masks), per-object Gaussian-to-continuum
lifting, and per-object continuous material parameters + rigid initial velocities fitted through
GIC's differentiable taichi MPM with per-object geometric and silhouette objectives
(`train_dynamic_MO.py`, `simulator/estimator_multi.py`). It runs in the main `baselines` env
(python 3.9 / taichi 1.2 / torch 2.4, like GIC) plus `scikit-learn` and `matplotlib`.

The released code handles **exactly two objects** (object logits `[N, 3]`, masks `[H, W, 2]`,
`prepare_gt_multi` objects 1 and 2). PhysON `multiobject_heterogeneous_new` scenes with three
objects are skipped by `gen_tasks.sh` with a note.

## Inputs MOSIV expects and how PhysON is converted (`../eval/convert_physon_to_mosiv.py`)

MOSIV reads the "GenesisMO" layout of its own benchmark. The converter writes it under
`data/PhysON_mosiv/<subset>/<scene>/` (gitignored) from a PhysON scene:

| MOSIV input | PhysON source | note |
|---|---|---|
| `all_data.json`, `data/m_<cam>_<frame>.png` | symlinked | same PAC-NeRF camera convention and image naming |
| `masks/o_<cam>_<frame>.npy` (`[H,W,2]` uint8 {0,255}) | **generated** from the GT particles of each object (`point_clouds/<f>.ply` split by `metadata.region_offsets`): z-buffered splats per object, filled inside the scene's foreground alpha (`a_*.png`) with the label of the nearest covered pixel | PhysON ships a whole-scene foreground mask only; MOSIV's benchmark provides simulator-rendered instance masks, so this is the equivalent oracle input |
| `point_clouds/{0,1}/<f>.ply` | GT particles per object | evaluation only (per-object Chamfer) |
| `metadata.json` (`obj1`/`obj2`, bounds, fps, ...) | from `metadata.json` | material *class* per object (elastic → 10, plasticine → 12, sand → 13, newtonian → 11, non_newtonian → 14) is taken from the metadata, as MOSIV assumes known classes; the *parameters* are initialised at MOSIV's per-class defaults (`generate_configs.py`), never the GT values |
| `config/physon/<subset>/<scene>.json` | generated | mirrors MOSIV's `generate_configs.py`: per-object bbox priors from the frame-0 GT particles (+5 cm), MOSIV's voxel/density settings per material mix, `fps` = PhysON physical rate (80), `dt = 1/(fps·200)`, `n_frames` (48), `iter_cnt` (300 upstream; `--iter_cnt` to shorten), `vel_iter_cnt` 80, `vel_estimation_frames` 4 |

Differences forced by the data, all documented in the config:

* **Floor**: PhysON floors let objects slide and lift off (`bc_style 2`, "separate"); MOSIV's own
  configs use a sticky floor (`0`). `--bc_style` selects it.
* **External force**: PhysON scenes are driven by a declared analytic force (curl gusts of several
  m/s², gravity separate). MOSIV has gravity only, so `simulator/mpm_simulator.py` gained an
  external acceleration field on the grid (dense box covering the force carriers, one sample per
  observed frame, added to gravity in `grid_op`; constant for autodiff) and
  `simulator/estimator_multi.py` fills it from `force_field.npz` through `utils/physon_force.py`
  (carrier mode: k-NN inverse-distance from the per-particle `external_accel`). `force_mode: oracle`
  (default, known input) or `none` (as-is ablation).
* **Trajectory saving**: upstream re-simulates the full trajectory every iteration for logging;
  `physics.traj_save_interval` (10) makes that periodic.
* **Prediction/evaluation**: `predict_multi.py` refits the appearance for 40k iterations to render
  novel frames. `export_prediction.py` instead re-simulates the fitted scene with the same estimator
  (force included), writes `mpm/simulation_<f>.ply` (all particles, world metres) and
  `mpm/object<k>_<f>.ply`, renders the held-out camera's silhouette from the particles (small
  isotropic Gaussians, one colour per object) to `img_render/<view>_<f>.png` + `_mask.png`, and
  writes `prediction_metrics.json` (Chamfer vs GT overall and per object, silhouette IoU).
  `eval/eval_scene.py` and `eval/overlay_video.py` consume these.

## Running

Through the queue (`bash gen_tasks.sh mosiv_physon_mo`, see the top-level README), or by hand from
this directory with `source ../env.sh`:

```bash
$BASELINES_PY ../eval/convert_physon_to_mosiv.py --scene_data data/PhysON/multiobject_heterogeneous_new/0_11 \
    --out data/PhysON_mosiv/multiobject_heterogeneous_new/0_11 --config_out config/physon/multiobject_heterogeneous_new/0_11.json --iter_cnt 80
$BASELINES_PY train_dynamic_MO.py -c config/physon/multiobject_heterogeneous_new/0_11.json \
    -s data/PhysON_mosiv/multiobject_heterogeneous_new/0_11 -m output/physon/multiobject_heterogeneous_new/0_11 --reg_scale --reg_alpha
$BASELINES_PY export_prediction.py -c config/physon/multiobject_heterogeneous_new/0_11.json \
    -s data/PhysON_mosiv/multiobject_heterogeneous_new/0_11 -m output/physon/multiobject_heterogeneous_new/0_11 --view_id 0
```

Cost (H200, scene 0_11, ≈140k lifted particles, 200 substeps/frame): object-aware 3DGS 40k
iterations ≈ 10 min, lifting ≈ 3 min, velocity stage ≈ 15 s/iteration (4 frames), parameter stage
≈ 3.5 min/iteration over 48 frames (forward + taichi backward), i.e. ≈ 5 h for `--iter_cnt 80`;
upstream's 300 iterations would take ≈ 18 h per scene.
