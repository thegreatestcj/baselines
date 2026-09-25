# baselines

Self-contained codebase for running the system-identification baselines
(PAC-NeRF, GIC, MASIV, NeuMA, Spring-Gaus, Vid2Sim, OmniPhysGS, MOSIV) and evaluating
them with a unified protocol. No external repos or data needed beyond the steps below.
All public-benchmark data in https://huggingface.co/datasets/HoneyLane/gic-baselines-data;
our own dataset (PhysON) in https://huggingface.co/datasets/cmu-robotics-institute/PhysON
(private, see below). Setup scripts below.

## PhysON (our dataset) quickstart

Two task groups run the baselines on PhysON: `omniphysgs_physon_het` (OmniPhysGS on the
heterogeneous single-object scenes: two material regions per object, forced or
gravity-only) and `mosiv_physon_mo` (MOSIV on the multi-object scenes; MOSIV's released
code handles exactly two objects, the vendored copy is generalised to any number, so all
scenes run). Each group has two subsets: the defaults are `singleobject_heterogeneous`
(14 scenes, 24 fps physics) and `multiobject_heterogeneous` (10 scenes with 2-6 objects,
24 fps, gravity only); `PHYSON_HET_SUBSET=singleobject_heterogeneous_new` (12 scenes,
96 fps, declared force fields) and `PHYSON_MO_SUBSET=multiobject_heterogeneous_new`
(20 scenes, 2-4 objects, 80 fps, force fields) select the newer variants — set
`PHYSON_SUBSETS` accordingly for `setup_data.sh`. Everything below is what a fresh
machine needs; nothing else has to be edited.

```bash
bash env/setup_env.sh main         # conda env "baselines": MOSIV, converters, eval, hf CLI
bash env/setup_env.sh omniphysgs   # conda env "masiv" + venv "omniphysgs" layered over it
hf auth login                      # token with READ access to the private HF repo
                                   # cmu-robotics-institute/PhysON (ask us for org access)
bash setup_data.sh                 # public pack + PhysON, wires all symlinks
bash gen_tasks.sh mosiv_physon_mo omniphysgs_physon_het > tasks.txt
bash run_queue.sh 0,1,2,3 1        # your GPUs, 1 worker per GPU (80G class)
```

The `hf` CLI is installed into the `baselines` env by `setup_env.sh main`;
without org access to the PhysON repo `setup_data.sh` fails at the PhysON
step (set `PHYSON_SUBSETS=` to skip it, see below). No file in the repo needs
editing on a new machine: interpreters are found under `$(conda info --base)/envs`
by `env.sh`, and machine-specific overrides go into the gitignored `env.local.sh`.

`PHYSON_SUBSETS` (default `singleobject_heterogeneous multiobject_heterogeneous`)
selects what `setup_data.sh` fetches from the PhysON repo; `PHYSON_SUBSETS=` (empty)
skips PhysON so the public benchmark setup works without org access. Scenes still
downloading are skipped by `gen_tasks.sh` (stderr note) — rerun it after the
download completes. Each task is resumable
(converter → recon → fit/train → rollout → eval → overlay → `DONE`).

External force: every PhysON scene ships its analytic force field
(`force_field.npz`, gravity excluded on the way in). It is a declared scene
condition, so both methods apply it as known input by default (`oracle`).
`OMNIPHYSGS_PHYSON_ARGS='sim.force_mode=none'` / `MOSIV_PHYSON_CONVERT_ARGS='--force_mode none'`
run the methods as-is (ablation). `OMNIPHYSGS_PHYSON_ARGS` takes any OmegaConf k=v
override (e.g. `train.epochs=10`), `OMNIPHYSGS_RECON_ARGS='--iterations 15000'` sets
the static 3DGS budget, and `MOSIV_PHYSON_CONVERT_ARGS` takes the converter's options
(`--iter_cnt 80 --n_frames 48 --vel_iter_cnt 80 --bc_style 2`, see
`eval/convert_physon_to_mosiv.py --help`). They expand when `gen_tasks.sh` runs, so
`tasks.txt` records them.

Outputs per scene, under `results/physon_singleobject_heterogeneous_new/omniphysgs/`
and `results/physon_multiobject_heterogeneous/mosiv/` (subset name in the path): `<scene>.json` (per-frame CD/EMD vs GT
particles; OmniPhysGS also PSNR/SSIM/LPIPS on the held-out camera) and the
silhouette overlay of the held-out camera, `<scene>_overlay.mp4` /
`<scene>_overlay_frames.png` / `<scene>_overlay_iou.json` (GT red, prediction
cyan, overlap white, per-frame IoU) — the standard way test results are
presented here. `python eval/aggregate.py --by_group` rolls the jsons up. Raw runs
live in `OmniPhysGS/outputs/PhysON/<subset>/<scene>` (`particles/`,
`renders_test/`, `renders_test_alpha/`, `material.ply`, `params.json`, `metrics.json`)
and `MOSIV/output/physon/<subset>/<scene>` (`<scene>-pred.json` fitted per-object
parameters, `mpm/simulation_<f>.ply`, `img_render/`, `prediction_metrics.json`
with per-object Chamfer).

### Full run and what it costs

Default groups: 14 OmniPhysGS scenes (`singleobject_heterogeneous/*`) and 10 MOSIV scenes
(`multiobject_heterogeneous/*`, 2-6 objects each). Measured on one H200 per task with
nothing else on the GPU (the `_new` subsets were measured; the 24 fps subsets are
extrapolated from their 4x higher substep count per frame):

| task | budget (defaults) | wall time | peak GPU memory |
|---|---|---|---|
| OmniPhysGS, `_new` scene (96 fps) | static 3DGS 15k it, v0 closed form + 30 steps, 10 epochs × 6 stages × 10 steps (600 Adam steps, 4 views/frame, 8-frame BPTT windows, 69 substeps/frame) | ≈ 6.8 h | 23–31 GB |
| OmniPhysGS, 24 fps scene | as above but 200 substeps/frame (the dataset's own substep), 4-frame BPTT windows, `train.epochs=6` (12 stages × 10 steps per epoch) | ≈ 12 h | ≈ 35–45 GB |
| MOSIV scene | object-aware 3DGS 40k it, lifting, 3 × 80 velocity it, 80 parameter it over 48 frames (upstream default 300 ≈ 15 h) | ≈ 5 h (2 objects); CFL halving on fluid/sand scenes up to 2–3x | 30–45 GB (taichi cap `TI_DEVICE_MEMORY_GB=20` + torch) |

Budget ≈ 170 GPU-hours for the 14 OmniPhysGS scenes and ≈ 60–100 for the 10 MOSIV
scenes. Use as many workers per GPU as the memory allows (e.g. `bash run_queue.sh 0,1,2,3 3`
on 140 GB cards, 1–2 per 80 GB card); they share compute, so a wave of N concurrent
tasks takes roughly N/2 times a single task. `OMNIPHYSGS_PHYSON_ARGS='train.epochs=10'`
restores the full budget on the 24 fps subset. The cost is inherent to both methods:
every optimiser step runs the differentiable MPM forward *and* backward over the clip
(OmniPhysGS: 96³ grid × substeps × frames per window in pure PyTorch; MOSIV: 150k
particles × 48 frames × 200–400 substeps/frame in taichi with 100-substep re-forward
checkpointing, ≈ 2.5–3.5 min per parameter iteration).

Monitoring: `tasks.txt` (a task line disappears when a worker takes it),
`logs/<group>_<scene>.log` (stage prints, `[train]`/`Training progress` lines),
`timings.csv` (one row per finished task: tag, GPU, start, end, seconds, ok/fail),
`<workdir>/…/DONE` per scene. A killed machine loses nothing: rerun
`gen_tasks.sh` (finished scenes are skipped) and `run_queue.sh`; each stage resumes
from its checkpoints.

Key metrics, all written by `eval/eval_scene.py` + `eval/overlay_video.py` per scene
and rolled up by `python eval/physon_summary.py` (markdown table on stdout +
`results/physon_summary.csv`):

- **CD** — symmetric squared Chamfer distance between predicted and GT particles per frame,
  8192 samples each, reported in 10³ mm² (the PAC-NeRF / MASIV / Spring-Gaus convention);
  MOSIV additionally per object (`prediction_metrics.json`).
- **EMD** — earth mover's distance on 2048 samples, metres.
- **PSNR / SSIM / LPIPS** — renders of the held-out camera (single-object: cam 1,
  multi-object: cam 0) vs the GT frames; OmniPhysGS only (MOSIV's rollout renders silhouettes).
- **silhouette IoU** — per frame on the held-out camera from the overlay video
  (mean over the clip and last frame; the overlays are how results are shown here).
- **fitted parameters vs GT** — per expert / per object (E, ν, yield stress, friction,
  v0) from `params.json` / `<scene>-pred.json` against the scene metadata.

OmniPhysGS adaptation (details in `OmniPhysGS/README_baselines.md`): upstream
fits its per-particle constitutive mixture (KNN-transformer over expert
elasticity/plasticity models + PyTorch MPM) to a video-diffusion SDS prior;
here the SDS term is replaced by multi-view photometric + alpha supervision
against the observed frames, the expert parameters (E, nu, yield stress,
friction, cohesion) become learnable, the initial velocity comes from a
closed-form fit of the triangulated silhouette centroid (frozen during the
material fit), the known external force is fed to the MPM grid, and the
Gaussians come from a static 3DGS of frame 0 with internal particle filling.
Upstream `main.py` is untouched; the new entry points are `recon_static.py` and `fit.py`.

MOSIV adaptation (details in `MOSIV/README_baselines.md`): the authors' code
(private repo, vendored) runs unchanged except for (a) the GenesisMO input
conversion — PhysON ships no instance masks, so oracle per-object masks are
rasterised from the GT particles (MOSIV's own benchmark provides simulator
masks), per-object GT clouds are split by the metadata particle ranges, material
classes come from metadata and initial parameters are MOSIV's per-class defaults;
(b) the released code's two-object assumption (object logits [N,3], [H,W,2] masks,
obj1/obj2 branches in lifting, alpha targets and export) generalised to K objects
(ids 1..K in config order; masks are rasterised in passes of three channels);
(c) the declared external force, when the scene has one, added to the taichi MPM
grid update; (d) the physical frame rate and a `separate` floor; (e)
`export_prediction.py`, which re-simulates the fitted scene and writes
particles/silhouettes instead of MOSIV's appearance-refit prediction script.


## Vid2Sim-only quickstart (for the current handoff)

```bash
bash env/setup_env.sh vid2sim              # masiv base env + vid2sim venv only
hf auth login                              # token with read access to the data repo
bash setup_data.sh                         # downloads incl. Vid2Sim ckpts + GSO
bash gen_tasks.sh vid2sim_gso > tasks.txt  # 12 GSO cases (~20 min each)
bash run_queue.sh 0,1,2,3,4,5,6,7 1        # the GPUs you were given, 1 worker/GPU
```

`vid2sim_pacnerf` (10 PAC-NeRF elastic scenes) and `vid2sim_sg` (7 Spring-Gaus
scenes) run the benchmark adapters (`eval/convert_*_to_vid2sim.py`, inline in
each task) plus the future-window driver `eval/vid2sim_future.py`, which also
works on finished GSO cases (renders steps 16-23, per-window PSNR/SSIM,
per-step point clouds for `eval/eval_scene.py`).
Set `BASELINES_PY` to the vid2sim venv python in `env.local.sh` if you skip
the full env and still want `eval/`.

## Setup (once, everything)

```bash
bash env/setup_env.sh     # conda env "baselines" + NeuMA venv + CUDA builds
hf auth login             # token with read access to the data repo
bash setup_data.sh        # ~12G download, unpacks and wires all symlinks
```


## Run

Most public-benchmark baseline numbers are quoted from published results.
Required runs: `gic` (45; also the recon prerequisite for NeuMA), then
`neuma45` (regenerate tasks after gic finishes), and later the our-dataset
group. `pacnerf`/`masiv`/`sgs` groups are optional verification only.

```bash
bash gen_tasks.sh gic > tasks.txt        # required now (45 tasks)
bash run_queue.sh 0,1 2                  # GPUs 0 and 1, 2 workers per GPU
```

On 40G GPUs (A100-40G) run **1 worker per GPU**; 2 workers per GPU need
~60G+. Example for a 4-node cluster of 8 GPUs each: on node $i$ of 4, run
`bash gen_tasks.sh --shard $i/4 > tasks.txt && bash run_queue.sh 0,1,2,3,4,5,6,7 1`;
the whole public batch then finishes in roughly half a day.
Any subset of GPUs works: the queue only touches the cards you list, and
rerunning the same command later resumes where it left off (finished tasks
are skipped), so partial or changing allocations are fine.
NeuMA's full-fidelity configs were tuned on 80G GPUs and may not fit in 40G.

Each task = training + rollout + `DONE` marker, 2-4 h on one GPU. The queue
is resumable (rerun the same command after a crash or Ctrl-C), retries each
failure once, and logs per-task wall time to `timings.csv` and output to
`logs/`.

Multi-node, one shard per machine:

```bash
bash gen_tasks.sh --shard 0/3 > tasks.txt   # machine 0 of 3; merge results/ after
```

NeuMA's own benchmark data is not in the data pack yet; fetch it per
`NeuMA/README.md` before enabling the `neuma` task group.

## Evaluate

```bash
# per scene: particle CD/EMD (+ PSNR/SSIM/LPIPS with frame globs)
python eval/eval_scene.py --pred_plys '...' --gt_plys '...' --split 14 \
    --out results/<method>/<scene>.json
# roll everything up: results/summary.csv + markdown table
python eval/aggregate.py --by_group
```

Novel-interaction rollouts for a fitted scene (two seeded perturbations,
identical across methods):

```bash
bash run_novel.sh <GPU> GIC/output/pacnerf/<scene> GIC/data/pacnerf/<scene> \
    GIC/config/predict/<material>.json
```

## Layout

- `PAC-NeRF/ GIC/ MASIV/ NeuMA/ Spring-Gaus/ Vid2Sim/ OmniPhysGS/ MOSIV/` — vendored
  baseline code, upstream commits pinned in `VERSIONS.md`, local patches
  listed there.
- `env/` — environment spec + vendored CUDA deps. `eval/` — metrics, dataset
  converters (`convert_*_to_*.py` + templates), `overlay_video.py`.
- `data_store/` (after setup), `results/`, `logs/`, `timings.csv` — outputs.
