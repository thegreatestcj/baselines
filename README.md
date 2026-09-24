# baselines

Self-contained codebase for running the system-identification baselines
(PAC-NeRF, GIC, MASIV, NeuMA, Spring-Gaus, Vid2Sim, OmniPhysGS) and evaluating
them with a unified protocol. No external repos or data needed beyond the steps below.
All public-benchmark data in https://huggingface.co/datasets/HoneyLane/gic-baselines-data;
our own dataset (PhysON) in https://huggingface.co/datasets/cmu-robotics-institute/PhysON
(private, see below). Setup scripts below.

## PhysON (our dataset) quickstart

Task group `omniphysgs_physon_het` runs OmniPhysGS on the 12
`singleobject_heterogeneous_new` scenes (two material regions per object,
forced or gravity-only, plus a two-phase fluid and a sand+pusher scene).
Everything below is what a fresh machine needs; nothing else has to be edited.

```bash
bash env/setup_env.sh omniphysgs   # conda env "masiv" + venv "omniphysgs"
                                   # layered over it; needs nvcc (CUDA 12.x)
bash env/setup_env.sh main         # conda env "baselines": converters, eval, hf CLI
hf auth login                      # token with READ access to the private HF repo
                                   # cmu-robotics-institute/PhysON (ask us for org access)
bash setup_data.sh                 # public pack + PhysON, wires all symlinks
bash gen_tasks.sh omniphysgs_physon_het > tasks.txt   # 12 tasks
bash run_queue.sh 0,1,2,3 1        # your GPUs, 1 worker per GPU (80G class)
```

The `hf` CLI is installed into the `baselines` env by `setup_env.sh main`;
without org access to the PhysON repo `setup_data.sh` fails at the PhysON
step (set `PHYSON_SUBSETS=` to skip it, see below). No file in the repo needs
editing on a new machine: interpreters are found under `$(conda info --base)/envs`
by `env.sh`, and machine-specific overrides go into the gitignored `env.local.sh`.

`PHYSON_SUBSETS` (default `singleobject_heterogeneous_new`) selects what
`setup_data.sh` fetches from the PhysON repo; `PHYSON_SUBSETS=` (empty) skips
PhysON so the public benchmark setup works without org access. Scenes still
downloading are skipped by `gen_tasks.sh` (stderr note) — rerun it after the
download completes. `PHYSON_HET_SUBSET=singleobject_heterogeneous` switches
the group to the older 14-scene variant. Each task is resumable
(converter → recon → fit → rollout → eval → overlay → `DONE`).

External force: every PhysON scene ships its analytic force field
(`force_field.npz`, gravity excluded on the way in). It is a declared scene
condition, so the method applies it as known input by default (`oracle`).
`OMNIPHYSGS_PHYSON_ARGS='sim.force_mode=none'` runs the method as-is
(ablation); the same variable takes any OmegaConf k=v override (e.g.
`train.epochs=10`), and `OMNIPHYSGS_RECON_ARGS='--iterations 15000'` sets the
static 3DGS budget. They expand when `gen_tasks.sh` runs, so `tasks.txt`
records them.

Outputs per scene, under `results/physon_singleobject_heterogeneous_new/omniphysgs/`:
`<scene>.json` (per-frame CD/EMD vs GT particles, PSNR/SSIM/LPIPS on the
held-out camera) and the silhouette overlay of the held-out camera,
`<scene>_overlay.mp4` / `<scene>_overlay_frames.png` /
`<scene>_overlay_iou.json` (GT red, prediction cyan, overlap white,
per-frame IoU) — the standard way test results are presented here.
`python eval/aggregate.py --by_group` rolls the jsons up. Raw runs live in
`OmniPhysGS/outputs/PhysON/<subset>/<scene>` (`particles/`, `renders_test/`,
`renders_test_alpha/`, `material.ply`, `params.json`, `metrics.json`).

OmniPhysGS adaptation (details in `OmniPhysGS/README_baselines.md`): upstream
fits its per-particle constitutive mixture (KNN-transformer over expert
elasticity/plasticity models + PyTorch MPM) to a video-diffusion SDS prior;
here the SDS term is replaced by multi-view photometric + alpha supervision
against the observed frames, the expert parameters (E, nu, yield stress,
friction, cohesion) become learnable, a rigid initial velocity is learned,
the known external force is fed to the MPM grid, and the Gaussians come from
a static 3DGS of frame 0 with internal particle filling. Upstream `main.py`
is untouched; the new entry points are `recon_static.py` and `fit.py`.

Multi-object subset (`multiobject_heterogeneous_new`): the intended baseline is
MOSIV (Liu et al., ICLR 2026, https://arxiv.org/abs/2603.06022), whose code is
not public as of 2026-09 — nothing is wired for it yet.


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

- `PAC-NeRF/ GIC/ MASIV/ NeuMA/ Spring-Gaus/ Vid2Sim/ OmniPhysGS/` — vendored
  baseline code, upstream commits pinned in `VERSIONS.md`, local patches
  listed there.
- `env/` — environment spec + vendored CUDA deps. `eval/` — metrics, dataset
  converters (`convert_*_to_*.py` + templates), `overlay_video.py`.
- `data_store/` (after setup), `results/`, `logs/`, `timings.csv` — outputs.
