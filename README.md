# baselines

Self-contained codebase for running the system-identification baselines
(PAC-NeRF, GIC, MASIV, NeuMA, Spring-Gaus) and evaluating them with a unified
protocol. No external repos or data needed beyond the steps below.

## Requirements

- Linux, NVIDIA GPU (>=24G), CUDA 12.x toolkit with `nvcc`, conda.
- A HuggingFace account with read access to `HoneyLane/gic-baselines-data`
  (ask the maintainer).

## Setup (once)

```bash
bash env/setup_env.sh     # conda env "baselines" + NeuMA venv + CUDA builds
hf auth login             # token with read access to the data repo
bash setup_data.sh        # ~12G download, unpacks and wires all symlinks
```

If your python/CUDA live somewhere unusual, put overrides in `env.local.sh`
(see `env.sh` for the variables); everything else stays untouched.

## Run

```bash
bash gen_tasks.sh > tasks.txt     # 102 tasks: pacnerf(45) gic(45) sgs(12) [neuma]
bash run_queue.sh 0,1 2           # GPUs 0 and 1, 2 workers per GPU
```

On 40G GPUs (A100-40G) run **1 worker per GPU**; 2 workers per GPU need
~60G+. Example for a 4-node cluster of 8 GPUs each: on node $i$ of 4, run
`bash gen_tasks.sh --shard $i/4 > tasks.txt && bash run_queue.sh 0,1,2,3,4,5,6,7 1`;
the whole public batch then finishes in roughly half a day.
NeuMA's full-fidelity configs were tuned on 80G GPUs and may not fit in 40G.

Each task = training + rollout + `DONE` marker, 2-4 h on one GPU. The queue
is resumable (rerun the same command after a crash or Ctrl-C), retries each
failure once, and logs per-task wall time to `timings.csv` and output to
`logs/`.

Multi-node, one shard per machine:

```bash
bash gen_tasks.sh --shard 0/3 > tasks.txt   # machine 0 of 3; merge results/ after
```

MASIV batches through its own multi-GPU mode instead of the queue:

```bash
cd MASIV && torchrun --nproc-per-node=8 run.py train_dynamic \
  --config_path config/pacnerf --source_path data/PAC-NeRF-Data/data \
  --model_path output/PAC-NeRF-Output --gt_path data/PAC-NeRF-Data/simulation_data \
  --reg_scale --reg_alpha env.pretrain=jelly sim.center=2.0 sim.size=4.0 --subfolder
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

- `PAC-NeRF/ GIC/ MASIV/ NeuMA/ Spring-Gaus/` — vendored baseline code,
  upstream commits pinned in `VERSIONS.md`, local patches listed there.
- `env/` — environment spec + vendored CUDA deps. `eval/` — metrics.
- `data_store/` (after setup), `results/`, `logs/`, `timings.csv` — outputs.
