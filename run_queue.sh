#!/bin/bash
# Queue-based batch runner: N workers per GPU pull scenes from tasks.txt.
#
# Usage:
#   bash gen_tasks.sh > tasks.txt          # or: gen_tasks.sh gic sgs > tasks.txt
#   bash run_queue.sh 1,6 2                # GPUs 1 and 6, 2 workers per GPU
#
# Design notes (why it looks like this):
# - All four pipelines resume natively (PAC-NeRF per-stage ckpts + data.pt
#   preprocessing cache, GIC recon ckpts, Spring-Gaus static_gaussians,
#   MASIV recon/velocity) — so a crashed task is retried once, cheaply.
# - Workers start staggered 45s apart: taichi kernel compile, torch JIT and
#   matting/preprocessing are CPU-heavy cold starts; simultaneous starts
#   thrash the shared CPU.
# - TI_DEVICE_MEMORY_GB caps taichi preallocation (H200s are shared;
#   the upstream fraction-based defaults OOM). TORCH_CUDA_ARCH_LIST pins the
#   one-time JIT build to sm90. TI_OFFLINE_CACHE persists compiled taichi
#   kernels across processes where the taichi version supports it.
# - timings.csv gets one row per finished task for the cost table.
#
# MASIV is NOT in this queue: use its official multi-GPU batch mode instead
#   (torchrun --nproc-per-node=N run.py train_dynamic ... --subfolder).
set -u
cd "$(dirname "$0")"
GPUS=${1:?comma-separated GPU ids, e.g. 1,6}
PER=${2:-1}
Q=$PWD/tasks.txt
LOCK=$Q.lock
CSV=$PWD/timings.csv
source "$PWD/env.sh"
PY_GIC=$BASELINES_PY
[ -f "$Q" ] || { echo "no tasks.txt — run gen_tasks.sh first"; exit 1; }
[ -x "$PY_GIC" ] || { echo "no usable python — set BASELINES_PY or run env/setup_env.sh"; exit 1; }
[ -f "$CSV" ] || echo "tag,gpu,start,end,seconds,status" > "$CSV"

pop_task() {
  flock "$LOCK" bash -c "head -n1 '$Q'; sed -i '1d' '$Q'"
}

worker() {
  local gpu=$1 wid=$2
  while :; do
    local line; line=$(pop_task)
    [ -z "$line" ] && break
    local tag workdir done cmd
    IFS='|' read -r tag workdir done cmd <<< "$line"
    if [ -f "$workdir/$done" ]; then echo "[skip] $tag"; continue; fi
    local logf="$PWD/logs/$(tr :/ __ <<< "$tag").log"
    mkdir -p "$PWD/logs"
    local t0=$(date +%s) status=ok
    echo "[gpu$gpu.w$wid] $tag  $(date '+%m-%d %H:%M')"
    ( cd "$workdir" && CUDA_VISIBLE_DEVICES=$gpu ${cmd/python/$PY_GIC} ) > "$logf" 2>&1
    if [ $? -ne 0 ]; then
      echo "[gpu$gpu.w$wid] $tag failed once; retrying in 60s"
      sleep 60
      ( cd "$workdir" && CUDA_VISIBLE_DEVICES=$gpu ${cmd/python/$PY_GIC} ) >> "$logf" 2>&1 || status=fail
    fi
    local t1=$(date +%s)
    flock "$CSV" bash -c "echo '$tag,$gpu,$t0,$t1,$((t1-t0)),$status' >> '$CSV'"
    [ $status = fail ] && echo "[FAIL] $tag (log: $logf)"
  done
  echo "[gpu$gpu.w$wid] queue empty, exiting"
}

i=0
IFS=',' read -ra G <<< "$GPUS"
for gpu in "${G[@]}"; do
  for w in $(seq 1 "$PER"); do
    sleep $((i*45)) && worker "$gpu" "$w" &
    i=$((i+1))
  done
done
wait
echo "ALL WORKERS DONE $(date)"
