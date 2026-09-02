#!/bin/bash
# GIC baseline: all 45 batch sequences, sequential on one GPU.
# Usage: bash run_gic_all.sh <GPU_ID>
set -u
GPU=${1:?need GPU id}
cd /scr/chujunta/gic_staging/baselines/GIC
export CUDA_VISIBLE_DEVICES=$GPU
export TI_DEVICE_MEMORY_GB=20
PY=/home/chujunta/miniforge3/envs/gic/bin/python

run_one () {
  local cfg=$1 src=$2 out=$3
  if [ -f "$out/DONE" ]; then echo "[skip] $src"; return; fi
  echo "[run ] $src  $(date '+%m-%d %H:%M')"
  $PY train_dynamic.py -c "$cfg" -s "$src" -m "$out" --reg_scale --reg_alpha \
      > "logs_$(tr / _ <<< "$src").log" 2>&1 \
    && touch "$out/DONE" \
    || echo "[FAIL] $src"
}

for i in 0 1 2 3 4 5 6 7 8 9; do
  for mat in elastic newtonian non_newtonian; do
    run_one config/pacnerf/$mat/default.json data/pacnerf/$mat/$i output/pacnerf/$mat/$i
  done
  run_one config/pacnerf/plasticine/default.json data/pacnerf/plasticine_batch/$i output/pacnerf/plasticine/$i
done
for i in 0 1 2 3 4; do
  run_one config/pacnerf/sand/default.json data/pacnerf/sand_batch/$i output/pacnerf/sand/$i
done
echo "ALL DONE $(date)"
