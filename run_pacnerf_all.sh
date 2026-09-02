#!/bin/bash
# PAC-NeRF baseline: all 45 batch sequences, sequential on one GPU.
# Usage: bash run_pacnerf_all.sh <GPU_ID>
set -u
GPU=${1:?need GPU id}
cd /scr/chujunta/gic_staging/baselines/PAC-NeRF
export CUDA_HOME=/scr/chujunta/envs/masiv
export PATH=/scr/chujunta/envs/masiv/bin:$PATH
export CUDA_VISIBLE_DEVICES=$GPU
export TI_DEVICE_MEMORY_GB=20
PY=/home/chujunta/miniforge3/envs/gic/bin/python

SCENES=""
for i in 0 1 2 3 4 5 6 7 8 9; do SCENES="$SCENES elastic/$i newtonian/$i non_newtonian/$i plasticine/$i"; done
for i in 0 1 2 3 4; do SCENES="$SCENES sand/$i"; done

for s in $SCENES; do
  ckpt="checkpoint/$(sed 's|plasticine/|plasticine_batch/|; s|sand/|sand_batch/|' <<< "$s")"
  if [ -f "$ckpt/DONE" ]; then echo "[skip] $s"; continue; fi
  echo "[run ] $s  $(date '+%m-%d %H:%M')"
  $PY train.py --config="configs/$s.py" > "logs_$(tr / _ <<< "$s").log" 2>&1 \
    && touch "$ckpt/DONE" \
    || echo "[FAIL] $s"
done
echo "ALL DONE $(date)"
