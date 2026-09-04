#!/bin/bash
# Generate the task queue for run_queue.sh.
# Usage: bash gen_tasks.sh [pacnerf] [gic] [sgs] > tasks.txt
# Each line: <tag>|<workdir>|<done-marker>|<command>
# Tasks whose done-marker exists are skipped at generation time (and again at
# run time, so a stale queue is harmless).
set -u
cd "$(dirname "$0")"

# --shard i/N: emit only every N-th task (0-based shard i). For multi-node
# runs on node-local storage: node i generates its own shard, no shared
# state needed; merge results/ afterwards.
SHARD_I=0; SHARD_N=1
if [ "${1:-}" = "--shard" ]; then
  SHARD_I=${2%%/*}; SHARD_N=${2##*/}; shift 2
fi
_emitted=0
ALL=${@:-pacnerf gic sgs}

emit() { # tag workdir done cmd
  [ -f "$2/$3" ] && return
  _emitted=$((_emitted+1))
  [ $(( (_emitted-1) % SHARD_N )) -eq "$SHARD_I" ] || return 0
  echo "$1|$2|$3|$4"
}

if [[ " $ALL " == *" pacnerf "* ]]; then
  for s in elastic/{0..9} newtonian/{0..9} non_newtonian/{0..9} plasticine/{0..9} sand/{0..4}; do
    ckpt="checkpoint/$(sed 's|plasticine/|plasticine_batch/|; s|sand/|sand_batch/|' <<< "$s")"
    emit "pacnerf:$s" "$PWD/PAC-NeRF" "$ckpt/DONE" \
      "python train.py --config=configs/$s.py && touch $ckpt/DONE"
  done
fi

if [[ " $ALL " == *" gic "* ]]; then
  for i in {0..9}; do
    for mat in elastic newtonian non_newtonian; do
      emit "gic:$mat/$i" "$PWD/GIC" "output/pacnerf/$mat/$i/DONE" \
        "python train_dynamic.py -c config/pacnerf/$mat/default.json -s data/pacnerf/$mat/$i -m output/pacnerf/$mat/$i --reg_scale --reg_alpha && touch output/pacnerf/$mat/$i/DONE"
    done
    emit "gic:plasticine/$i" "$PWD/GIC" "output/pacnerf/plasticine/$i/DONE" \
      "python train_dynamic.py -c config/pacnerf/plasticine/default.json -s data/pacnerf/plasticine_batch/$i -m output/pacnerf/plasticine/$i --reg_scale --reg_alpha && touch output/pacnerf/plasticine/$i/DONE"
  done
  for i in {0..4}; do
    emit "gic:sand/$i" "$PWD/GIC" "output/pacnerf/sand/$i/DONE" \
      "python train_dynamic.py -c config/pacnerf/sand/default.json -s data/pacnerf/sand_batch/$i -m output/pacnerf/sand/$i --reg_scale --reg_alpha && touch output/pacnerf/sand/$i/DONE"
  done
fi

if [[ " $ALL " == *" sgs "* ]]; then
  # Spring-Gaus: 7 synthetic (MASIV comparison set) + 5 real. Static stage
  # resumes natively from checkpoints/<scene>/static_gaussians.
  for s in torus cross cream apple toothpaste chess banana; do
    emit "sgs:$s" "$PWD/Spring-Gaus" "checkpoints/$s/DONE" \
      "python train.py -g 0 --cfg config/mpm_synthetic/$s.yaml --eval_cam 5 --exp_id batch_$s && touch checkpoints/$s/DONE"
  done
  for s in bun burger dog pig potato; do
    emit "sgs:real_$s" "$PWD/Spring-Gaus" "checkpoints/$s/DONE" \
      "python train.py -g 0 --cfg config/real_capture/$s.yaml --eval_cam 0 --exp_id batch_$s && touch checkpoints/$s/DONE"
  done
fi
