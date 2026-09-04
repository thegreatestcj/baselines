#!/bin/bash
# Generate the task queue for run_queue.sh.
# Usage: bash gen_tasks.sh [--shard i/N] [pacnerf] [gic] [sgs] [neuma] > tasks.txt
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
ALL=${@:-pacnerf gic sgs neuma}

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
      "python train.py --config=configs/$s.py && python test.py --config=configs/$s.py --num-frame=16 --cam-id=0 && touch $ckpt/DONE"
  done
fi

if [[ " $ALL " == *" gic "* ]]; then
  gic_task() { # mat idx src predict_cfg
    local out="output/pacnerf/$1/$2"
    emit "gic:$1/$2" "$PWD/GIC" "$out/DONE" \
      "python train_dynamic.py -c config/pacnerf/$1/default.json -s $3 -m $out --reg_scale --reg_alpha && python new_trajectory.py -c config/predict/$4.json -s $3 -m $out -vid 0 -cid 0 --save_ply && touch $out/DONE"
  }
  for i in {0..9}; do
    for mat in elastic newtonian non_newtonian; do
      gic_task "$mat" "$i" "data/pacnerf/$mat/$i" "$mat"
    done
    gic_task plasticine "$i" "data/pacnerf/plasticine_batch/$i" plasticine
  done
  for i in {0..4}; do
    gic_task sand "$i" "data/pacnerf/sand_batch/$i" granular
  done
fi

if [[ " $ALL " == *" neuma "* ]]; then
  # NeuMA runs its own synthetic benchmark (BouncyBall etc.). Uses its venv
  # interpreter directly, so run_queue's interpreter substitution skips it.
  # NOTE: full-fidelity configs; ~1-2 days/scene on 80G GPUs (paper setup).
  for cfg in NeuMA/experiments/configs/synthetic/finetune-*.yaml; do
    name=$(basename "$cfg" .yaml); name=${name#finetune-}
    [[ "$name" == *smoke* ]] && continue
    datadir=$(awk '/^ *path:/{print $2; exit}' "$cfg")
    [ -d "NeuMA/$datadir" ] || continue  # data not fetched yet
    emit "neuma:$name" "$PWD/NeuMA" "experiments/logs/DONE_$name" \
      "PYTHONPATH=. .venv/bin/python experiments/finetune.py -c experiments/configs/synthetic/finetune-$name.yaml && PYTHONPATH=. .venv/bin/python experiments/render.py -c experiments/configs/synthetic/finetune-$name.yaml --eval_steps 400 --transform_file eval_dynamic.json --load_lora 1000_lora.pt --video_name batch --debug_views e_2 --skip_frames 5 --init_frame 0 && touch experiments/logs/DONE_$name"
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
