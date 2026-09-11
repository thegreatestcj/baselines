#!/bin/bash
# Generate the task queue for run_queue.sh.
# Usage: bash gen_tasks.sh [--shard i/N] [pacnerf] [gic] [masiv] [sgs] [neuma] > tasks.txt
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
# No default: public-benchmark baseline numbers are QUOTED in the paper
# (yi2025masiv); these groups exist for optional verification runs only.
# The required batch is the our-dataset group (pending the data converter).
[ $# -gt 0 ] || { echo "usage: gen_tasks.sh [--shard i/N] pacnerf|gic|masiv|sgs|neuma|neuma45|vid2sim_gso|vid2sim_pacnerf|vid2sim_sg ..." >&2; exit 1; }
ALL="$@"

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

if [[ " $ALL " == *" masiv "* ]]; then
  # MASIV upstream's README batch mode (run.py) is not in the released code,
  # so MASIV batches through this queue like everything else. env.pretrain
  # follows their documented batch invocation (jelly). Own conda env.
  source env.sh
  if [ -x "${MASIV_PY:-}" ]; then
    masiv_task() { # mat idx datasub gtsub
      local out="output/pacnerf/$1/$2" src="data/PAC-NeRF-Data/data/$3" gt="data/PAC-NeRF-Data/simulation_data/$4"
      emit "masiv:$1/$2" "$PWD/MASIV" "$out/DONE" \
        "$MASIV_PY train_dynamic.py --config_path config/pacnerf/$1/default.yaml --source_path $src --model_path $out --reg_scale --reg_alpha env.pretrain=jelly sim.center=2.0 sim.size=4.0 && $MASIV_PY predict.py --config_path config/pacnerf/$1/default.yaml --source_path $src --model_path $out --gt_path $gt --reg_scale --reg_alpha env.pretrain=jelly sim.center=2.0 sim.size=4.0 --load_iter -1 --iteration 40000 && touch $out/DONE"
    }
    for i in {0..9}; do
      for mat in elastic newtonian non_newtonian; do masiv_task "$mat" "$i" "$mat/$i" "$mat/$i"; done
      masiv_task plasticine "$i" "plasticine_batch/$i" "plasticine/$i"
    done
    for i in {0..4}; do masiv_task sand "$i" "sand_batch/$i" "sand/$i"; done
  else
    echo "gen_tasks: masiv env not found (run env/setup_env.sh); skipping masiv tasks" >&2
  fi
fi

if [[ " $ALL " == *" vid2sim_gso "* ]]; then
  # Vid2Sim on its 12-case GSO test set. Full pipeline per case (Stage I
  # predictor + LGM, Stage II 3DGS/LBS refine + joint opt, ~1.5-3h each).
  # Own venv (see env/setup_env.sh); gated on it like masiv.
  source env.sh
  if [ -x "${VID2SIM_PY:-}" ]; then
    for s in backpack bell blocks bus cream elephant grandfather leather lion mario sofa turtle; do
      emit "vid2sim:$s" "$PWD/Vid2Sim" "outputs/$s/DONE" \
        "$VID2SIM_PY run_pipeline.py --data_name $s && touch outputs/$s/DONE"
    done
  else
    echo "gen_tasks: vid2sim env not found (run env/setup_env.sh); skipping vid2sim tasks" >&2
  fi
fi

if [[ " $ALL " == *" vid2sim_pacnerf "* ]]; then
  # Vid2Sim on the 10 PAC-NeRF elastic scenes. The converter is cheap and
  # idempotent, so it runs inline before each job; then the full pipeline
  # and the future driver (14 real frames -> replay 0..13 scored, sim rolled
  # to step 15 so sim_points/ plys line up with the GT particle sequences
  # for eval/eval_scene.py).
  source env.sh
  if [ -x "${VID2SIM_PY:-}" ]; then
    vid2sim_pacnerf_task() { # idx
      local n="elastic_$1"
      emit "vid2sim:elastic/$1" "$PWD/Vid2Sim" "outputs/$n/DONE" \
        "$VID2SIM_PY ../eval/convert_pacnerf_to_vid2sim.py --scene_data ../GIC/data/pacnerf/elastic/$1 --gic_cfg ../GIC/config/pacnerf/elastic/default.json --out_name $n && $VID2SIM_PY run_pipeline.py --config config/pacnerf/$n.yaml --dataset_dir dataset/PACNeRF --data_name $n && $VID2SIM_PY ../eval/vid2sim_future.py --config config/pacnerf/$n.yaml --dataset_dir dataset/PACNeRF --data_name $n --split 14 --total_frames 14 && touch outputs/$n/DONE"
    }
    for i in {0..9}; do vid2sim_pacnerf_task "$i"; done
  else
    echo "gen_tasks: vid2sim env not found (run env/setup_env.sh); skipping vid2sim_pacnerf tasks" >&2
  fi
fi

if [[ " $ALL " == *" vid2sim_sg "* ]]; then
  # Vid2Sim on the 7 Spring-Gaus mpm_synthetic scenes (MASIV comparison
  # set). Same shape as vid2sim_pacnerf; 30 frames = observed 20 + future
  # 10, so the driver scores replay 0..19 and future 20..29 (GT particle
  # plys for CD live in Spring-Gaus/data/mpm_synthetic/simulation/<scene>).
  # Case names carry an sg_ prefix so outputs/ can't collide with the GSO
  # cream case.
  source env.sh
  if [ -x "${VID2SIM_PY:-}" ]; then
    vid2sim_sg_task() { # scene
      local n="sg_$1"
      emit "vid2sim:sg/$1" "$PWD/Vid2Sim" "outputs/$n/DONE" \
        "$VID2SIM_PY ../eval/convert_springgaus_to_vid2sim.py --scene_data ../Spring-Gaus/data/mpm_synthetic/render/$1 --sg_cfg ../Spring-Gaus/config/mpm_synthetic/$1.yaml --out_name $n && $VID2SIM_PY run_pipeline.py --config config/springgaus/$n.yaml --dataset_dir dataset/SpringGaus --data_name $n && $VID2SIM_PY ../eval/vid2sim_future.py --config config/springgaus/$n.yaml --dataset_dir dataset/SpringGaus --data_name $n --split 20 --total_frames 30 && touch outputs/$n/DONE"
    }
    for s in torus cross cream apple toothpaste chess banana; do vid2sim_sg_task "$s"; done
  else
    echo "gen_tasks: vid2sim env not found (run env/setup_env.sh); skipping vid2sim_sg tasks" >&2
  fi
fi

if [[ " $ALL " == *" neuma45 "* ]]; then
  # NeuMA on the 45 PAC-NeRF sequences (no published numbers exist).
  # Each task: convert (needs the scene's finished GIC recon) -> full-config
  # finetune -> future rollout. Emitted only for scenes whose gic task is
  # DONE; rerun gen_tasks.sh after the gic group finishes to pick up more.
  neuma45_task() { # mat idx datasub
    local gicout="GIC/output/pacnerf/$1/$2" name="$1_$2"
    local prior=jelly
    case "$1" in plasticine) prior=plasticine;; sand) prior=sand;; esac
    [ -f "$gicout/DONE" ] || return 0
    emit "neuma45:$1/$2" "$PWD" "NeuMA/experiments/logs/DONE_$name" \
      "NeuMA/.venv/bin/python eval/convert_pacnerf_to_neuma.py --scene_data GIC/data/pacnerf/$3 --gic_out $gicout --gic_cfg GIC/config/pacnerf/$1/default.json --out_name $name --pretrained_ckpt experiments/base_models/${prior}_0300.pt && cd NeuMA && PYTHONPATH=. .venv/bin/python experiments/finetune.py -c experiments/configs/pacnerf/$name.yaml && PYTHONPATH=. .venv/bin/python experiments/render.py -c experiments/configs/pacnerf/$name.yaml --eval_steps 16 --transform_file eval_dynamic.json --load_lora 1000_lora.pt --video_name batch --skip_frames 1 --init_frame 0 && cd .. && touch NeuMA/experiments/logs/DONE_$name"
  }
  for i in {0..9}; do
    for mat in elastic newtonian non_newtonian; do neuma45_task "$mat" "$i" "$mat/$i"; done
    neuma45_task plasticine "$i" "plasticine_batch/$i"
  done
  for i in {0..4}; do neuma45_task sand "$i" "sand_batch/$i"; done
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

if [[ " $ALL " == *" sgs_elastic "* ]]; then
  # Spring-Gaus on the 10 PAC-NeRF elastic scenes. The data converter is cheap
  # and idempotent, so it runs inline before each training job; it also writes
  # config/pacnerf_elastic/pacnerf_$i.yaml from eval/springgaus_pacnerf_template.yaml.
  for i in {0..9}; do
    emit "sgs:elastic/$i" "$PWD/Spring-Gaus" "checkpoints/pacnerf_$i/DONE" \
      "python ../eval/convert_pacnerf_to_springgaus.py --scene_data ../GIC/data/pacnerf/elastic/$i --gt ../MASIV/data/PAC-NeRF-Data/simulation_data/elastic/$i --out_name pacnerf_$i && python train.py -g 0 --cfg config/pacnerf_elastic/pacnerf_$i.yaml --eval_cam 5 --exp_id batch_pacnerf_$i && touch checkpoints/pacnerf_$i/DONE"
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
