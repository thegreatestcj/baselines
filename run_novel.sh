#!/bin/bash
# Novel-interaction rollouts for fitted GIC scenes.
# Protocol: per scene 2 sampled variants (gravtilt, velperturb) — see eval/gen_novel_cfg.py.
# Pure forward simulation from the identified physics; no re-fitting.
#
# Usage: bash run_novel.sh <GPU> <model_path> <source_path> <predict_template> [cid]
#   e.g. bash run_novel.sh 0 GIC/output/pacnerf/torus GIC/data/pacnerf/torus \
#          GIC/config/predict/elastic.json
# Outputs per variant: <model_path>/novel_<variant>/ (renders + mpm/*.ply)
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
GPU=${1:?gpu id}; MODEL=${2:?model_path}; SRC=${3:?source_path}; TPL=${4:?predict template}; CID=${5:-0}

for variant in gravtilt velperturb; do
  cfg=$("$BASELINES_PY" eval/gen_novel_cfg.py "$MODEL" "$TPL" "$variant" "$CID")
  echo "[novel] $variant -> $cfg"
  ( cd GIC && CUDA_VISIBLE_DEVICES=$GPU "$BASELINES_PY" new_trajectory.py \
      -c "../$cfg" -s "../$SRC" -m "../$MODEL" -vid 0 -cid "$CID" \
      --save_ply --out_name "novel_$variant" ) \
    > "logs/novel_$(tr / _ <<< "$MODEL")_$variant.log" 2>&1
done
echo "novel rollouts done: $MODEL"
