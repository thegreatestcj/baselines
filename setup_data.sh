#!/bin/bash
# One-shot data setup for the baselines repo.
# Downloads all prepared datasets/weights from the HF dataset repo and wires
# the symlinks each baseline expects. Requires: `hf` CLI logged in with a token
# that can read PhysSimCode/gic-baselines-data.
#
# Usage: bash setup_data.sh [DATA_STORE_DIR]
#   DATA_STORE_DIR: where to put the actual files (default: ./data_store)
set -euo pipefail
cd "$(dirname "$0")"
STORE=${1:-$PWD/data_store}
REPO=HoneyLane/gic-baselines-data

mkdir -p "$STORE"
hf download "$REPO" --repo-type dataset --local-dir "$STORE"

link () { mkdir -p "$(dirname "$2")"; ln -sfn "$1" "$2"; }

# PAC-NeRF: data/<material>/<id>/{all_data.json,data/,transforms_*.json}
#   (masks m_*.png are already generated for all 45 batch scenes)
link "$STORE/pacnerf"                       PAC-NeRF/data
mkdir -p PAC-NeRF/checkpoint
ln -sfn "$STORE/checkpoint/pytorch_resnet101.pth" PAC-NeRF/checkpoint/pytorch_resnet101.pth

# GIC
link "$STORE/pacnerf"                       GIC/data/pacnerf
link "$STORE/checkpoint"                    GIC/data/checkpoint
link "$STORE/spring_gaus"                   GIC/data/sgs

# MASIV (vendored here once its smoke run finishes; path pre-wired)
link "$STORE/pacnerf"                       MASIV/data/PAC-NeRF-Data/data
ln -sfn "$STORE/checkpoint/pytorch_resnet101.pth" MASIV/data/PAC-NeRF-Data/pytorch_resnet101.pth 2>/dev/null || true
link "$STORE/spring_gaus"                   MASIV/data/Spring-Gaus

# Spring-Gaus
link "$STORE/spring_gaus"                   Spring-Gaus/data

# Vid2Sim
link "$STORE/vid2sim/dataset"               Vid2Sim/dataset
link "$STORE/vid2sim/checkpoints"           Vid2Sim/checkpoints

# NeuMA: TODO once its setup lands

echo "Data store ready at $STORE; symlinks wired."
