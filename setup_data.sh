#!/bin/bash
# One-shot data setup for the baselines repo.
# Downloads all prepared datasets/weights from the HF dataset repo and wires
# the symlinks each baseline expects. Requires: `hf` CLI logged in with a token
# that can read HoneyLane/gic-baselines-data.
#
# Usage: bash setup_data.sh [DATA_STORE_DIR]
#   DATA_STORE_DIR: where to put the actual files (default: ./data_store)
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
STORE=${1:-$PWD/data_store}
REPO=HoneyLane/gic-baselines-data

# hf CLI from the baselines env (installed by env/setup_env.sh), with a
# PATH fallback for people who bring their own.
HF="$(dirname "$BASELINES_PY")/hf"
[ -x "$HF" ] || HF=$(command -v hf) || { echo "hf CLI not found — run env/setup_env.sh first" >&2; exit 1; }

mkdir -p "$STORE"
"$HF" download "$REPO" --repo-type dataset --local-dir "$STORE"

# pacnerf and the Vid2Sim GSO set ship as tars (tens of thousands of small
# files rate-limit the Hub otherwise); unpack once, then drop the tars.
if [ ! -d "$STORE/pacnerf" ] && [ -f "$STORE/pacnerf.tar" ]; then
  tar xf "$STORE/pacnerf.tar" -C "$STORE" && rm -f "$STORE/pacnerf.tar"
fi

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

# NeuMA: TODO once its setup lands

echo "Data store ready at $STORE; symlinks wired."
