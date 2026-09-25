#!/bin/bash
# One-shot data setup for the baselines repo.
# Downloads all prepared datasets/weights from the HF dataset repo and wires
# the symlinks each baseline expects. Requires: `hf` CLI logged in with a token
# that can read HoneyLane/gic-baselines-data.
#
# Usage: bash setup_data.sh [DATA_STORE_DIR]
#   DATA_STORE_DIR: where to put the actual files (default: ./data_store)
#   PHYSON_SUBSETS: space-separated PhysON subsets to fetch (our dataset,
#     private HF repo cmu-robotics-institute/PhysON — needs a token with read
#     access to that org). Default: the subsets the omniphysgs_physon_het and
#     mosiv_physon_mo groups use. Set
#     PHYSON_SUBSETS= (empty) to skip PhysON entirely; the public benchmark
#     setup then works without org access.
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
STORE=${1:-$PWD/data_store}
REPO=HoneyLane/gic-baselines-data
PHYSON_REPO=cmu-robotics-institute/PhysON
PHYSON_SUBSETS=${PHYSON_SUBSETS-singleobject_heterogeneous multiobject_heterogeneous}

# hf CLI from the baselines env (installed by env/setup_env.sh), with a
# PATH fallback for people who bring their own.
HF="$(dirname "$BASELINES_PY")/hf"
[ -x "$HF" ] || HF=$(command -v hf) || { echo "hf CLI not found — run env/setup_env.sh first" >&2; exit 1; }

mkdir -p "$STORE"
# hf download resumes partial downloads; retry a few times around Hub 429s.
for attempt in 1 2 3 4; do
  "$HF" download "$REPO" --repo-type dataset --local-dir "$STORE" && break
  [ "$attempt" = 4 ] && { echo "HF download failed after 4 attempts" >&2; exit 1; }
  echo "download interrupted (attempt $attempt), retrying in 90s..."; sleep 90
done

# pacnerf and the Vid2Sim GSO set ship as tars (tens of thousands of small
# files rate-limit the Hub otherwise); unpack once, then drop the tars.
if [ ! -d "$STORE/pacnerf" ] && [ -f "$STORE/pacnerf.tar" ]; then
  tar xf "$STORE/pacnerf.tar" -C "$STORE" && rm -f "$STORE/pacnerf.tar"
fi
if [ ! -d "$STORE/spring_gaus" ] && [ -f "$STORE/spring_gaus.tar" ]; then
  tar xf "$STORE/spring_gaus.tar" -C "$STORE" && rm -f "$STORE/spring_gaus.tar"
fi
if [ ! -d "$STORE/vid2sim/checkpoints" ] && [ -f "$STORE/vid2sim_ckpts.tar" ]; then
  tar xf "$STORE/vid2sim_ckpts.tar" -C "$STORE" && rm -f "$STORE/vid2sim_ckpts.tar"
fi
if [ ! -d "$STORE/vid2sim/dataset" ] && [ -f "$STORE/vid2sim_gso.tar" ]; then
  tar xf "$STORE/vid2sim_gso.tar" -C "$STORE" && rm -f "$STORE/vid2sim_gso.tar"
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

# Vid2Sim: pretrained ckpts + GSO test set (12 cases). dataset/ itself is
# vendored (holds a README), so only the GSO subdir is linked.
link "$STORE/vid2sim/checkpoints"           Vid2Sim/checkpoints
link "$STORE/vid2sim/dataset/GSO"           Vid2Sim/dataset/GSO

# NeuMA: TODO once its setup lands

# PhysON (our dataset): <subset>/<scene>/{all_data.json,data/,point_clouds/,
# physics.h5,force_field.npz,...}. One download call per subset (each is
# tens of thousands of small files; same resume + 429-retry loop as above).
# The physon_* task groups read it through OmniPhysGS/data/PhysON and MOSIV/data/PhysON.
if [ -n "$PHYSON_SUBSETS" ]; then
  for sub in $PHYSON_SUBSETS; do
    for attempt in 1 2 3 4; do
      "$HF" download "$PHYSON_REPO" --repo-type dataset --local-dir "$STORE/physon" \
        --include "$sub/*" && break
      [ "$attempt" = 4 ] && { echo "PhysON download ($sub) failed after 4 attempts" >&2; exit 1; }
      echo "PhysON download ($sub) interrupted (attempt $attempt), retrying in 90s..."; sleep 90
    done
  done
  link "$STORE/physon"                      MASIV/data/PhysON
  link "$STORE/physon"                      OmniPhysGS/data/PhysON
  link "$STORE/physon"                      MOSIV/data/PhysON
else
  echo "PHYSON_SUBSETS empty: skipping PhysON (our dataset)."
fi

echo "Data store ready at $STORE; symlinks wired."
