#!/bin/bash
# Bootstrap the conda envs / venvs on a fresh machine.
# Requires: conda, a CUDA 12.x toolkit with `nvcc`, and a GPU driver >= it.
#
#   bash env/setup_env.sh            # everything (all baselines)
#   bash env/setup_env.sh vid2sim    # Vid2Sim only: masiv env (its base
#                                    # layer) + vid2sim venv + eval deps
#
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh

# The pip sections compile CUDA extensions at install time — nvcc must be
# reachable.
if ! command -v nvcc >/dev/null; then
  echo "ERROR: nvcc not found. Point CUDA_HOME at a CUDA 12.x toolkit" \
       "(env.local.sh) and re-run." >&2
  exit 1
fi

TARGET=${1:-all}
want() { [ "$TARGET" = all ] || [ "$TARGET" = "$1" ]; }

if want main; then
# --- main env: python 3.9 + torch cu121 + taichi 1.2 stack ---
PY="$(conda info --base)/envs/baselines/bin/python"
if [ ! -x "$PY" ]; then
  conda env create -n baselines -f env/environment.yml
fi
$PY -m pip install ninja yacs termcolor gitpython "huggingface_hub[cli]"

# Compiled CUDA deps are vendored with the cstdint header fix for newer gcc
# (unpatchable as git+ installs). Two rasterizers coexist under different
# module names: GIC imports diff_gauss (jukgei fork), Spring-Gaus imports
# diff_gaussian_rasterization (its own submodule).
$PY -m pip install --no-build-isolation env/third_party/diff-gaussian-rasterization env/third_party/simple-knn
$PY -m pip install --no-build-isolation Spring-Gaus/submodules/diff-gaussian-rasterization
fi

if want masiv || want vid2sim; then
# --- MASIV env (python 3.10 + newer torch/taichi; own spec).
#     Also the base layer for the Vid2Sim venv. ---
MPY="$(conda info --base)/envs/masiv/bin/python"
if [ ! -x "$MPY" ]; then
  conda env create -n masiv -f MASIV/environment.yml
fi
# same vendored (header-patched) CUDA deps as the main env
$MPY -m pip install --no-build-isolation env/third_party/diff-gaussian-rasterization env/third_party/simple-knn
_TV=$($MPY -c "import torch; print(torch.__version__.split('+')[0])")
_CU=$($MPY -c "import torch; print('cu'+torch.version.cuda.replace('.',''))")
if [ "$TARGET" != vid2sim ]; then
  # MASIV's own heavy extras, not needed when only Vid2Sim will run.
  # torch-scatter needs the prebuilt PyG wheel index (pip build isolation
  # hides the env's torch); pytorch3d built without isolation.
  $MPY -m pip install torch-scatter -f "https://data.pyg.org/whl/torch-${_TV}+${_CU}.html" \
    || $MPY -m pip install --no-build-isolation torch-scatter
  $MPY -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
fi
fi

if want neuma; then
# --- NeuMA venv (needs old warp-lang 0.6.1, incompatible with main env) ---
if [ ! -x NeuMA/.venv/bin/python ]; then
  $PY -m venv --system-site-packages NeuMA/.venv
fi
NeuMA/.venv/bin/pip install warp-lang==0.6.1 e3nn==0.5.1 viser==0.2.3 \
  nerfview==0.0.3 pyvista==0.44.0 splines==0.3.2 natsort torchmetrics \
  tensorboardX mediapy omegaconf==2.3.0 py7zr
( cd NeuMA/extern/diff-gaussian-rasterization && ../../.venv/bin/pip install --no-build-isolation . )
fi

if want vid2sim; then
# --- Vid2Sim venv (layered over the masiv env: torch 2.4.1+cu124, diff_gauss,
# simple_knn, warp, numpy 1.26.4 come from there via --system-site-packages) ---
VPY="${VID2SIM_PY:-$(conda info --base)/envs/vid2sim/bin/python}"
if [ ! -x "$VPY" ]; then
  "$MPY" -m venv --system-site-packages "$(dirname "$(dirname "$VPY")")"
fi
VPIP="$(dirname "$VPY")/pip"
# kaolin from the official wheel index matching the masiv torch build
"$VPIP" install kaolin==0.17.0 \
  -f "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-${_TV}_${_CU}.html"
# pyg-lib is REQUIRED (torch_geometric.nn.pool.fps dies without it in Stage II)
"$VPIP" install torch-cluster pyg-lib \
  -f "https://data.pyg.org/whl/torch-${_TV%.*}.0+${_CU}.html"
# kiui pinned to 0.2.14: 0.3.x ships a broken typing.py that shadows stdlib.
# torch_geometric pinned to 2.6.1: 2.8+ requires pyg-lib>=0.6.0 for fps, but
# the cu124 wheel index tops out at 0.4.0; 2.6.x falls back to torch-cluster.
"$VPIP" install transformers==4.47.0 torch_geometric==2.6.1 torchmetrics tyro \
  kiui==0.2.14 roma easydict safetensors pygltflib usd-core gdown
# eval-harness deps so a vid2sim-only install can run eval/ without the main
# env (point BASELINES_PY at VID2SIM_PY in env.local.sh in that case)
"$VPIP" install plyfile scikit-image lpips imageio "huggingface_hub[cli]"
# LGM's rasterizer (vendored with the cstdint/cstdio header fix for newer gcc)
( cd Vid2Sim/gs/submodules/diff-gaussian-rasterization && \
  "$VPIP" install --no-build-isolation . )
fi

echo "Envs ready ($TARGET). Now: bash setup_data.sh"
