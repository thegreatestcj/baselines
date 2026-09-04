#!/bin/bash
# Bootstrap the "baselines" conda env on a fresh machine.
# Covers PAC-NeRF, GIC and Spring-Gaus (they share one stack), and creates
# the NeuMA venv on top of it. Requires: conda, a CUDA 12.x toolkit (nvcc),
# and a GPU driver >= the toolkit version.
#
#   bash env/setup_env.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh

# The pip section of env/environment.yml compiles CUDA extensions
# (rasterizers, pytorch3d) at create time — nvcc must be reachable.
if ! command -v nvcc >/dev/null; then
  echo "ERROR: nvcc not found. Point CUDA_HOME at a CUDA 12.x toolkit" \
       "(env.local.sh) and re-run." >&2
  exit 1
fi

# --- main env: python 3.9 + torch cu121 + taichi 1.2 stack ---
PY="$(conda info --base)/envs/baselines/bin/python"
if [ ! -x "$PY" ]; then
  conda env create -n baselines -f env/environment.yml
fi
$PY -m pip install ninja yacs termcolor gitpython "huggingface_hub[cli]"

# Compiled rasterizer/simple-knn are vendored (pinned upstream + cstdint fix
# for newer gcc); installed here rather than from git+ so the patch applies.
$PY -m pip install env/third_party/diff-gaussian-rasterization env/third_party/simple-knn

# Spring-Gaus reuses the same rasterizer/simple-knn (installed above from
# env/environment.yml as pinned git+https builds); the vendored Spring-Gaus
# code is already patched for the 3-value rasterizer return.

# --- NeuMA venv (needs old warp-lang 0.6.1, incompatible with main env) ---
if [ ! -x NeuMA/.venv/bin/python ]; then
  $PY -m venv --system-site-packages NeuMA/.venv
fi
NeuMA/.venv/bin/pip install warp-lang==0.6.1 e3nn==0.5.1 viser==0.2.3 \
  nerfview==0.0.3 pyvista==0.44.0 splines==0.3.2 natsort torchmetrics \
  tensorboardX mediapy omegaconf==2.3.0 py7zr
( cd NeuMA/extern/diff-gaussian-rasterization && ../../.venv/bin/pip install . )

echo "Envs ready. Now: bash setup_data.sh"
