#!/bin/bash
# Bootstrap the "gic-baselines" conda env on a fresh machine.
# Covers PAC-NeRF, GIC and Spring-Gaus (they share one stack), and creates
# the NeuMA venv on top of it. Requires: conda, a CUDA 12.x toolkit (nvcc),
# and a GPU driver >= the toolkit version.
#
#   bash env/setup_env.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

# --- main env: python 3.9 + torch cu121 + taichi 1.2 stack (GIC's spec) ---
conda env create -n gic-baselines -f GIC/environment.yml || true
PY=$(conda env list | awk '/gic-baselines/{print $NF}')/bin/python
$PY -m pip install ninja yacs termcolor gitpython

# Spring-Gaus rasterizer/simple-knn come from GIC's environment.yml already
# (diff-gaussian-rasterization + simple-knn). Its rasterizer returns 3+
# values; the vendored Spring-Gaus code is already patched for that.

# --- NeuMA venv (needs old warp-lang 0.6.1, incompatible with main env) ---
$PY -m venv --system-site-packages NeuMA/.venv
NeuMA/.venv/bin/pip install warp-lang==0.6.1 e3nn==0.5.1 viser==0.2.3 \
  nerfview==0.0.3 pyvista==0.44.0 splines==0.3.2 natsort torchmetrics \
  tensorboardX mediapy omegaconf==2.3.0 py7zr
( cd NeuMA/extern/diff-gaussian-rasterization && ../../.venv/bin/pip install . )

echo "Envs ready. Now: source env.sh && bash setup_data.sh"
