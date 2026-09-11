#!/bin/bash
# Central machine config — the ONLY file you should need to touch on a new
# machine. Sourced by run_queue.sh (and usable interactively:
# `source env.sh` before running any baseline by hand).

# Machine-local overrides (not committed): put your BASELINES_PY / CUDA_HOME
# exports in env.local.sh next to this file.
[ -f "$(dirname "${BASH_SOURCE[0]}")/env.local.sh" ] && source "$(dirname "${BASH_SOURCE[0]}")/env.local.sh"

# Python interpreter that has the baseline deps (see env/setup_env.sh to
# build it as a conda env named "baselines").
if [ -z "${BASELINES_PY:-}" ] && command -v conda >/dev/null; then
  _cand="$(conda info --base 2>/dev/null)/envs/baselines/bin/python"
  [ -x "$_cand" ] && BASELINES_PY=$_cand
fi
export BASELINES_PY=${BASELINES_PY:-$(command -v python)}

# CUDA toolkit with nvcc, needed once for torch JIT builds (PAC-NeRF).
# Any CUDA 12.x works with the cu12x torch builds.
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH

# taichi preallocation cap in GB. The upstream repos preallocate a FRACTION
# of total GPU memory, which OOMs on large shared GPUs — 16G is plenty for
# every scene in the benchmarks.
export TI_DEVICE_MEMORY_GB=${TI_DEVICE_MEMORY_GB:-16}

# Optional: pin TORCH_CUDA_ARCH_LIST to your GPU's compute capability in
# env.local.sh to speed up one-time JIT builds (e.g. 8.0 for A100, 9.0 for
# H100/H200). Left unset, torch detects the visible GPU, which is correct
# on any homogeneous machine.
# MASIV runs in its own conda env (newer torch/taichi than the main stack).
if [ -z "${MASIV_PY:-}" ] && command -v conda >/dev/null; then
  _cand="$(conda info --base 2>/dev/null)/envs/masiv/bin/python"
  [ -x "$_cand" ] && MASIV_PY=$_cand
fi
export MASIV_PY=${MASIV_PY:-}

# Vid2Sim runs in its own venv layered over the masiv env (see
# env/setup_env.sh). Override VID2SIM_PY in env.local.sh if the venv lives
# elsewhere (e.g. outside the conda envs dir on scratch storage).
if [ -z "${VID2SIM_PY:-}" ] && command -v conda >/dev/null; then
  _cand="$(conda info --base 2>/dev/null)/envs/vid2sim/bin/python"
  [ -x "$_cand" ] && VID2SIM_PY=$_cand
fi
export VID2SIM_PY=${VID2SIM_PY:-}

# Keep CPU thread pools from oversubscribing when several workers share a node.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

export TI_OFFLINE_CACHE=1
export TI_OFFLINE_CACHE_FILE_PATH=${TI_OFFLINE_CACHE_FILE_PATH:-$(dirname "${BASH_SOURCE[0]}")/.ti_cache}
