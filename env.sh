#!/bin/bash
# Central machine config — the ONLY file you should need to touch on a new
# machine. Sourced by run_queue.sh (and usable interactively:
# `source env.sh` before running any baseline by hand).

# Machine-local overrides (not committed): put your BASELINES_PY / CUDA_HOME
# exports in env.local.sh next to this file.
[ -f "$(dirname "${BASH_SOURCE[0]}")/env.local.sh" ] && source "$(dirname "${BASH_SOURCE[0]}")/env.local.sh"

# Python interpreter that has the baseline deps (see env/setup_env.sh to
# build it as a conda env named "gic-baselines").
if [ -z "${BASELINES_PY:-}" ] && command -v conda >/dev/null \
   && conda env list 2>/dev/null | grep -q "gic-baselines"; then
  BASELINES_PY=$(conda env list | awk '/gic-baselines/{print $NF}')/bin/python
fi
export BASELINES_PY=${BASELINES_PY:-$(command -v python)}

# CUDA toolkit with nvcc, needed once for torch JIT builds (PAC-NeRF).
# Any CUDA 12.x works with the cu12x torch builds.
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH

# taichi preallocation cap in GB. The upstream repos preallocate a FRACTION
# of total GPU memory, which OOMs on large shared GPUs — 20G is plenty for
# every scene in the benchmarks.
export TI_DEVICE_MEMORY_GB=${TI_DEVICE_MEMORY_GB:-20}

# One-time JIT/compile speedups. Set TORCH_CUDA_ARCH_LIST to your GPU's
# compute capability (9.0=H100/H200, 8.9=RTX4090, 8.0=A100).
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
export TI_OFFLINE_CACHE=1
export TI_OFFLINE_CACHE_FILE_PATH=${TI_OFFLINE_CACHE_FILE_PATH:-$(dirname "${BASH_SOURCE[0]}")/.ti_cache}
