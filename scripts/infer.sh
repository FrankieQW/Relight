#!/usr/bin/env bash
set -Eeuo pipefail

export HOME="${HOME:-/home/frankie}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/frankie/programs/Relight}"
# CONDA_ROOT="${CONDA_ROOT:-/mnt/afs_fangwenqi/miniconda3}"
# CONDA_ENV="${CONDA_ENV:-${CONDA_ROOT}/envs/flux-kontext}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_ROOT}/configs/train.yaml}"

# source "${CONDA_ROOT}/etc/profile.d/conda.sh"
# set +u
# conda activate "${CONDA_ENV}"
# set -u

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OPENCV_IO_ENABLE_OPENEXR=1
cd "${PROJECT_ROOT}"
exec python infer.py --config "${CONFIG_FILE}" "$@"

