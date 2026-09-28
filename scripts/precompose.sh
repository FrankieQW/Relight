#!/usr/bin/env bash
set -Eeuo pipefail

export HOME="${HOME:-/home/frankie}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/frankie/programs/Relight}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_ROOT}/configs/train.yaml}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OPENCV_IO_ENABLE_OPENEXR=1
cd "${PROJECT_ROOT}"
exec python precompose.py --config "${CONFIG_FILE}" "$@"
