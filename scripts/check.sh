#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_ROOT}/configs/train.yaml}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OPENCV_IO_ENABLE_OPENEXR=1

cd "${PROJECT_ROOT}"
python -c "import torch, diffusers, transformers, accelerate, peft, cv2, yaml; print('dependencies OK')"
python train.py --config "${CONFIG_FILE}" --check-config
python train.py --config "${CONFIG_FILE}" --check-data

