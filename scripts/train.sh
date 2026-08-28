#!/usr/bin/env bash
set -Eeuo pipefail

export HOME="${HOME:-/mnt/afs_fangwenqi}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs_fangwenqi/new_method}"
CONDA_ROOT="${CONDA_ROOT:-/mnt/afs_fangwenqi/miniconda3}"
CONDA_ENV="${CONDA_ENV:-${CONDA_ROOT}/envs/flux-kontext}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_ROOT}/configs/train.yaml}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MASTER_PORT="${MASTER_PORT:-29621}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
set +u
conda activate "${CONDA_ENV}"
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

cd "${PROJECT_ROOT}"
EXPECTED_GPUS="${NUM_PROCESSES}" python - <<'PY'
import os
import torch

expected = int(os.environ["EXPECTED_GPUS"])
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
actual = torch.cuda.device_count()
if actual != expected:
    raise RuntimeError(f"expected {expected} visible GPUs, got {actual}")
print(f"visible GPUs: {actual}")
for index in range(actual):
    print(f"gpu[{index}]: {torch.cuda.get_device_name(index)}")
PY

python train.py --config "${CONFIG_FILE}" --check-config
exec accelerate launch \
  --num_machines 1 \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MASTER_PORT}" \
  --mixed_precision bf16 \
  train.py --config "${CONFIG_FILE}" "$@"

