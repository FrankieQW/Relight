#!/usr/bin/env bash
set -Eeuo pipefail

export HOME="${HOME:-/home/frankie}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/frankie/programs/Relight}"
# CONDA_ROOT="${CONDA_ROOT:-/home/frankie/miniconda3}"
# CONDA_ENV="${CONDA_ENV:-${CONDA_ROOT}/envs/flux-kontext}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_ROOT}/configs/train.yaml}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
MASTER_PORT="${MASTER_PORT:-29621}"

# source "${CONDA_ROOT}/etc/profile.d/conda.sh"
# set +u
# conda activate "${CONDA_ENV}"
# set -u

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
unset NCCL_DEBUG_SUBSYS
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
# RTX 6000D systems without GPU P2P can fail in NCCL's shareable cuMem-host
# path. This keeps the safer shared-memory fallback and remains overridable.
export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
# Disabling SHM forces the conservative socket transport. It is slower, but
# avoids the CUDA-700 failure observed on this dual RTX 6000D host. Override
# with NCCL_SHM_DISABLE=0 after the smoke test if SHM is known to work.
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_NET="${NCCL_NET:-Socket}"

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
if [[ "${SKIP_NCCL_CHECK:-0}" != "1" && "${NUM_PROCESSES}" -gt 1 ]]; then
  python -m torch.distributed.run \
    --standalone \
    --nproc_per_node "${NUM_PROCESSES}" \
    scripts/check_nccl.py
fi
exec accelerate launch \
  --multi_gpu \
  --num_machines 1 \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MASTER_PORT}" \
  --mixed_precision bf16 \
  --dynamo_backend no \
  train.py --config "${CONFIG_FILE}" "$@"
