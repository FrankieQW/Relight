#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ACTION="${1:-all}"
ENV_FILE="${STAGE2_ENV_FILE:-${SCRIPT_DIR}/stage2.env}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing operator environment file: ${ENV_FILE}" >&2
  echo "Copy ${SCRIPT_DIR}/stage2.env.example to ${SCRIPT_DIR}/stage2.env and fill it first." >&2
  exit 2
fi

set -a
# shellcheck source=/dev/null
source "${ENV_FILE}"
set +a

PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT_BASE_CONFIG="${PROJECT_BASE_CONFIG:-configs/default.yaml}"
TOKENLIGHT_BASE_CONFIG="${TOKENLIGHT_BASE_CONFIG:-Lumina-T2X/lumina_next_t2i/config.yaml}"
RUNTIME_DIR="${RUNTIME_DIR:-outputs/stage2_runtime}"
if [[ "${RUNTIME_DIR}" = /* ]]; then
  RUNTIME_DIR="$(realpath -m "${RUNTIME_DIR}")"
else
  RUNTIME_DIR="$(realpath -m "${REPO_ROOT}/${RUNTIME_DIR}")"
fi
PROJECT_CONFIG="${RUNTIME_DIR}/project.yaml"
FORMAL_CONFIG="${RUNTIME_DIR}/tokenlight_stage2.yaml"
SMOKE_PHASE1_CONFIG="${RUNTIME_DIR}/tokenlight_smoke_phase1.yaml"
SMOKE_PHASE2_CONFIG="${RUNTIME_DIR}/tokenlight_smoke_phase2.yaml"
EVAL_CONFIG="${RUNTIME_DIR}/tokenlight_evaluate.yaml"
VLLM_PID=""

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

resolve_repo_path() {
  local value="$1"
  if [[ "${value}" = /* ]]; then
    printf '%s\n' "${value}"
  else
    realpath -m "${REPO_ROOT}/${value}"
  fi
}

require_file() { [[ -f "$1" ]] || die "required file does not exist: $1"; }
require_dir() { [[ -d "$1" ]] || die "required directory does not exist: $1"; }
require_command() { command -v "$1" >/dev/null 2>&1 || die "command not found: $1"; }

cleanup() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    log "Stopping script-owned vLLM process ${VLLM_PID}"
    kill "${VLLM_PID}" || true
    wait "${VLLM_PID}" || true
  fi
}
trap cleanup EXIT

materialize_configs() {
  require_command "${PYTHON_BIN}"
  mkdir -p "${RUNTIME_DIR}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/config.py" materialize \
    --repo-root "${REPO_ROOT}" \
    --project-base "$(resolve_repo_path "${PROJECT_BASE_CONFIG}")" \
    --tokenlight-base "$(resolve_repo_path "${TOKENLIGHT_BASE_CONFIG}")" \
    --runtime-dir "${RUNTIME_DIR}"
}

require_training_inputs() {
  require_dir "${STAGE1_CHECKPOINT}"
  require_dir "${VAE_PATH}"
}

endpoint_ready() {
  "${PYTHON_BIN}" - "${VLLM_BASE_URL%/}/models" <<'PY'
import sys
import urllib.request
try:
    with urllib.request.urlopen(sys.argv[1], timeout=3) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
}

ensure_vllm() {
  if endpoint_ready; then
    return
  fi
  if [[ "${START_VLLM,,}" != "true" ]]; then
    die "vLLM endpoint is not reachable at ${VLLM_BASE_URL}; start it or set START_VLLM=true"
  fi
  require_command "${VLLM_BIN}"
  log "Starting vLLM ${VLLM_MODEL} on GPU(s) ${VLLM_GPU_IDS}"
  CUDA_VISIBLE_DEVICES="${VLLM_GPU_IDS}" "${VLLM_BIN}" serve "${VLLM_MODEL}" \
    --host 127.0.0.1 --port 8000 >"${RUNTIME_DIR}/vllm.log" 2>&1 &
  VLLM_PID=$!
  local deadline=$((SECONDS + VLLM_WAIT_SECONDS))
  until endpoint_ready; do
    kill -0 "${VLLM_PID}" 2>/dev/null || die "vLLM exited; see ${RUNTIME_DIR}/vllm.log"
    (( SECONDS < deadline )) || die "timed out waiting for vLLM; see ${RUNTIME_DIR}/vllm.log"
    sleep 5
  done
}

run_data() {
  require_dir "${OBJECT_ROOT}"
  [[ -n "${HDRI_ROOT:-}" ]] || die "HDRI_ROOT is required for composition ambient renders"
  require_dir "${HDRI_ROOT}"
  require_file "${BLENDER_BIN}"
  materialize_configs
  cd "${REPO_ROOT}"
  local annotation_path="${REPO_ROOT}/data/annotation_construction.json"
  case "${REUSE_ANNOTATION,,}" in
    true)
      log "M1-M3: verifying and reusing frozen object, scene, and annotation artifacts"
      "${PYTHON_BIN}" "${SCRIPT_DIR}/config.py" verify-reused-annotation \
        --project-config "${PROJECT_CONFIG}"
      ;;
    false)
      log "M1: preparing object inventory"
      "${PYTHON_BIN}" -m datacompose.cli prepare-objects \
        --config "${PROJECT_CONFIG}" --inventory-mode markdown

      log "M2: preparing normalized scene blends"
      "${PYTHON_BIN}" -m datacompose.cli prepare-scenes --config "${PROJECT_CONFIG}" \
        --blender-bin "${BLENDER_BIN}" --workers 1 --resume

      ensure_vllm
      log "M3: building annotation_construction.json"
      "${PYTHON_BIN}" -m datacompose.cli annotate-construction \
        --config "${PROJECT_CONFIG}" --base-url "${VLLM_BASE_URL}" \
        --model "${VLLM_MODEL}" --concurrency "${VLLM_CONCURRENCY}"
      cleanup
      VLLM_PID=""
      ;;
    *)
      die "REUSE_ANNOTATION must be true or false, got: ${REUSE_ANNOTATION}"
      ;;
  esac

  log "M4: preparing geometry"
  "${PYTHON_BIN}" -m datacompose.cli prepare-geometry --config "${PROJECT_CONFIG}"

  log "M4: building deterministic composition jobs"
  "${PYTHON_BIN}" -m datacompose.cli build-render-jobs --config "${PROJECT_CONFIG}" \
    --annotations "${annotation_path}"

  log "M4: rendering composition components"
  OBJECT_ROOT="${OBJECT_ROOT}" "${PYTHON_BIN}" -m datacompose.cli render \
    --config "${PROJECT_CONFIG}" --blender-bin "${BLENDER_BIN}" --workers "${RENDER_WORKERS}"

  log "Building manifests and running strict data gates"
  cd "${REPO_ROOT}/Lumina-T2X"
  "${PYTHON_BIN}" "${REPO_ROOT}/datacompose/tokenlight/build_manifests.py" --config "${FORMAL_CONFIG}"
  "${PYTHON_BIN}" "${REPO_ROOT}/datacompose/tokenlight/validate_components.py" --config "${FORMAL_CONFIG}"
  "${PYTHON_BIN}" tools/tokenlight_data/inspect_dataset.py --config "${FORMAL_CONFIG}"
}

run_smoke() {
  require_training_inputs
  materialize_configs
  require_file "$(resolve_repo_path "${DATASET_ROOT}")/manifests/train.jsonl"
  require_file "$(resolve_repo_path "${DATASET_ROOT}")/manifests/validation.jsonl"
  log "Preparing deterministic fixed-manifest smoke configuration"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/config.py" prepare-smoke \
    --formal-config "${FORMAL_CONFIG}" --runtime-dir "${RUNTIME_DIR}" \
    --max-rows "${SMOKE_MAX_MANIFEST_ROWS}" \
    --max-schedule-samples "${SMOKE_MAX_SCHEDULE_SAMPLES}"

  local smoke_output="${RUNTIME_DIR}/smoke_outputs/${SMOKE_RUN_ID}"
  [[ ! -e "${smoke_output}" ]] || die "smoke run already exists: ${smoke_output}; use a new SMOKE_RUN_ID"
  cd "${REPO_ROOT}/Lumina-T2X"
  log "Training smoke phase 1: forward/backward/update/checkpoint"
  CUDA_VISIBLE_DEVICES="${SMOKE_GPU_ID}" "${PYTHON_BIN}" lumina_next_t2i/train_tokenlight.py \
    --config "${SMOKE_PHASE1_CONFIG}"

  local summary="${smoke_output}/smoke_summary.json"
  require_file "${summary}"
  local checkpoint
  checkpoint="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["checkpoint"])' "${summary}")"
  require_dir "${checkpoint}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/config.py" smoke-resume \
    --phase-one-config "${SMOKE_PHASE1_CONFIG}" --checkpoint "${checkpoint}" \
    --output "${SMOKE_PHASE2_CONFIG}"

  log "Training smoke phase 2: exact resume and next optimizer update"
  CUDA_VISIBLE_DEVICES="${SMOKE_GPU_ID}" "${PYTHON_BIN}" lumina_next_t2i/train_tokenlight.py \
    --config "${SMOKE_PHASE2_CONFIG}"
  "${PYTHON_BIN}" - "${summary}" <<'PY'
import json
import sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
if summary.get("status") != "pass" or summary.get("train_ready") is not True or summary.get("resume_verified") is not True:
    raise SystemExit(f"smoke gate did not pass: {summary}")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
}

run_tests() {
  materialize_configs
  require_file "${BLENDER_BIN}"
  cd "${REPO_ROOT}"
  log "Running LightConstruction repository checks"
  "${PYTHON_BIN}" -m unittest discover -s tests -p 'test_*.py'
  log "Running focused Blender camera and fixture checks"
  "${BLENDER_BIN}" --background --factory-startup \
    --python "${REPO_ROOT}/tests/blender_composition_focus.py"
  log "Running Lumina-T2X TokenLight repository checks"
  "${PYTHON_BIN}" -m unittest discover -s Lumina-T2X/tests -p 'test_*.py'
}

run_train() {
  require_training_inputs
  materialize_configs
  if [[ "${FORMAL_RESUME_CHECKPOINT:-}" == "" ]]; then
    local run_dir
    run_dir="$(resolve_repo_path "${TOKENLIGHT_OUTPUT_ROOT}")/${FORMAL_RUN_ID}"
    [[ ! -e "${run_dir}" ]] || die "formal run already exists: ${run_dir}; set FORMAL_RESUME_CHECKPOINT or choose a new FORMAL_RUN_ID"
  else
    require_dir "${FORMAL_RESUME_CHECKPOINT}"
  fi
  local smoke_summary="${RUNTIME_DIR}/smoke_outputs/${SMOKE_RUN_ID}/smoke_summary.json"
  require_file "${smoke_summary}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/config.py" verify-smoke \
    --formal-config "${FORMAL_CONFIG}" \
    --fixture-metadata "${RUNTIME_DIR}/smoke_dataset/source.json" \
    --summary "${smoke_summary}"
  IFS=',' read -r -a train_gpu_array <<< "${TRAIN_GPU_IDS}"
  (( ${#train_gpu_array[@]} == 8 )) || die "formal Stage-2 training requires exactly 8 TRAIN_GPU_IDS entries"
  cd "${REPO_ROOT}/Lumina-T2X"
  log "Starting Stage-2 formal DDP training on physical GPU(s): ${TRAIN_GPU_IDS}"
  CUDA_VISIBLE_DEVICES="${TRAIN_GPU_IDS}" "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone --nproc_per_node=8 \
    lumina_next_t2i/train_tokenlight.py --config "${FORMAL_CONFIG}"
}

latest_formal_checkpoint() {
  local checkpoint_root
  checkpoint_root="$(resolve_repo_path "${TOKENLIGHT_OUTPUT_ROOT}")/${FORMAL_RUN_ID}/checkpoints"
  [[ -d "${checkpoint_root}" ]] || return 1
  find "${checkpoint_root}" -mindepth 1 -maxdepth 1 -type d -name 'step_*' -print | sort | tail -n 1
}

run_eval() {
  require_training_inputs
  materialize_configs
  local checkpoint="${EVAL_CHECKPOINT:-}"
  if [[ -z "${checkpoint}" ]]; then
    checkpoint="$(latest_formal_checkpoint)" || die "no formal checkpoint found; set EVAL_CHECKPOINT"
  fi
  [[ -n "${checkpoint}" ]] || die "no formal checkpoint found; set EVAL_CHECKPOINT"
  require_dir "${checkpoint}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/config.py" evaluation \
    --formal-config "${FORMAL_CONFIG}" --checkpoint "${checkpoint}" --output "${EVAL_CONFIG}"
  cd "${REPO_ROOT}/Lumina-T2X"
  log "Evaluating ${checkpoint} on GPU ${EVAL_GPU_ID}"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU_ID}" "${PYTHON_BIN}" lumina_next_t2i/evaluate_tokenlight.py \
    --config "${EVAL_CONFIG}"
}

case "${ACTION}" in
  data) run_data ;;
  tests) run_tests ;;
  smoke) run_smoke ;;
  train) run_train ;;
  eval) run_eval ;;
  all)
    run_data
    run_tests
    run_smoke
    run_train
    run_eval
    ;;
  *) die "usage: $0 {data|tests|smoke|train|eval|all}" ;;
esac

log "Stage-2 action '${ACTION}' completed successfully"
