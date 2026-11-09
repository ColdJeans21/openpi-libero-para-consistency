#!/usr/bin/env bash
set -euo pipefail

# Smoke-test and then run the full Action Expert parameter-scope ablation.

ROOT="${ROOT:-/home/gml/xz/openpi}"
CONFIG_NAME="pi0_libero_goal_teacher_expert_full"
SMOKE_RUN_NAME="teacher_consistency_lam1_expert_full_smoke20"
FULL_RUN_NAME="teacher_consistency_lam1_expert_full_30k"
SSD_CHECKPOINT_BASE="${SSD_CHECKPOINT_BASE:-/data/openpi_checkpoints}"
SSD_CONFIG_ROOT="${SSD_CHECKPOINT_BASE}/${CONFIG_NAME}"
REPO_CONFIG_ROOT="${ROOT}/checkpoints/${CONFIG_NAME}"
LOG_ROOT="${ROOT}/logs/teacher_expert_full_ssd"
ORCHESTRATOR_LOG="${ROOT}/logs/teacher_expert_full_ssd_orchestrator.log"
RUN_MANIFEST="${ROOT}/examples/libero_para/runs/teacher_expert_full_ssd.md"
MAX_ALLOWED_PEAK_MIB="${MAX_ALLOWED_PEAK_MIB:-23500}"
MIN_FREE_GIB="${MIN_FREE_GIB:-200}"
# Default future runs to live W&B logging now that credentials are available.
# The currently running experiment was started offline and is synced by its
# completion watcher instead.
TRAIN_WANDB_MODE="${TRAIN_WANDB_MODE:-online}"
MONITOR_PID=""

timestamp() {
  date --iso-8601=seconds
}

log() {
  echo "[$(timestamp)] $*"
}

require_new_ssd() {
  local mountpoint available_kib
  mountpoint=$(findmnt -n -o TARGET --target "${SSD_CHECKPOINT_BASE}")
  if [[ "${mountpoint}" != "/data" ]]; then
    echo "Refusing to train: ${SSD_CHECKPOINT_BASE} is mounted on ${mountpoint}, expected /data" >&2
    exit 1
  fi
  available_kib=$(df --output=avail "${SSD_CHECKPOINT_BASE}" | tail -n 1 | tr -d ' ')
  if (( available_kib < MIN_FREE_GIB * 1024 * 1024 )); then
    echo "Refusing to train: new SSD has less than ${MIN_FREE_GIB} GiB free" >&2
    exit 1
  fi
}

prepare_run() {
  local run_name="$1"
  local target="${SSD_CONFIG_ROOT}/${run_name}"
  local link="${REPO_CONFIG_ROOT}/${run_name}"

  if [[ -e "${target}" ]]; then
    echo "Refusing to overwrite existing SSD experiment: ${target}" >&2
    exit 1
  fi
  if [[ -L "${link}" ]]; then
    if [[ "$(readlink "${link}")" != "${target}" ]]; then
      echo "Existing link points to the wrong target: ${link}" >&2
      exit 1
    fi
  elif [[ -e "${link}" ]]; then
    echo "Refusing to replace existing repository checkpoint path: ${link}" >&2
    exit 1
  else
    ln -s "${target}" "${link}"
  fi
}

stop_monitor() {
  if [[ -n "${MONITOR_PID}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
    kill -TERM "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
  MONITOR_PID=""
}

run_smoke() {
  local target="${SSD_CONFIG_ROOT}/${SMOKE_RUN_NAME}"
  local smoke_log="${LOG_ROOT}/train_${SMOKE_RUN_NAME}.log"
  local memory_log="${LOG_ROOT}/${SMOKE_RUN_NAME}_gpu_memory.csv"
  local train_status peak_mib

  log "starting 20-step batch-4 smoke test"
  echo "gpu_index,memory_used_mib" > "${memory_log}"
  (
    while true; do
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits >> "${memory_log}"
      sleep 2
    done
  ) &
  MONITOR_PID=$!

  set +e
  env \
    CUDA_VISIBLE_DEVICES=0,1 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    WANDB_MODE=disabled \
    uv run "${ROOT}/scripts/train.py" "${CONFIG_NAME}" \
      --checkpoint-base-dir "${SSD_CHECKPOINT_BASE}" \
      --exp-name "${SMOKE_RUN_NAME}" \
      --num-train-steps 20 \
      --save-interval 20 \
      --keep-period 20 \
      --no-wandb-enabled \
      2>&1 | tee "${smoke_log}"
  train_status=${PIPESTATUS[0]}
  set -e
  stop_monitor

  if (( train_status != 0 )); then
    log "smoke test failed with status ${train_status}; full run will not start"
    exit "${train_status}"
  fi
  test -f "${target}/19/_CHECKPOINT_METADATA"

  # Force numeric comparison: gsub() marks the field as a string in some awk
  # implementations, which would otherwise compare values lexicographically.
  peak_mib=$(awk -F, 'NR > 1 {value=$2+0; if (value > peak) peak=value} END {print peak+0}' "${memory_log}")
  cp "${memory_log}" "${target}/GPU_MEMORY_SMOKE.csv"
  cp "${RUN_MANIFEST}" "${target}/EXPERIMENT.md"
  printf 'peak_memory_mib=%s\nthreshold_mib=%s\n' "${peak_mib}" "${MAX_ALLOWED_PEAK_MIB}" \
    > "${target}/SMOKE_RESULT.txt"
  log "smoke test passed; sampled peak=${peak_mib} MiB, threshold=${MAX_ALLOWED_PEAK_MIB} MiB"

  if (( peak_mib > MAX_ALLOWED_PEAK_MIB )); then
    log "smoke test exceeded memory safety threshold; full run will not start"
    exit 1
  fi
}

run_full_training() {
  local target="${SSD_CONFIG_ROOT}/${FULL_RUN_NAME}"
  local train_log="${LOG_ROOT}/train_${FULL_RUN_NAME}.log"

  log "starting full 30k Action Expert training"
  env \
    CUDA_VISIBLE_DEVICES=0,1 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    WANDB_MODE="${TRAIN_WANDB_MODE}" \
    uv run "${ROOT}/scripts/train.py" "${CONFIG_NAME}" \
      --checkpoint-base-dir "${SSD_CHECKPOINT_BASE}" \
      --exp-name "${FULL_RUN_NAME}" \
      --keep-period 10000 \
      2>&1 | tee "${train_log}"

  test -f "${target}/29999/_CHECKPOINT_METADATA"
  cp "${RUN_MANIFEST}" "${target}/EXPERIMENT.md"
  log "full training complete"
}

run_matched_validation() {
  log "starting matched Goal/Para validation"
  env \
    RUN_NAME="${FULL_RUN_NAME}" \
    CHECKPOINT_ROOT="${REPO_CONFIG_ROOT}/${FULL_RUN_NAME}" \
    CONFIG_NAME="${CONFIG_NAME}" \
    STEPS_STRING="10000 20000 29999" \
    SERVER_GPU=1 \
    CLIENT_GPU=0 \
    PORT=8000 \
    SEED=7 \
    GOAL_TRIALS=10 \
    bash "${ROOT}/scripts/run_libero_goal_para_validation.sh"
}

run_plus_validation() {
  local run_specs
  run_specs="${FULL_RUN_NAME}|${CONFIG_NAME}|${REPO_CONFIG_ROOT}/${FULL_RUN_NAME}/29999"
  log "starting full 410-case LIBERO-Plus Language validation"
  RUN_SPECS_OVERRIDE="${run_specs}" \
    bash "${ROOT}/scripts/run_libero_plus_language_eval.sh"
}

trap stop_monitor EXIT INT TERM
cd "${ROOT}"
mkdir -p "${SSD_CONFIG_ROOT}" "${REPO_CONFIG_ROOT}" "${LOG_ROOT}"
require_new_ssd
prepare_run "${SMOKE_RUN_NAME}"
prepare_run "${FULL_RUN_NAME}"
run_smoke
run_full_training
run_matched_validation
run_plus_validation
log "Action Expert ablation training and all validations complete"
