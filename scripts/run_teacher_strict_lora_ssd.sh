#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/gml/xz/openpi}"
CONFIG_NAME="pi0_libero_goal_teacher_strict_lora"
SMOKE_RUN_NAME="teacher_consistency_lam1_strict_lora_smoke20"
FULL_RUN_NAME="teacher_consistency_lam1_strict_lora_30k"
SSD_CHECKPOINT_BASE="${SSD_CHECKPOINT_BASE:-/data/openpi_checkpoints}"
SSD_CONFIG_ROOT="${SSD_CHECKPOINT_BASE}/${CONFIG_NAME}"
REPO_CONFIG_ROOT="${ROOT}/checkpoints/${CONFIG_NAME}"
LOG_ROOT="${ROOT}/logs/teacher_strict_lora_ssd"
RUN_MANIFEST="${ROOT}/examples/libero_para/runs/teacher_strict_lora_ssd.md"
MAX_ALLOWED_PEAK_MIB="${MAX_ALLOWED_PEAK_MIB:-23500}"
MIN_FREE_GIB="${MIN_FREE_GIB:-100}"
MONITOR_PID=""

timestamp() { date --iso-8601=seconds; }
log() { echo "[$(timestamp)] $*"; }

prepare_run() {
  local run_name="$1" target link
  target="${SSD_CONFIG_ROOT}/${run_name}"
  link="${REPO_CONFIG_ROOT}/${run_name}"
  [[ ! -e "${target}" ]] || { echo "Refusing to overwrite ${target}" >&2; exit 1; }
  if [[ -L "${link}" ]]; then
    [[ "$(readlink "${link}")" == "${target}" ]] || { echo "Wrong symlink ${link}" >&2; exit 1; }
  elif [[ -e "${link}" ]]; then
    echo "Refusing to replace ${link}" >&2
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
  local target log_path memory_log status peak
  target="${SSD_CONFIG_ROOT}/${SMOKE_RUN_NAME}"
  log_path="${LOG_ROOT}/train_${SMOKE_RUN_NAME}.log"
  memory_log="${LOG_ROOT}/${SMOKE_RUN_NAME}_gpu_memory.csv"
  prepare_run "${SMOKE_RUN_NAME}"
  echo "gpu_index,memory_used_mib" > "${memory_log}"
  (while true; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits >> "${memory_log}"; sleep 2; done) &
  MONITOR_PID=$!

  set +e
  env CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
    uv run "${ROOT}/scripts/train.py" "${CONFIG_NAME}" \
      --checkpoint-base-dir "${SSD_CHECKPOINT_BASE}" \
      --exp-name "${SMOKE_RUN_NAME}" \
      --num-train-steps 20 --save-interval 20 --keep-period 20 --no-wandb-enabled \
      2>&1 | tee "${log_path}"
  status=${PIPESTATUS[0]}
  set -e
  stop_monitor

  peak=$(awk -F, 'NR > 1 {v=$2+0; if (v>p) p=v} END {print p+0}' "${memory_log}")
  mkdir -p "${target}"
  cp "${memory_log}" "${target}/GPU_MEMORY_SMOKE.csv"
  cp "${RUN_MANIFEST}" "${target}/EXPERIMENT.md"
  printf 'exit_status=%s\npeak_memory_mib=%s\nthreshold_mib=%s\n' \
    "${status}" "${peak}" "${MAX_ALLOWED_PEAK_MIB}" > "${target}/SMOKE_RESULT.txt"
  (( status == 0 && peak <= MAX_ALLOWED_PEAK_MIB ))
  test -f "${target}/19/_CHECKPOINT_METADATA"
  log "smoke passed at ${peak} MiB"
}

run_training() {
  local target log_path
  target="${SSD_CONFIG_ROOT}/${FULL_RUN_NAME}"
  log_path="${LOG_ROOT}/train_${FULL_RUN_NAME}.log"
  prepare_run "${FULL_RUN_NAME}"
  env CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=online \
    uv run "${ROOT}/scripts/train.py" "${CONFIG_NAME}" \
      --checkpoint-base-dir "${SSD_CHECKPOINT_BASE}" \
      --exp-name "${FULL_RUN_NAME}" --keep-period 10000 \
      2>&1 | tee "${log_path}"
  test -f "${target}/29999/_CHECKPOINT_METADATA"
  cp "${RUN_MANIFEST}" "${target}/EXPERIMENT.md"
  log "full training complete"
}

run_validation() {
  env RUN_NAME="${FULL_RUN_NAME}" \
    CHECKPOINT_ROOT="${REPO_CONFIG_ROOT}/${FULL_RUN_NAME}" \
    CONFIG_NAME="${CONFIG_NAME}" STEPS_STRING="10000 20000 29999" \
    SERVER_GPU=1 CLIENT_GPU=0 PORT=8000 SEED=7 GOAL_TRIALS=10 \
    WANDB_ENTITY=xzzsryouxiwang-shenzhen-university \
    bash "${ROOT}/scripts/run_libero_goal_para_validation.sh"
  env RUN_SPECS_OVERRIDE="${FULL_RUN_NAME}|${CONFIG_NAME}|${REPO_CONFIG_ROOT}/${FULL_RUN_NAME}/29999" \
    bash "${ROOT}/scripts/run_libero_plus_language_eval.sh"
}

trap stop_monitor EXIT INT TERM
cd "${ROOT}"
mkdir -p "${SSD_CONFIG_ROOT}" "${REPO_CONFIG_ROOT}" "${LOG_ROOT}"
[[ "$(findmnt -n -o TARGET --target "${SSD_CHECKPOINT_BASE}")" == "/data" ]]
available_kib=$(df --output=avail "${SSD_CHECKPOINT_BASE}" | tail -n 1 | tr -d ' ')
(( available_kib >= MIN_FREE_GIB * 1024 * 1024 ))

log "starting strict-LoRA smoke test"
run_smoke
log "starting strict-LoRA 30k training"
run_training
log "starting strict-LoRA validation"
run_validation
log "strict-LoRA experiment complete"
