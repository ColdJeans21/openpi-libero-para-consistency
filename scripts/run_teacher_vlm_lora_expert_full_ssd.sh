#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/gml/xz/openpi}"
CONFIG_NAME="pi0_libero_goal_teacher_vlm_lora_expert_full"
FULL_RUN_NAME="teacher_consistency_lam1_vlm_lora_expert_full_30k"
SSD_CHECKPOINT_BASE="${SSD_CHECKPOINT_BASE:-/data/openpi_checkpoints}"
SSD_CONFIG_ROOT="${SSD_CHECKPOINT_BASE}/${CONFIG_NAME}"
REPO_CONFIG_ROOT="${ROOT}/checkpoints/${CONFIG_NAME}"
LOG_ROOT="${ROOT}/logs/teacher_vlm_lora_expert_full_ssd"
RUN_MANIFEST="${ROOT}/examples/libero_para/runs/teacher_vlm_lora_expert_full_ssd.md"
MAX_ALLOWED_PEAK_MIB="${MAX_ALLOWED_PEAK_MIB:-23500}"
MIN_FREE_GIB="${MIN_FREE_GIB:-150}"
TRAIN_WANDB_MODE="${TRAIN_WANDB_MODE:-online}"
MONITOR_PID=""

timestamp() { date --iso-8601=seconds; }
log() { echo "[$(timestamp)] $*"; }

require_new_ssd() {
  local mountpoint available_kib
  mountpoint=$(findmnt -n -o TARGET --target "${SSD_CHECKPOINT_BASE}")
  [[ "${mountpoint}" == "/data" ]] || { echo "Expected /data, found ${mountpoint}" >&2; exit 1; }
  available_kib=$(df --output=avail "${SSD_CHECKPOINT_BASE}" | tail -n 1 | tr -d ' ')
  (( available_kib >= MIN_FREE_GIB * 1024 * 1024 )) || {
    echo "New SSD has less than ${MIN_FREE_GIB} GiB free" >&2
    exit 1
  }
}

prepare_run() {
  local run_name="$1" target link
  target="${SSD_CONFIG_ROOT}/${run_name}"
  link="${REPO_CONFIG_ROOT}/${run_name}"
  [[ ! -e "${target}" ]] || { echo "Refusing to overwrite ${target}" >&2; exit 1; }
  if [[ -L "${link}" ]]; then
    [[ "$(readlink "${link}")" == "${target}" ]] || { echo "Wrong symlink: ${link}" >&2; exit 1; }
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
  local batch="$1" run_name target smoke_log memory_log status peak
  run_name="teacher_consistency_lam1_vlm_lora_expert_full_smoke_b${batch}"
  target="${SSD_CONFIG_ROOT}/${run_name}"
  smoke_log="${LOG_ROOT}/train_${run_name}.log"
  memory_log="${LOG_ROOT}/${run_name}_gpu_memory.csv"
  prepare_run "${run_name}"

  log "starting 20-step smoke test at global batch ${batch}"
  echo "gpu_index,memory_used_mib" > "${memory_log}"
  (while true; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits >> "${memory_log}"; sleep 2; done) &
  MONITOR_PID=$!

  set +e
  env CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
    uv run "${ROOT}/scripts/train.py" "${CONFIG_NAME}" \
      --checkpoint-base-dir "${SSD_CHECKPOINT_BASE}" \
      --exp-name "${run_name}" \
      --batch-size "${batch}" \
      --num-train-steps 20 \
      --save-interval 20 \
      --keep-period 20 \
      --no-wandb-enabled \
      2>&1 | tee "${smoke_log}"
  status=${PIPESTATUS[0]}
  set -e
  stop_monitor

  peak=$(awk -F, 'NR > 1 {v=$2+0; if (v>p) p=v} END {print p+0}' "${memory_log}")
  mkdir -p "${target}"
  cp "${memory_log}" "${target}/GPU_MEMORY_SMOKE.csv"
  cp "${RUN_MANIFEST}" "${target}/EXPERIMENT.md"
  printf 'batch_size=%s\nexit_status=%s\npeak_memory_mib=%s\nthreshold_mib=%s\n' \
    "${batch}" "${status}" "${peak}" "${MAX_ALLOWED_PEAK_MIB}" > "${target}/SMOKE_RESULT.txt"

  if (( status == 0 && peak <= MAX_ALLOWED_PEAK_MIB )) && [[ -f "${target}/19/_CHECKPOINT_METADATA" ]]; then
    log "batch ${batch} smoke passed at ${peak} MiB"
    return 0
  fi
  log "batch ${batch} smoke failed safety gate: status=${status}, peak=${peak} MiB"
  return 1
}

run_training() {
  local batch="$1" target train_log
  target="${SSD_CONFIG_ROOT}/${FULL_RUN_NAME}"
  train_log="${LOG_ROOT}/train_${FULL_RUN_NAME}.log"
  prepare_run "${FULL_RUN_NAME}"
  log "starting full training with global batch ${batch}"
  env CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE="${TRAIN_WANDB_MODE}" \
    uv run "${ROOT}/scripts/train.py" "${CONFIG_NAME}" \
      --checkpoint-base-dir "${SSD_CHECKPOINT_BASE}" \
      --exp-name "${FULL_RUN_NAME}" \
      --batch-size "${batch}" \
      --keep-period 10000 \
      2>&1 | tee "${train_log}"
  test -f "${target}/29999/_CHECKPOINT_METADATA"
  cp "${RUN_MANIFEST}" "${target}/EXPERIMENT.md"
  printf 'selected_global_batch=%s\n' "${batch}" > "${target}/RUN_SETTINGS.txt"
  log "full training complete"
}

run_validation() {
  log "starting matched Goal/Para validation"
  env RUN_NAME="${FULL_RUN_NAME}" \
    CHECKPOINT_ROOT="${REPO_CONFIG_ROOT}/${FULL_RUN_NAME}" \
    CONFIG_NAME="${CONFIG_NAME}" \
    STEPS_STRING="10000 20000 29999" \
    SERVER_GPU=1 CLIENT_GPU=0 PORT=8000 SEED=7 GOAL_TRIALS=10 \
    WANDB_ENTITY=xzzsryouxiwang-shenzhen-university \
    bash "${ROOT}/scripts/run_libero_goal_para_validation.sh"

  env RUN_SPECS_OVERRIDE="${FULL_RUN_NAME}|${CONFIG_NAME}|${REPO_CONFIG_ROOT}/${FULL_RUN_NAME}/29999" \
    bash "${ROOT}/scripts/run_libero_plus_language_eval.sh"
}

trap stop_monitor EXIT INT TERM
cd "${ROOT}"
mkdir -p "${SSD_CONFIG_ROOT}" "${REPO_CONFIG_ROOT}" "${LOG_ROOT}"
require_new_ssd

SELECTED_BATCH=""
if run_smoke 4; then
  SELECTED_BATCH=4
elif run_smoke 2; then
  SELECTED_BATCH=2
else
  log "both smoke tests failed; aborting before full training"
  exit 1
fi

run_training "${SELECTED_BATCH}"
run_validation
log "VLM LoRA + full Action Expert experiment complete"
