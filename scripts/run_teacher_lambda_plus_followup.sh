#!/usr/bin/env bash
set -euo pipefail

# Wait for the active teacher-lambda training/Goal/Para queue, then evaluate
# the final lambda=0.1 and lambda=10 checkpoints on all 410 official
# LIBERO-Plus Language instances. Lambda=1 already has a complete result.

ROOT="${ROOT:-/home/gml/xz/openpi}"
MAIN_QUEUE_PID="${MAIN_QUEUE_PID:-19107}"
POLL_SECONDS="${POLL_SECONDS:-60}"
CONFIG_NAME="pi0_libero_goal_lora_consistency"
LOG_FILE="${ROOT}/logs/teacher_lambda_plus_followup.log"

timestamp() {
  date --iso-8601=seconds
}

log() {
  echo "[$(timestamp)] $*" | tee -a "${LOG_FILE}"
}

cd "${ROOT}"
mkdir -p "${ROOT}/logs"
log "waiting for teacher lambda queue PID ${MAIN_QUEUE_PID}"

while kill -0 "${MAIN_QUEUE_PID}" 2>/dev/null; do
  sleep "${POLL_SECONDS}"
done

for run_name in teacher_consistency_lam01_30k teacher_consistency_lam10_30k; do
  checkpoint="${ROOT}/checkpoints/${CONFIG_NAME}/${run_name}/29999/_CHECKPOINT_METADATA"
  goal_result="${ROOT}/reports/validation/${run_name}/step_29999/libero_goal_seed7.json"
  para_result="${ROOT}/reports/validation/${run_name}/step_29999/libero_para_seed7/summary.json"
  if [[ ! -f "${checkpoint}" || ! -f "${goal_result}" || ! -f "${para_result}" ]]; then
    log "required final checkpoint or matched validation is missing for ${run_name}"
    exit 1
  fi
done

run_specs=$(printf '%s\n' \
  "teacher_consistency_lam01_30k|${CONFIG_NAME}|${ROOT}/checkpoints/${CONFIG_NAME}/teacher_consistency_lam01_30k/29999" \
  "teacher_consistency_lam10_30k|${CONFIG_NAME}|${ROOT}/checkpoints/${CONFIG_NAME}/teacher_consistency_lam10_30k/29999")

log "starting full LIBERO-Plus Language evaluation for lambda=0.1 and lambda=10"
RUN_SPECS_OVERRIDE="${run_specs}" \
  bash "${ROOT}/scripts/run_libero_plus_language_eval.sh" \
  2>&1 | tee -a "${LOG_FILE}"
log "teacher lambda LIBERO-Plus follow-up complete"
