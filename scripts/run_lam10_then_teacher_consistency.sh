#!/usr/bin/env bash
set -euo pipefail

# Continue the controlled LIBERO-Para ablation sequence without overlapping GPU
# jobs: finish lambda=10, validate it, then train and validate the isolated
# canonical-teacher consistency objective.

ROOT="${ROOT:-/home/gml/xz/openpi}"
CONFIG_NAME="pi0_libero_goal_lora_consistency"
LAM10_RUN="consistency_lam10_30k"
TEACHER_RUN="teacher_consistency_lam1_30k"
CHECKPOINT_BASE="${ROOT}/checkpoints/${CONFIG_NAME}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MIN_FREE_GIB="${MIN_FREE_GIB:-35}"

timestamp() {
  date --iso-8601=seconds
}

wait_for_external_training() {
  local run_name="$1"
  local final_checkpoint="${CHECKPOINT_BASE}/${run_name}/29999/_CHECKPOINT_METADATA"
  local process_pattern="[s]cripts/train.py .*--exp-name ${run_name}"

  echo "[$(timestamp)] waiting for ${run_name} training"
  while [[ ! -f "${final_checkpoint}" ]]; do
    if ! pgrep -f "${process_pattern}" >/dev/null; then
      echo "${run_name} stopped before producing its final checkpoint" >&2
      exit 1
    fi
    sleep "${POLL_SECONDS}"
  done

  while pgrep -f "${process_pattern}" >/dev/null; do
    sleep 10
  done
  echo "[$(timestamp)] ${run_name} training complete"
}

run_validation() {
  local run_name="$1"
  echo "[$(timestamp)] validating ${run_name}"
  env \
    RUN_NAME="${run_name}" \
    CHECKPOINT_ROOT="${CHECKPOINT_BASE}/${run_name}" \
    CONFIG_NAME="${CONFIG_NAME}" \
    STEPS_STRING="10000 20000 29999" \
    SERVER_GPU=1 \
    CLIENT_GPU=0 \
    PORT=8000 \
    SEED=7 \
    GOAL_TRIALS=10 \
    bash "${ROOT}/scripts/run_libero_goal_para_validation.sh"
}

require_teacher_disk_space() {
  local available_kib
  available_kib=$(df --output=avail "${ROOT}" | tail -n 1 | tr -d ' ')
  if (( available_kib < MIN_FREE_GIB * 1024 * 1024 )); then
    echo "Need at least ${MIN_FREE_GIB} GiB free before ${TEACHER_RUN}; stopping safely" >&2
    exit 1
  fi
}

run_teacher_training() {
  local checkpoint_root="${CHECKPOINT_BASE}/${TEACHER_RUN}"
  if [[ -e "${checkpoint_root}" ]]; then
    echo "Refusing to overwrite existing teacher experiment: ${checkpoint_root}" >&2
    exit 1
  fi

  require_teacher_disk_space
  echo "[$(timestamp)] starting ${TEACHER_RUN}"
  env \
    CUDA_VISIBLE_DEVICES=0,1 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run "${ROOT}/scripts/train.py" "${CONFIG_NAME}" \
      --model.paired-loss-mode teacher_consistency \
      --model.consistency-loss-weight 1.0 \
      --exp-name "${TEACHER_RUN}" \
      --keep-period 10000 \
      --overwrite \
      > "${ROOT}/logs/train_${TEACHER_RUN}.log" 2>&1
  echo "[$(timestamp)] ${TEACHER_RUN} training complete"
}

cd "${ROOT}"
wait_for_external_training "${LAM10_RUN}"
run_validation "${LAM10_RUN}"
run_teacher_training
run_validation "${TEACHER_RUN}"
echo "[$(timestamp)] lambda=10 and teacher-consistency sequence complete"
