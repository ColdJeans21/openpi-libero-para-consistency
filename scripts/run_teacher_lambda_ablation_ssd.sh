#!/usr/bin/env bash
set -euo pipefail

# Run the stop-gradient teacher-consistency lambda ablation serially.  Large
# checkpoints live on the new SSD; repository-local symlinks preserve the
# paths expected by the existing serving and validation tooling.

ROOT="${ROOT:-/home/gml/xz/openpi}"
CONFIG_NAME="pi0_libero_goal_lora_consistency"
SSD_CHECKPOINT_BASE="${SSD_CHECKPOINT_BASE:-/data/openpi_checkpoints}"
SSD_CONFIG_ROOT="${SSD_CHECKPOINT_BASE}/${CONFIG_NAME}"
REPO_CONFIG_ROOT="${ROOT}/checkpoints/${CONFIG_NAME}"
LOG_ROOT="${ROOT}/logs/teacher_lambda_ablation_ssd"
RUN_MANIFEST="${ROOT}/examples/libero_para/runs/teacher_lambda_ablation_ssd.md"
MIN_FREE_GIB="${MIN_FREE_GIB:-200}"
# W&B authentication currently returns HTTP 403.  Offline mode preserves the
# complete metric stream and run id for a later `wandb sync` without blocking
# a multi-hour GPU job.  Set TRAIN_WANDB_MODE=online after credentials recover.
TRAIN_WANDB_MODE="${TRAIN_WANDB_MODE:-offline}"

RUN_SPECS=(
  "teacher_consistency_lam01_30k|0.1"
  "teacher_consistency_lam10_30k|10.0"
)

timestamp() {
  date --iso-8601=seconds
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

run_training() {
  local run_name="$1"
  local lambda="$2"
  local target="${SSD_CONFIG_ROOT}/${run_name}"
  local train_log="${LOG_ROOT}/train_${run_name}.log"

  echo "[$(timestamp)] starting ${run_name}: teacher consistency lambda=${lambda}"
  echo "[$(timestamp)] checkpoint target: ${target}"
  env \
    CUDA_VISIBLE_DEVICES=0,1 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    WANDB_MODE="${TRAIN_WANDB_MODE}" \
    uv run "${ROOT}/scripts/train.py" "${CONFIG_NAME}" \
      --model.paired-loss-mode teacher_consistency \
      --model.consistency-loss-weight "${lambda}" \
      --checkpoint-base-dir "${SSD_CHECKPOINT_BASE}" \
      --exp-name "${run_name}" \
      --keep-period 10000 \
      2>&1 | tee "${train_log}"

  test -f "${target}/29999/_CHECKPOINT_METADATA"
  cp "${RUN_MANIFEST}" "${target}/EXPERIMENT.md"
  echo "[$(timestamp)] ${run_name} training complete"
}

run_validation() {
  local run_name="$1"
  local checkpoint_root="${REPO_CONFIG_ROOT}/${run_name}"

  echo "[$(timestamp)] validating ${run_name} on matched Goal/Para"
  env \
    RUN_NAME="${run_name}" \
    CHECKPOINT_ROOT="${checkpoint_root}" \
    CONFIG_NAME="${CONFIG_NAME}" \
    STEPS_STRING="10000 20000 29999" \
    SERVER_GPU=1 \
    CLIENT_GPU=0 \
    PORT=8000 \
    SEED=7 \
    GOAL_TRIALS=10 \
    bash "${ROOT}/scripts/run_libero_goal_para_validation.sh"
}

cd "${ROOT}"
mkdir -p "${SSD_CONFIG_ROOT}" "${REPO_CONFIG_ROOT}" "${LOG_ROOT}"
require_new_ssd

for spec in "${RUN_SPECS[@]}"; do
  IFS='|' read -r run_name lambda <<< "${spec}"
  prepare_run "${run_name}"
  run_training "${run_name}" "${lambda}"
  run_validation "${run_name}"
done

echo "[$(timestamp)] teacher lambda ablation and matched validation complete"
