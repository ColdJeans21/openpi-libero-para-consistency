#!/usr/bin/env bash
set -euo pipefail

# Resume only the evaluation stages of the full Action Expert ablation after
# its training checkpoint has already completed.

ROOT="${ROOT:-/home/gml/xz/openpi}"
RUN_NAME="teacher_consistency_lam1_expert_full_30k"
CONFIG_NAME="pi0_libero_goal_teacher_expert_full"
CHECKPOINT_ROOT="${ROOT}/checkpoints/${CONFIG_NAME}/${RUN_NAME}"

cd "${ROOT}"
test -f "${CHECKPOINT_ROOT}/29999/_CHECKPOINT_METADATA"
test -f "${CHECKPOINT_ROOT}/wandb_id.txt"

env \
  RUN_NAME="${RUN_NAME}" \
  CHECKPOINT_ROOT="${CHECKPOINT_ROOT}" \
  CONFIG_NAME="${CONFIG_NAME}" \
  STEPS_STRING="10000 20000 29999" \
  SERVER_GPU=1 \
  CLIENT_GPU=0 \
  PORT=8000 \
  SEED=7 \
  GOAL_TRIALS=10 \
  WANDB_ENTITY=xzzsryouxiwang-shenzhen-university \
  bash "${ROOT}/scripts/run_libero_goal_para_validation.sh"

env \
  RUN_SPECS_OVERRIDE="${RUN_NAME}|${CONFIG_NAME}|${CHECKPOINT_ROOT}/29999" \
  bash "${ROOT}/scripts/run_libero_plus_language_eval.sh"

echo "[$(date --iso-8601=seconds)] Action Expert validation continuation complete"
