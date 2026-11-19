#!/usr/bin/env bash
set -euo pipefail

# Matched architecture ablation against E02 (para_aug_lam0_30k).
# The data, paired-FM objective, initialization, seed, batch size, and 30k
# budget are unchanged; only the fixed independent language readout is added.

ROOT="${ROOT:-/home/gml/xz/openpi}"
PHASE="${1:-all}"
CONFIG="pi0_libero_goal_paired_fm_language_fixed"
RUN="paired_fm_language_fixed_30k"
SSD_CHECKPOINT_BASE="${SSD_CHECKPOINT_BASE:-/data/openpi_checkpoints}"
LOG_ROOT="${ROOT}/logs/paired_fm_language_fixed"

log() { echo "[$(date --iso-8601=seconds)] $*"; }

prepare() {
  local run="$1" target link
  target="${SSD_CHECKPOINT_BASE}/${CONFIG}/${run}"
  link="${ROOT}/checkpoints/${CONFIG}/${run}"
  [[ ! -e "${target}" ]] || { echo "Refusing to overwrite ${target}" >&2; exit 1; }
  mkdir -p "${SSD_CHECKPOINT_BASE}/${CONFIG}" "${ROOT}/checkpoints/${CONFIG}"
  if [[ -L "${link}" ]]; then
    [[ "$(readlink "${link}")" == "${target}" ]]
  elif [[ -e "${link}" ]]; then
    echo "Refusing to replace ${link}" >&2
    exit 1
  else
    ln -s "${target}" "${link}"
  fi
}

record() {
  local target="$1" run="$2"
  git rev-parse HEAD > "${target}/CODE_COMMIT.txt"
  git diff -- src/openpi scripts/run_paired_fm_language_fixed_ssd.sh > "${target}/WORKTREE.patch"
  cat > "${target}/RUN_LABEL.txt" <<EOF
config=${CONFIG}
run=${run}
checkpoint_base=${SSD_CHECKPOINT_BASE}
matched_control_config=pi0_libero_goal_lora_para_aug
matched_control_run=para_aug_lam0_30k
training_data=original_libero_goal_10_task_canonical_paraphrase_pairs
objective=paired_fm_lambda0_shared_flow_coordinates
only_experimental_variable=fixed_independent_language_cross_attention
EOF
}

smoke() {
  local run="paired_fm_language_fixed_smoke20"
  local target="${SSD_CHECKPOINT_BASE}/${CONFIG}/${run}"
  if [[ -f "${target}/SMOKE_PASSED" ]]; then
    log "20-step smoke already passed"
    return
  fi
  prepare "${run}"
  log "starting 20-step real-data smoke"
  env CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
    uv run scripts/train.py "${CONFIG}" --checkpoint-base-dir "${SSD_CHECKPOINT_BASE}" \
      --exp-name "${run}" --num-train-steps 20 --save-interval 20 --keep-period 20 \
      --log-interval 1 --no-wandb-enabled 2>&1 | tee "${LOG_ROOT}/train_${run}.log"
  test -f "${target}/19/_CHECKPOINT_METADATA"
  record "${target}" "${run}"
  date --iso-8601=seconds > "${target}/SMOKE_PASSED"
}

train_full() {
  local target="${SSD_CHECKPOINT_BASE}/${CONFIG}/${RUN}"
  test -f "${SSD_CHECKPOINT_BASE}/${CONFIG}/paired_fm_language_fixed_smoke20/SMOKE_PASSED"
  prepare "${RUN}"
  log "starting matched 30k training"
  env CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=online \
    WANDB_ENTITY=xzzsryouxiwang-shenzhen-university \
    uv run scripts/train.py "${CONFIG}" --checkpoint-base-dir "${SSD_CHECKPOINT_BASE}" \
      --exp-name "${RUN}" --save-interval 30000 --keep-period 30000 \
      2>&1 | tee "${LOG_ROOT}/train_${RUN}.log"
  test -f "${target}/29999/_CHECKPOINT_METADATA"
  record "${target}" "${RUN}"

  env RUN_NAME="${RUN}" CONFIG_NAME="${CONFIG}" \
    CHECKPOINT_ROOT="${ROOT}/checkpoints/${CONFIG}/${RUN}" STEPS_STRING=29999 \
    SERVER_GPU=1 CLIENT_GPU=0 PORT=8000 SEED=7 GOAL_TRIALS=10 \
    bash scripts/run_libero_goal_para_validation.sh

  env RUN_SPECS_OVERRIDE="${RUN}|${CONFIG}|${ROOT}/checkpoints/${CONFIG}/${RUN}/29999" \
    SERVER_GPU=1 CLIENT_GPU=0 PORT=8000 SEED=7 \
    bash scripts/run_libero_plus_language_eval.sh
}

cd "${ROOT}"
mkdir -p "${LOG_ROOT}"
exec 9>"${LOG_ROOT}/runner.lock"
flock -n 9 || { echo "Paired-FM language experiment already active" >&2; exit 1; }
[[ "${PHASE}" =~ ^(smoke|train|all)$ ]]
[[ "$(findmnt -n -o TARGET --target "${SSD_CHECKPOINT_BASE}")" == /data ]]

if [[ "${PHASE}" == smoke || "${PHASE}" == all ]]; then smoke; fi
if [[ "${PHASE}" == train || "${PHASE}" == all ]]; then train_full; fi
date --iso-8601=seconds > "${LOG_ROOT}/COMPLETED_${PHASE}"
log "phase ${PHASE} complete"
