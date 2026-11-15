#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/gml/xz/openpi}"
PHASE="${1:-all}"
SSD_CHECKPOINT_BASE="${SSD_CHECKPOINT_BASE:-/data/openpi_checkpoints}"
LOG_ROOT="${ROOT}/logs/semantic_residual_v2"
MANIFEST="${ROOT}/examples/libero_para/runs/semantic_residual_v2_ssd.md"
MAX_ALLOWED_PEAK_MIB="${MAX_ALLOWED_PEAK_MIB:-23500}"
MONITOR_PID=""
SPECS=(
  "pi0_libero_goal_teacher_prefix_control|teacher_prefix_control_30k"
  "pi0_libero_goal_teacher_semantic_residual|teacher_prefix_residual_r16_30k"
)
export OPENPI_TEXT_MODEL_DIR="${OPENPI_TEXT_MODEL_DIR:-/data/openpi_models/t5-large}"
export PYTHONUNBUFFERED=1

log() { echo "[$(date --iso-8601=seconds)] $*"; }
require_idle_gpus() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
    awk '$1 > 1500 {bad=1} END {exit bad}' || {
      log 'GPU memory is already occupied; refusing to compete with another job'; exit 1;
    }
}
stop_monitor() {
  if [[ -n "${MONITOR_PID}" ]]; then
    kill -TERM "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
  MONITOR_PID=""
}
trap stop_monitor EXIT INT TERM

prepare() {
  local cfg="$1" run="$2" target link
  target="${SSD_CHECKPOINT_BASE}/${cfg}/${run}"
  link="${ROOT}/checkpoints/${cfg}/${run}"
  [[ ! -e "${target}" ]] || { log "Refusing to overwrite ${target}"; exit 1; }
  mkdir -p "${SSD_CHECKPOINT_BASE}/${cfg}" "${ROOT}/checkpoints/${cfg}"
  if [[ -L "${link}" ]]; then
    [[ "$(readlink "${link}")" == "${target}" ]]
  elif [[ -e "${link}" ]]; then
    log "Refusing to replace ${link}"; exit 1
  else
    ln -s "${target}" "${link}"
  fi
}

record() {
  local target="$1" cfg="$2" run="$3"
  cp "${MANIFEST}" "${target}/EXPERIMENT.md"
  cp "${LOG_ROOT}/launch_commit.txt" "${target}/CODE_COMMIT.txt"
  cp "${LOG_ROOT}/launch_worktree.patch" "${target}/WORKTREE.patch"
  sha256sum assets/semantic_interface/t5_large_train_v1.npz > "${target}/TEXT_ASSET.sha256"
  printf 'config=%s\nrun=%s\ncheckpoint_base=%s\nphase=%s\n' \
    "${cfg}" "${run}" "${SSD_CHECKPOINT_BASE}" "${PHASE}" > "${target}/RUN_LABEL.txt"
}

smoke() {
  local cfg="$1" full="$2" run target memory_log status peak
  run="${full%_30k}_smoke20"
  target="${SSD_CHECKPOINT_BASE}/${cfg}/${run}"
  if [[ -f "${target}/SMOKE_PASSED" ]]; then
    log "${run}: verified smoke marker exists"; return
  fi
  require_idle_gpus
  prepare "${cfg}" "${run}"
  memory_log="${LOG_ROOT}/${run}_gpu_memory.csv"
  echo 'gpu_index,memory_used_mib' > "${memory_log}"
  (while true; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits >> "${memory_log}"; sleep 2; done) &
  MONITOR_PID=$!
  log "${run}: real-data 20-step smoke"
  set +e
  env CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
    uv run scripts/train.py "${cfg}" --checkpoint-base-dir "${SSD_CHECKPOINT_BASE}" \
      --exp-name "${run}" --num-train-steps 20 --save-interval 20 --keep-period 20 \
      --log-interval 1 --no-wandb-enabled 2>&1 | tee "${LOG_ROOT}/train_${run}.log"
  status=${PIPESTATUS[0]}
  set -e
  stop_monitor
  peak=$(awk -F, 'NR > 1 {v=$2+0; if (v>p) p=v} END {print p+0}' "${memory_log}")
  (( status == 0 && peak > 0 && peak <= MAX_ALLOWED_PEAK_MIB ))
  test -f "${target}/19/_CHECKPOINT_METADATA"
  record "${target}" "${cfg}" "${run}"
  cp "${memory_log}" "${target}/GPU_MEMORY_SMOKE.csv"
  env CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 \
    uv run scripts/smoke_semantic_policy.py --config "${cfg}" --checkpoint "${target}/19" \
      2>&1 | tee "${LOG_ROOT}/infer_${run}.log"
  printf 'peak_memory_mib=%s\n' "${peak}" > "${target}/SMOKE_PASSED"
  log "${run}: train/reload smoke passed; peak ${peak} MiB"
}

evaluate() {
  local run="$1" cfg="$2" target="$3" steps="$4" namespace="$5" auto_upload=1
  if [[ "${namespace}" != validation ]]; then auto_upload=0; fi
  env RUN_NAME="${run}" CONFIG_NAME="${cfg}" CHECKPOINT_ROOT="${target}" \
    STEPS_STRING="${steps}" SERVER_GPU=1 CLIENT_GPU=0 PORT=8000 SEED=7 GOAL_TRIALS=10 \
    LOG_VALIDATION_TO_WANDB="${auto_upload}" \
    bash scripts/run_libero_goal_para_validation.sh
  env RUN_SPECS_OVERRIDE="${run}|${cfg}|${target}/29999" \
    bash scripts/run_libero_plus_language_eval.sh
  if uv run scripts/log_validation_to_wandb.py --checkpoint-dir "${target}" --checkpoint-step 29999 \
    --libero-goal-summary "reports/validation/${run}/step_29999/libero_goal_seed7.json" \
    --libero-para-summary "reports/validation/${run}/step_29999/libero_para_seed7/summary.json" \
    --libero-plus-summary "reports/libero_plus_language/${run}/seed7/summary.json" \
    --metric-namespace "${namespace}" --entity xzzsryouxiwang-shenzhen-university; then
    date --iso-8601=seconds > "${target}/${namespace}_FINAL_WANDB_LOGGED.txt"
  else
    log "${run}: W&B upload failed; local results retained for retry"
  fi
}

full() {
  local cfg="$1" run="$2" target
  target="${SSD_CHECKPOINT_BASE}/${cfg}/${run}"
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r smoke_cfg smoke_run <<< "${spec}"
    test -f "${SSD_CHECKPOINT_BASE}/${smoke_cfg}/${smoke_run%_30k}_smoke20/SMOKE_PASSED"
  done
  require_idle_gpus
  prepare "${cfg}" "${run}"
  log "${run}: starting matched 30k Teacher training"
  env CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=online \
    WANDB_ENTITY=xzzsryouxiwang-shenzhen-university \
    uv run scripts/train.py "${cfg}" --checkpoint-base-dir "${SSD_CHECKPOINT_BASE}" \
      --exp-name "${run}" --keep-period 10000 \
      2>&1 | tee "${LOG_ROOT}/train_${run}.log"
  test -f "${target}/29999/_CHECKPOINT_METADATA"
  record "${target}" "${cfg}" "${run}"
  evaluate "${run}" "${cfg}" "${target}" "10000 20000 29999" validation
  if [[ "${cfg}" == pi0_libero_goal_teacher_semantic_residual ]]; then
    log "${run}: evaluating the SAME checkpoint with residual scale zero"
    evaluate "${run}_residual_off" pi0_libero_goal_teacher_semantic_residual_off \
      "${target}" "29999" validation_residual_off
  fi
  log "${run}: training and all evaluations complete"
}

cd "${ROOT}"
mkdir -p "${LOG_ROOT}"
exec 9>"${LOG_ROOT}/runner.lock"
flock -n 9 || { log 'Residual experiment runner already active'; exit 1; }
[[ "${PHASE}" == smoke || "${PHASE}" == full || "${PHASE}" == all ]]
[[ "$(findmnt -n -o TARGET --target "${SSD_CHECKPOINT_BASE}")" == /data ]]
available_kib=$(df --output=avail "${SSD_CHECKPOINT_BASE}" | tail -n 1 | tr -d ' ')
(( available_kib >= 150 * 1024 * 1024 ))
test -f assets/semantic_interface/t5_large_train_v1.npz
test -f "${OPENPI_TEXT_MODEL_DIR}/model.safetensors"
git rev-parse HEAD > "${LOG_ROOT}/launch_commit.txt"
git diff -- src/openpi/models src/openpi/training/config.py src/openpi/transforms.py \
  src/openpi/shared/semantic_text_test.py scripts/log_validation_to_wandb.py \
  scripts/smoke_semantic_policy.py scripts/run_libero_goal_para_validation.sh \
  scripts/run_semantic_residual_ssd.sh > "${LOG_ROOT}/launch_worktree.patch"
if [[ "${PHASE}" != full ]]; then
  for spec in "${SPECS[@]}"; do IFS='|' read -r cfg run <<< "${spec}"; smoke "${cfg}" "${run}"; done
fi
if [[ "${PHASE}" != smoke ]]; then
  for spec in "${SPECS[@]}"; do IFS='|' read -r cfg run <<< "${spec}"; full "${cfg}" "${run}"; done
fi
log "semantic-residual phase ${PHASE} complete"
