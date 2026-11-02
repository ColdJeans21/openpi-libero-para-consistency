#!/usr/bin/env bash
set -euo pipefail

# Run matched LIBERO-Goal and fixed-subset LIBERO-Para closed-loop validation
# for one training run. Outputs are namespaced by RUN_NAME, so separate
# ablations cannot overwrite each other's summaries.

ROOT="${ROOT:-/home/gml/xz/openpi}"
LIBERO_PARA_ROOT="${LIBERO_PARA_ROOT:-/home/gml/xz/LIBERO-Para}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME to the experiment name}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:?Set CHECKPOINT_ROOT to the checkpoint directory}"
CONFIG_NAME="${CONFIG_NAME:?Set CONFIG_NAME to the OpenPI training config}"
STEPS_STRING="${STEPS_STRING:-10000 20000 29999}"
MANIFEST="${MANIFEST:-${ROOT}/examples/libero_para/validation_subset_v1_seed7.json}"
REPORT_ROOT="${REPORT_ROOT:-${ROOT}/reports/validation/${RUN_NAME}}"
SERVER_GPU="${SERVER_GPU:-1}"
CLIENT_GPU="${CLIENT_GPU:-0}"
PORT="${PORT:-8000}"
SEED="${SEED:-7}"
GOAL_TRIALS="${GOAL_TRIALS:-10}"
WANDB_ENTITY="${WANDB_ENTITY:-xzzsryouxiwang-shenzhen-university}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/validation/${RUN_NAME}}"

read -r -a STEPS <<< "${STEPS_STRING}"
SERVER_PID=""

timestamp() {
  date --iso-8601=seconds
}

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      kill -KILL -- "-${SERVER_PID}" 2>/dev/null || kill -KILL "${SERVER_PID}" 2>/dev/null || true
    fi
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}

wait_for_server() {
  for _ in $(seq 1 120); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      return 1
    fi
    if python3 - "${PORT}" <<'PY'
import socket
import sys

with socket.socket() as sock:
    sock.settimeout(0.2)
    raise SystemExit(sock.connect_ex(("127.0.0.1", int(sys.argv[1]))))
PY
    then
      return 0
    fi
    sleep 2
  done
  return 1
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

run_goal() {
  local step="$1"
  local step_dir="$2"
  local summary="${step_dir}/libero_goal_seed${SEED}.json"
  local log="${LOG_ROOT}/goal_step${step}_seed${SEED}.log"
  local video_dir="/data/libero/validation_${RUN_NAME}_step${step}_seed${SEED}"

  if [[ -f "${summary}" ]]; then
    echo "[$(timestamp)] step ${step}: Goal summary exists; skipping"
    return
  fi

  echo "[$(timestamp)] step ${step}: evaluating LIBERO-Goal (${GOAL_TRIALS} trials/task)"
  docker run --rm --name "libero-goal-${RUN_NAME//_/-}-${step}" \
    --gpus "device=${CLIENT_GPU}" \
    --network host \
    -e MUJOCO_GL=egl \
    -e MUJOCO_EGL_DEVICE_ID=0 \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e PYOPENGL_PLATFORM=egl \
    -e PYTHONPATH=/app:/app/packages/openpi-client/src:/app/third_party/libero \
    -v "${ROOT}:/app" \
    -v "${ROOT}/data:/data" \
    --entrypoint /.venv/bin/python \
    libero:latest \
    /app/examples/libero/main.py \
      --args.host localhost \
      --args.port "${PORT}" \
      --args.task-suite-name libero_goal \
      --args.num-trials-per-task "${GOAL_TRIALS}" \
      --args.seed "${SEED}" \
      --args.video-out-path "${video_dir}" \
      --args.summary-path "/app/reports/validation/${RUN_NAME}/step_${step}/libero_goal_seed${SEED}.json" \
      2>&1 | tee "${log}"
  require_file "${summary}"
}

run_para() {
  local step="$1"
  local step_dir="$2"
  local output_dir="${step_dir}/libero_para_seed${SEED}"
  local summary="${output_dir}/summary.json"
  local log="${LOG_ROOT}/para_step${step}_seed${SEED}.log"

  if [[ -f "${summary}" ]]; then
    echo "[$(timestamp)] step ${step}: Para summary exists; skipping"
    return
  fi
  mkdir -p "${output_dir}"
  if find "${output_dir}" -mindepth 1 -maxdepth 1 -type f | grep -q .; then
    echo "Refusing to append to partial Para output directory: ${output_dir}" >&2
    exit 1
  fi

  echo "[$(timestamp)] step ${step}: evaluating fixed LIBERO-Para subset"
  docker run --rm --name "libero-para-${RUN_NAME//_/-}-${step}" \
    --gpus "device=${CLIENT_GPU}" \
    --network host \
    -e MUJOCO_GL=egl \
    -e MUJOCO_EGL_DEVICE_ID=0 \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e PYOPENGL_PLATFORM=egl \
    -e PYTHONPATH=/app/third_party/libero:/app/packages/openpi-client/src \
    -v "${LIBERO_PARA_ROOT}:/para:ro" \
    -v "${ROOT}:/app" \
    --entrypoint /.venv/bin/python \
    libero:latest \
    /app/examples/libero_para/eval_openpi.py \
      --libero-para-root /para \
      --host localhost \
      --port "${PORT}" \
      --gpu 0 \
      --seed "${SEED}" \
      --task-manifest /app/examples/libero_para/validation_subset_v1_seed7.json \
      --model-name "${RUN_NAME}-step${step}" \
      --output-dir "/app/reports/validation/${RUN_NAME}/step_${step}/libero_para_seed${SEED}" \
      2>&1 | tee "${log}"
  require_file "${summary}"
}

log_to_wandb() {
  local step="$1"
  local step_dir="$2"
  local marker="${step_dir}/wandb_logged.txt"
  if [[ -f "${marker}" ]]; then
    echo "[$(timestamp)] step ${step}: W&B marker exists; skipping backfill"
    return
  fi

  if uv run scripts/log_validation_to_wandb.py \
    --checkpoint-dir "${CHECKPOINT_ROOT}" \
    --checkpoint-step "${step}" \
    --libero-goal-summary "${step_dir}/libero_goal_seed${SEED}.json" \
    --libero-para-summary "${step_dir}/libero_para_seed${SEED}/summary.json" \
    --project openpi \
    --entity "${WANDB_ENTITY}"; then
    date --iso-8601=seconds > "${marker}"
  else
    echo "W&B backfill failed for step ${step}; evaluations are preserved." >&2
  fi
}

trap stop_server EXIT INT TERM
cd "${ROOT}"
mkdir -p "${REPORT_ROOT}" "${LOG_ROOT}"
require_file "${MANIFEST}"
require_file "${CHECKPOINT_ROOT}/wandb_id.txt"

for step in "${STEPS[@]}"; do
  require_file "${CHECKPOINT_ROOT}/${step}/_CHECKPOINT_METADATA"
  step_dir="${REPORT_ROOT}/step_${step}"
  mkdir -p "${step_dir}"

  if [[ -f "${step_dir}/wandb_logged.txt" ]]; then
    echo "[$(timestamp)] step ${step}: already complete"
    continue
  fi

  server_log="${LOG_ROOT}/server_step${step}.log"
  echo "[$(timestamp)] step ${step}: starting policy server"
  setsid env \
    CUDA_VISIBLE_DEVICES="${SERVER_GPU}" \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run scripts/serve_policy.py --port "${PORT}" policy:checkpoint \
      --policy.config="${CONFIG_NAME}" \
      --policy.dir="${CHECKPOINT_ROOT}/${step}" \
      > "${server_log}" 2>&1 < /dev/null &
  SERVER_PID=$!

  if ! wait_for_server; then
    tail -n 100 "${server_log}" >&2 || true
    exit 1
  fi

  run_goal "${step}" "${step_dir}"
  run_para "${step}" "${step_dir}"
  stop_server
  log_to_wandb "${step}" "${step_dir}"
  echo "[$(timestamp)] step ${step}: complete"
done

echo "[$(timestamp)] all matched Goal/Para validations complete for ${RUN_NAME}"
