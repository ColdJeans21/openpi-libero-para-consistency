#!/usr/bin/env bash
set -euo pipefail

# Evaluate the final checkpoints from the controlled language-consistency study
# on the official LIBERO-Plus Language Instructions factor.  Runs are serial so
# the policy server can use GPU 1 and the simulator can use GPU 0.

ROOT="${ROOT:-/home/gml/xz/openpi}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/home/gml/xz/LIBERO-plus}"
SERVER_GPU="${SERVER_GPU:-1}"
CLIENT_GPU="${CLIENT_GPU:-0}"
PORT="${PORT:-8000}"
SEED="${SEED:-7}"
MAX_TASKS="${MAX_TASKS:--1}"
if [[ "${MAX_TASKS}" == "-1" ]]; then
  REPORT_ROOT="${REPORT_ROOT:-${ROOT}/reports/libero_plus_language}"
  LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/libero_plus_language}"
else
  REPORT_ROOT="${REPORT_ROOT:-${ROOT}/reports/libero_plus_language_smoke_${MAX_TASKS}}"
  LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/libero_plus_language_smoke_${MAX_TASKS}}"
fi
SERVER_PID=""

if [[ -n "${RUN_SPECS_OVERRIDE:-}" ]]; then
  # One newline-delimited `run_name|config_name|checkpoint_dir` entry per run.
  # Follow-up ablations can reuse the resumable evaluator without silently
  # re-running the four-model reference comparison.
  mapfile -t RUN_SPECS <<< "${RUN_SPECS_OVERRIDE}"
else
  RUN_SPECS=(
    "canonical_30k|pi0_libero_goal_lora_canonical|${ROOT}/checkpoints/pi0_libero_goal_lora_canonical/canonical_30k/29999"
    "para_aug_lam0_30k|pi0_libero_goal_lora_para_aug|${ROOT}/checkpoints/pi0_libero_goal_lora_para_aug/para_aug_lam0_30k/29999"
    "consistency_lam1_30k|pi0_libero_goal_lora_consistency|${ROOT}/checkpoints/pi0_libero_goal_lora_consistency/consistency_lam1_30k/29999"
    "teacher_consistency_lam1_30k|pi0_libero_goal_lora_consistency|${ROOT}/checkpoints/pi0_libero_goal_lora_consistency/teacher_consistency_lam1_30k/29999"
  )
fi

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

trap stop_server EXIT INT TERM
cd "${ROOT}"
mkdir -p "${REPORT_ROOT}" "${LOG_ROOT}"
require_file "${LIBERO_PLUS_ROOT}/libero/libero/benchmark/task_classification.json"

for spec in "${RUN_SPECS[@]}"; do
  IFS='|' read -r run_name config_name checkpoint_dir <<< "${spec}"
  require_file "${checkpoint_dir}/_CHECKPOINT_METADATA"
  output_dir="${REPORT_ROOT}/${run_name}/seed${SEED}"
  if [[ "${output_dir}" != "${ROOT}/"* ]]; then
    echo "REPORT_ROOT must be inside ROOT so Docker can write results: ${output_dir}" >&2
    exit 1
  fi
  container_output_dir="/app/${output_dir#"${ROOT}/"}"
  summary="${output_dir}/summary.json"
  mkdir -p "${output_dir}"

  if [[ "${MAX_TASKS}" == "-1" ]] && [[ -f "${summary}" ]] && \
      python3 - "${summary}" <<'PY'
import json
import sys

raise SystemExit(0 if json.load(open(sys.argv[1], encoding="utf-8")).get("complete") else 1)
PY
  then
    echo "[$(timestamp)] ${run_name}: complete summary exists; skipping"
    continue
  fi

  server_log="${LOG_ROOT}/${run_name}_server.log"
  eval_log="${LOG_ROOT}/${run_name}_seed${SEED}.log"
  echo "[$(timestamp)] ${run_name}: starting policy server"
  setsid env \
    CUDA_VISIBLE_DEVICES="${SERVER_GPU}" \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run scripts/serve_policy.py --port "${PORT}" policy:checkpoint \
      --policy.config="${config_name}" \
      --policy.dir="${checkpoint_dir}" \
      > "${server_log}" 2>&1 < /dev/null &
  SERVER_PID=$!

  if ! wait_for_server; then
    tail -n 100 "${server_log}" >&2 || true
    exit 1
  fi

  echo "[$(timestamp)] ${run_name}: evaluating LIBERO-Plus Language"
  docker run --rm --name "libero-plus-language-${run_name//_/-}" \
    --gpus "device=${CLIENT_GPU}" \
    --network host \
    -e MUJOCO_GL=egl \
    -e MUJOCO_EGL_DEVICE_ID=0 \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e PYOPENGL_PLATFORM=egl \
    -e PYTHONPATH=/app:/app/packages/openpi-client/src:/app/third_party/libero \
    -v "${ROOT}:/app" \
    -v "${LIBERO_PLUS_ROOT}:/libero-plus:ro" \
    --entrypoint /.venv/bin/python \
    libero:latest \
    /app/examples/libero_plus/eval_language_openpi.py \
      --libero-plus-root /libero-plus \
      --suite libero_goal \
      --host localhost \
      --port "${PORT}" \
      --gpu 0 \
      --seed "${SEED}" \
      --model-name "${run_name}" \
      --max-tasks "${MAX_TASKS}" \
      --output-dir "${container_output_dir}" \
      2>&1 | tee "${eval_log}"

  require_file "${summary}"
  stop_server
  echo "[$(timestamp)] ${run_name}: complete"
done

echo "[$(timestamp)] all LIBERO-Plus Language evaluations complete"
