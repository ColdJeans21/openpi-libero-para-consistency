#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/gml/xz/openpi}"
LIBERO_PARA_ROOT="${LIBERO_PARA_ROOT:-/home/gml/xz/LIBERO-Para}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT}/checkpoints/pi0_libero_goal_lora_consistency/consistency_lam01_30k}"
CONFIG_NAME="${CONFIG_NAME:-pi0_libero_goal_lora_consistency}"
MANIFEST="${MANIFEST:-${ROOT}/examples/libero_para/validation_subset_v1_seed7.json}"
SERVER_GPU="${SERVER_GPU:-1}"
CLIENT_GPU="${CLIENT_GPU:-0}"
PORT="${PORT:-8000}"
SEED="${SEED:-7}"
WANDB_ENTITY="${WANDB_ENTITY:-xzzsryouxiwang-shenzhen-university}"
STEPS=(5000 10000 15000 20000 25000 29999)

SERVER_PID=""

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
    fi
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}

wait_for_server() {
  for _ in $(seq 1 90); do
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

log_to_wandb() {
  local step="$1"
  local output_dir="$2"
  local marker="${output_dir}/wandb_logged.txt"
  if [[ -f "${marker}" ]]; then
    return
  fi
  if uv run scripts/log_validation_to_wandb.py \
    --checkpoint-dir "${CHECKPOINT_ROOT}" \
    --checkpoint-step "${step}" \
    --libero-goal-summary "${ROOT}/reports/validation/step_${step}/libero_goal_seed7.json" \
    --libero-para-summary "${output_dir}/summary.json" \
    --project openpi \
    --entity "${WANDB_ENTITY}"; then
    date --iso-8601=seconds > "${marker}"
  else
    echo "W&B backfill failed for step ${step}; evaluation will continue." >&2
  fi
}

trap stop_server EXIT INT TERM
cd "${ROOT}"
mkdir -p logs

for step in "${STEPS[@]}"; do
  output_dir="${ROOT}/reports/validation/step_${step}/libero_para_seed7"
  summary_path="${output_dir}/summary.json"
  mkdir -p "${output_dir}"

  if [[ -f "${summary_path}" ]]; then
    echo "[$(date --iso-8601=seconds)] step ${step}: summary exists; skipping evaluation"
    log_to_wandb "${step}" "${output_dir}"
    continue
  fi

  if find "${output_dir}" -mindepth 1 -maxdepth 1 -type f | grep -q .; then
    echo "Refusing to append to partial output directory: ${output_dir}" >&2
    exit 1
  fi

  server_log="${ROOT}/logs/para_server_step${step}.log"
  eval_log="${ROOT}/logs/eval_para_step${step}_seed${SEED}.log"
  echo "[$(date --iso-8601=seconds)] step ${step}: starting policy server"
  setsid env \
    CUDA_VISIBLE_DEVICES="${SERVER_GPU}" \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run scripts/serve_policy.py --port "${PORT}" policy:checkpoint \
      --policy.config="${CONFIG_NAME}" \
      --policy.dir="${CHECKPOINT_ROOT}/${step}" \
      > "${server_log}" 2>&1 &
  SERVER_PID=$!

  if ! wait_for_server; then
    tail -n 80 "${server_log}" >&2 || true
    exit 1
  fi

  echo "[$(date --iso-8601=seconds)] step ${step}: evaluating fixed 100-task subset"
  docker run --rm --name "libero-para-step-${step}" \
    --gpus "device=${CLIENT_GPU}" \
    --network host \
    -e MUJOCO_GL=egl \
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
      --model-name "pi0-consistency-lam01-step${step}" \
      --output-dir "/app/reports/validation/step_${step}/libero_para_seed7" \
      2>&1 | tee "${eval_log}"

  stop_server
  log_to_wandb "${step}" "${output_dir}"
  echo "[$(date --iso-8601=seconds)] step ${step}: complete"
done

echo "[$(date --iso-8601=seconds)] all LIBERO-Para checkpoint validations complete"
