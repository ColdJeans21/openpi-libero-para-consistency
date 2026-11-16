#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/gml/xz/openpi}"
PHASE="${1:-all}"
DATA_ROOT="${DATA_ROOT:-/data/openpi_triplets/cfd_pilot_v1}"
SSD_CHECKPOINT_BASE="${SSD_CHECKPOINT_BASE:-/data/openpi_checkpoints}"
LOG_ROOT="${ROOT}/logs/triplet_language_study"
PORT="${PORT:-8011}"
SERVER_PID=""
SPECS=(
  "pi0_libero_goal_triplet_fm_control|triplet_fm_control_30k"
  "pi0_libero_goal_triplet_language_fixed|triplet_language_fixed_30k"
  "pi0_libero_goal_triplet_language_dynamic|triplet_language_dynamic_30k"
)

log() { echo "[$(date --iso-8601=seconds)] $*"; }
stop_server() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
  fi
}
trap stop_server EXIT INT TERM

start_teacher() {
  setsid env CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    .venv/bin/python scripts/serve_triplet_teacher.py --port "${PORT}" \
      > "${LOG_ROOT}/teacher_server.log" 2>&1 < /dev/null &
  SERVER_PID=$!
  for _ in $(seq 1 180); do
    kill -0 "${SERVER_PID}" 2>/dev/null || { tail -80 "${LOG_ROOT}/teacher_server.log"; return 1; }
    if python3 - "${PORT}" <<'PY'
import socket, sys
with socket.socket() as sock:
    sock.settimeout(.2)
    raise SystemExit(sock.connect_ex(('127.0.0.1', int(sys.argv[1]))))
PY
    then return; fi
    sleep 2
  done
  return 1
}

collect() {
  if [[ -f "${DATA_ROOT}/manifest.json" ]] && \
     python3 -c 'import json,sys; raise SystemExit(not json.load(open(sys.argv[1])).get("completed"))' \
       "${DATA_ROOT}/manifest.json"; then
    log "triplet dataset already complete"
    return
  fi
  log "starting frozen official-teacher triplet collection"
  start_teacher
  docker run --rm --name libero-triplet-collector --gpus device=0 --network host \
    -e MUJOCO_GL=egl -e MUJOCO_EGL_DEVICE_ID=0 -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e PYOPENGL_PLATFORM=egl \
    -e PYTHONPATH=/app/src:/app/packages/openpi-client/src:/app/third_party/libero \
    -v "${ROOT}:/app" -v /data:/data --entrypoint /.venv/bin/python libero:latest \
      /app/examples/libero_triplets/collect_teacher_triplets.py \
      --manifest /app/examples/libero_attention/manifest_v1.json \
      --output-dir "${DATA_ROOT}" --port "${PORT}" \
      2>&1 | tee "${LOG_ROOT}/collect.log"
  stop_server
  python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["completed"]' "${DATA_ROOT}/manifest.json"
}

prepare() {
  local cfg="$1" run="$2" target link
  target="${SSD_CHECKPOINT_BASE}/${cfg}/${run}"
  link="${ROOT}/checkpoints/${cfg}/${run}"
  [[ ! -e "${target}" ]] || { echo "Refusing to overwrite ${target}" >&2; exit 1; }
  mkdir -p "${SSD_CHECKPOINT_BASE}/${cfg}" "${ROOT}/checkpoints/${cfg}"
  if [[ -L "${link}" ]]; then
    [[ "$(readlink "${link}")" == "${target}" ]]
  elif [[ -e "${link}" ]]; then
    echo "Refusing to replace ${link}" >&2; exit 1
  else
    ln -s "${target}" "${link}"
  fi
}

record() {
  local target="$1" cfg="$2" run="$3"
  cp examples/libero_triplets/experiment_protocol.md "${target}/EXPERIMENT_PROTOCOL.md"
  cp "${DATA_ROOT}/manifest.json" "${target}/TRIPLET_DATA_MANIFEST.json"
  git rev-parse HEAD > "${target}/CODE_COMMIT.txt"
  git diff -- src/openpi examples/libero_triplets scripts > "${target}/WORKTREE.patch"
  printf 'config=%s\nrun=%s\ncheckpoint_base=%s\n' "${cfg}" "${run}" "${SSD_CHECKPOINT_BASE}" \
    > "${target}/RUN_LABEL.txt"
}

smoke() {
  local cfg="$1" full="$2" run target
  run="${full%_30k}_smoke20"
  target="${SSD_CHECKPOINT_BASE}/${cfg}/${run}"
  if [[ -f "${target}/SMOKE_PASSED" ]]; then return; fi
  prepare "${cfg}" "${run}"
  log "${run}: 20-step real-triplet smoke"
  env CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
    uv run scripts/train.py "${cfg}" --checkpoint-base-dir "${SSD_CHECKPOINT_BASE}" \
      --exp-name "${run}" --num-train-steps 20 --save-interval 20 --keep-period 20 \
      --log-interval 1 --no-wandb-enabled 2>&1 | tee "${LOG_ROOT}/train_${run}.log"
  test -f "${target}/19/_CHECKPOINT_METADATA"
  record "${target}" "${cfg}" "${run}"
  date --iso-8601=seconds > "${target}/SMOKE_PASSED"
}

train_full() {
  local cfg="$1" run="$2" target
  target="${SSD_CHECKPOINT_BASE}/${cfg}/${run}"
  test -f "${SSD_CHECKPOINT_BASE}/${cfg}/${run%_30k}_smoke20/SMOKE_PASSED"
  prepare "${cfg}" "${run}"
  log "${run}: 30k training"
  env CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=online \
    WANDB_ENTITY=xzzsryouxiwang-shenzhen-university \
    uv run scripts/train.py "${cfg}" --checkpoint-base-dir "${SSD_CHECKPOINT_BASE}" \
      --exp-name "${run}" --save-interval 30000 --keep-period 30000 \
      2>&1 | tee "${LOG_ROOT}/train_${run}.log"
  test -f "${target}/29999/_CHECKPOINT_METADATA"
  record "${target}" "${cfg}" "${run}"
  env RUN_NAME="${run}" CONFIG_NAME="${cfg}" CHECKPOINT_ROOT="${ROOT}/checkpoints/${cfg}/${run}" \
    STEPS_STRING=29999 SERVER_GPU=1 CLIENT_GPU=0 PORT=8000 SEED=7 GOAL_TRIALS=10 \
    bash scripts/run_libero_goal_para_validation.sh
  env RUN_SPECS_OVERRIDE="${run}|${cfg}|${ROOT}/checkpoints/${cfg}/${run}/29999" \
    bash scripts/run_libero_plus_language_eval.sh
}

cd "${ROOT}"
mkdir -p "${LOG_ROOT}"
exec 9>"${LOG_ROOT}/runner.lock"
flock -n 9 || { echo 'Triplet language study already active' >&2; exit 1; }
[[ "${PHASE}" =~ ^(collect|smoke|train|all)$ ]]
[[ "$(findmnt -n -o TARGET --target "${SSD_CHECKPOINT_BASE}")" == /data ]]
collect
if [[ "${PHASE}" == collect ]]; then exit 0; fi
if [[ "${PHASE}" == smoke || "${PHASE}" == all ]]; then
  for spec in "${SPECS[@]}"; do IFS='|' read -r cfg run <<< "${spec}"; smoke "${cfg}" "${run}"; done
fi
if [[ "${PHASE}" == train || "${PHASE}" == all ]]; then
  for spec in "${SPECS[@]}"; do IFS='|' read -r cfg run <<< "${spec}"; train_full "${cfg}" "${run}"; done
fi
date --iso-8601=seconds > "${LOG_ROOT}/COMPLETED_${PHASE}"
log "phase ${PHASE} complete"
