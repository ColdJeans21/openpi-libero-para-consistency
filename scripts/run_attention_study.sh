#!/usr/bin/env bash
# Inference-only study: select attention bias on FM, then transfer it to Teacher.
set -euo pipefail
ROOT="${ROOT:-/home/gml/xz/openpi}"
cd "${ROOT}"
export PYTHONUNBUFFERED=1
OUT="${ROOT}/reports/attention/v1"
LOG="${ROOT}/logs/attention_v1"
PORT="${PORT:-8010}"
FM="${ROOT}/checkpoints/pi0_libero_goal_lora_canonical/canonical_30k/29999"
TEACHER="${ROOT}/checkpoints/pi0_libero_goal_lora_consistency/teacher_consistency_lam1_30k/29999"
SERVER_PID=""
mkdir -p "${OUT}" "${LOG}"
log() { echo "[$(date --iso-8601=seconds)] $*"; }
refresh() { CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/summarize_language_experiments.py; }
stop_server() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
  fi
}
trap stop_server EXIT
trap 'exit 130' INT TERM
start_server() {
  local model="$1" beta="$2" checkpoint="$3" label="$4"
  python3 - "${PORT}" <<'PY'
import socket, sys
with socket.socket() as sock:
    sock.bind(('127.0.0.1', int(sys.argv[1])))
PY
  setsid env CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    .venv/bin/python scripts/serve_attention_policy.py --model "${model}" \
      --beta "${beta}" --checkpoint "${checkpoint}" --port "${PORT}" \
      > "${LOG}/${label}_server.log" 2>&1 < /dev/null &
  SERVER_PID=$!
  for _ in $(seq 1 120); do
    kill -0 "${SERVER_PID}" || { tail -60 "${LOG}/${label}_server.log"; return 1; }
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
counterfactual() {
  local model="$1" beta="$2" split="$3" label="$4" checkpoint="${FM}"
  if [[ "${model}" == teacher ]]; then checkpoint="${TEACHER}"; fi
  log "Counterfactual ${label}: ${model}, beta=${beta}, split=${split}"
  start_server "${model}" "${beta}" "${checkpoint}" "${label}"
  docker run --rm --name "attention-${label//_/-}" --gpus device=0 --network host \
    -e MUJOCO_GL=egl -e MUJOCO_EGL_DEVICE_ID=0 -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e PYOPENGL_PLATFORM=egl \
    -e PYTHONPATH=/app/src:/app/packages/openpi-client/src:/app/third_party/libero \
    -v "${ROOT}:/app" --entrypoint /.venv/bin/python libero:latest \
    /app/examples/libero_attention/eval_counterfactual.py \
      --manifest /app/examples/libero_attention/manifest_v1.json \
      --output-dir "/app/reports/attention/v1/${label}" \
      --audit-dir /app/reports/attention/v1/shared_scene_audit \
      --split "${split}" --model "${model}" --beta "${beta}" --port "${PORT}" \
    2>&1 | tee "${LOG}/${label}.log"
  stop_server
  refresh
}
standard_eval() {
  local model="$1" beta="$2" run="attention_${1}_selected_v1"
  local checkpoint="${FM}" cfg=pi0_libero_goal_lora_canonical
  if [[ "${model}" == teacher ]]; then
    checkpoint="${TEACHER}"; cfg=pi0_libero_goal_lora_consistency
  fi
  env ATTENTION_MODEL="${model}" ATTENTION_BETA="${beta}" PORT="${PORT}" \
    RUN_NAME="${run}" CONFIG_NAME="${cfg}" CHECKPOINT_ROOT="${checkpoint%/29999}" \
    STEPS_STRING=29999 LOG_VALIDATION_TO_WANDB=0 SERVER_GPU=1 CLIENT_GPU=0 \
    bash scripts/run_libero_goal_para_validation.sh
  env ATTENTION_MODEL="${model}" ATTENTION_BETA="${beta}" PORT="${PORT}" \
    RUN_SPECS_OVERRIDE="${run}|${cfg}|${checkpoint}" \
    bash scripts/run_libero_plus_language_eval.sh
  refresh
}
for checkpoint in "${FM}" "${TEACHER}"; do test -f "${checkpoint}/_CHECKPOINT_METADATA"; done
log 'Deferred real-checkpoint GPU smoke: beta-zero equivalence, frozen parameters, finite intervention'
env CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 \
  .venv/bin/python scripts/smoke_attention_policy.py --checkpoint "${FM}" \
  2>&1 | tee "${LOG}/gpu_preflight.log"
counterfactual fm 0 dev fm_dev_b0
counterfactual fm 0.6931471805599453 dev fm_dev_bln2
counterfactual fm 1.3862943611198906 dev fm_dev_bln4
CUDA_VISIBLE_DEVICES='' .venv/bin/python - <<'PY'
import json
from pathlib import Path
from openpi.evaluation.attention_protocol import select_beta
root = Path('reports/attention/v1')
choice = select_beta([json.loads((root / f'fm_dev_{tag}/summary.json').read_text())
                      for tag in ('b0', 'bln2', 'bln4')])
target = root / 'selection.json'
if target.exists() and json.loads(target.read_text()) != choice:
    raise ValueError('Refusing to change a frozen beta selection')
target.write_text(json.dumps(choice, indent=2) + '\n')
print(json.dumps(choice, indent=2))
PY
beta=$(python3 -c "import json; print(json.load(open('reports/attention/v1/selection.json'))['selected_beta'])")
counterfactual fm 0 test fm_test_b0
counterfactual teacher 0 test teacher_test_b0
if [[ "${beta}" != 0.0 ]]; then
  counterfactual fm "${beta}" test fm_test_selected
  counterfactual teacher "${beta}" test teacher_test_selected
  standard_eval fm "${beta}"
  standard_eval teacher "${beta}"
else
  log 'FM dev selected zero: record null intervention; skip duplicate selected=baseline evaluations'
fi
refresh
if ! .venv/bin/python scripts/log_attention_to_wandb.py; then
  log 'W&B upload failed; local JSON/CSV/Markdown retained for retry'
fi
date --iso-8601=seconds > "${OUT}/COMPLETED"
log 'FM-first attention study complete'
