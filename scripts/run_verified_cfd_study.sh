#!/usr/bin/env bash
set -euo pipefail
cd /home/gml/xz/openpi
DATA=/data/openpi_triplets/verified_initial_pairs_v2
LOG=logs/verified_cfd_v2
mkdir -p "$LOG"
exec 9>"$LOG/runner.lock"
flock -n 9 || exit 1
SERVER=""
cleanup() { if [[ -n "$SERVER" ]]; then kill -TERM -- "-$SERVER" 2>/dev/null || true; wait "$SERVER" 2>/dev/null || true; fi; }
trap cleanup EXIT
trap 'echo COLLECTION_FAILED_CHECK_LOGS > "$LOG/STATUS"' ERR
trap 'exit 130' INT TERM
nvidia-smi >/dev/null
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then echo 'GPU busy; refusing to interfere'; exit 1; fi
if ! python3 -c 'import json,sys; assert json.load(open(sys.argv[1])).get("completed")' "$DATA/manifest.json" 2>/dev/null; then
  echo COLLECTING_VERIFIED_PAIRS > "$LOG/STATUS"
  setsid env CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/serve_triplet_teacher.py --port 8011 > "$LOG/teacher.log" 2>&1 < /dev/null &
  SERVER=$!
  ready=0
  for _ in $(seq 1 180); do
    kill -0 "$SERVER" || exit 1
    if python3 -c 'import socket; s=socket.socket(); s.settimeout(.2); exit(s.connect_ex(("127.0.0.1",8011)))'; then ready=1; break; fi
    sleep 2
  done
  [[ "$ready" == 1 ]]
  docker run --rm --name libero-verified-pairs-v2 --gpus device=0 --network host \
    -e MUJOCO_GL=egl -e MUJOCO_EGL_DEVICE_ID=0 -e NVIDIA_DRIVER_CAPABILITIES=all -e PYOPENGL_PLATFORM=egl \
    -e PYTHONPATH=/app:/app/src:/app/packages/openpi-client/src:/app/third_party/libero \
    -v "$PWD:/app" -v /data:/data --entrypoint /.venv/bin/python libero:latest \
    /app/examples/libero_triplets/collect_verified_pairs.py --manifest /app/examples/libero_attention/manifest_v1.json \
    --output-dir "$DATA" --port 8011 2>&1 | tee "$LOG/collect.log"
  cleanup
  SERVER=""
fi
echo VERIFIED_DATA_READY > "$LOG/STATUS"
