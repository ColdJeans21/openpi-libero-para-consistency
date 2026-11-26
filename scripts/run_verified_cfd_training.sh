#!/usr/bin/env bash
# Launch only after verified collection and source tests have passed.
set -Eeuo pipefail
cd /home/gml/xz/openpi
PHASE="${1:-all}"
[[ "$PHASE" =~ ^(smoke|all)$ ]]
LOG=logs/verified_cfd_v2
DATA=/data/openpi_triplets/verified_initial_pairs_v2
SSD=/data/openpi_checkpoints
mkdir -p "$LOG"
exec 9>"$LOG/runner.lock"
flock 9
trap 'echo FAILED_CHECK_LOGS > "$LOG/STATUS"' ERR
[[ "$(findmnt -n -o TARGET --target "$SSD")" == /data ]]
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || { echo 'GPU busy; not starting'; exit 1; }
python3 - "$DATA/manifest.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
assert m['completed'] and len(m['shards']) >= 20
PY
# Actual transformed dataset validation before loading large weights.
JAX_PLATFORMS=cpu .venv/bin/python scripts/check_verified_cfd_data.py > "$LOG/data_check.json"
run_train() {
  local cfg="$1" run="$2" steps="$3" target link save_interval keep_period
  save_interval="$steps"
  keep_period="$steps"
  if [[ "$steps" == 30000 ]]; then save_interval=1000; keep_period=5000; fi
  target="$SSD/$cfg/$run"
  link="$PWD/checkpoints/$cfg/$run"
  mkdir -p "$(dirname "$target")" "$(dirname "$link")"
  if [[ -L "$link" ]]; then [[ "$(readlink "$link")" == "$target" ]];
  elif [[ -e "$link" ]]; then echo "Unexpected checkpoint path $link"; return 1;
  else ln -s "$target" "$link"; fi
  if [[ ! -f "$target/$((steps-1))/_CHECKPOINT_METADATA" ]]; then
    local resume=()
    if [[ -d "$target" ]]; then
      if find "$target" -name _CHECKPOINT_METADATA -print -quit | read -r _; then resume=(--resume);
      else echo "Incomplete directory without resumable checkpoint: $target"; return 1; fi
    fi
    local wandb=(--no-wandb-enabled)
    if [[ "$steps" == 30000 ]]; then wandb=(--wandb-enabled); fi
    echo "TRAINING $run" > "$LOG/STATUS"
    env CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
      WANDB_MODE="${WANDB_MODE:-online}" WANDB_ENTITY=xzzsryouxiwang-shenzhen-university \
      .venv/bin/python scripts/train.py "$cfg" --checkpoint-base-dir "$SSD" --exp-name "$run" \
      --num-train-steps "$steps" --save-interval "$save_interval" --keep-period "$keep_period" \
      --log-interval 1 "${wandb[@]}" "${resume[@]}" > "$LOG/train_$run.log" 2>&1
  fi
  test -f "$target/$((steps-1))/_CHECKPOINT_METADATA"
  # Reject nonfinite optimizer/loss traces rather than queue the full experiment.
  .venv/bin/python - "$LOG/train_$run.log" <<'PY'
import re,sys
s=open(sys.argv[1]).read()
assert re.search(r'Step \d+:',s), 'No training-step metrics'
assert not re.search(r'(?:loss[^:=,]*|grad_norm)\s*[:=]\s*(?:nan|inf)\b',s,re.I), 'Nonfinite training metrics'
PY
  cp examples/libero_triplets/verified_protocol_v2.md "$target/EXPERIMENT_PROTOCOL.md"
  cp "$DATA/manifest.json" "$target/VERIFIED_DATA_MANIFEST.json"
  cp "$LOG/data_check.json" "$target/DATA_CHECK.json"
  git rev-parse HEAD > "$target/CODE_COMMIT.txt"
  git diff -- src/openpi scripts examples/libero_triplets > "$target/WORKTREE.patch"
  nvidia-smi > "$target/GPU.txt"
  printf 'config=%s\nrun=%s\nbase=pi0_base\n' "$cfg" "$run" > "$target/RUN_LABEL.txt"
}
for r in 0 1 2; do
  cfg="pi0_libero_goal_verified_r$r"
  run="verified_r${r}_smoke20"
  if [[ ! -f "$SSD/$cfg/$run/SMOKE_PASSED" ]]; then
    run_train "$cfg" "$run" 20
    date --iso-8601=seconds > "$SSD/$cfg/$run/SMOKE_PASSED"
  fi
done
echo ALL_GPU_SMOKES_PASSED > "$LOG/STATUS"
[[ "$PHASE" == all ]] || exit 0
for r in 0 1 2; do
  cfg="pi0_libero_goal_verified_r$r"
  run="verified_r${r}_30k"
  run_train "$cfg" "$run" 30000
  echo "EVALUATING $run" > "$LOG/STATUS"
  env RUN_NAME="$run" CONFIG_NAME="$cfg" CHECKPOINT_ROOT="$PWD/checkpoints/$cfg/$run" \
    STEPS_STRING=29999 SERVER_GPU=1 CLIENT_GPU=0 PORT=8000 SEED=7 GOAL_TRIALS=10 \
    bash scripts/run_libero_goal_para_validation.sh
  env RUN_SPECS_OVERRIDE="$run|$cfg|$PWD/checkpoints/$cfg/$run/29999" bash scripts/run_libero_plus_language_eval.sh
  python3 scripts/summarize_language_experiments.py
  # Report generation reads raw summaries and leaves missing scores blank.
done
echo R0_R1_R2_TRAIN_AND_EVAL_COMPLETE > "$LOG/STATUS"
python3 scripts/summarize_language_experiments.py
