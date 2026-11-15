#!/usr/bin/env bash
# This watcher waits for the current runner INCLUDING all of its evaluations.
set -euo pipefail
ROOT="${ROOT:-/home/gml/xz/openpi}"
cd "${ROOT}"
LOG="${ROOT}/logs/attention_v1"
mkdir -p "${LOG}"
exec 7>"${LOG}/queue.lock"
flock -n 7 || { echo 'Attention watcher already active'; exit 1; }
state() { printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee "${LOG}/STATUS"; }
trap 'state BLOCKED_OR_FAILED_check_queue_log' ERR
state WAITING_FOR_RESIDUAL_TRAINING_AND_ALL_EVALUATIONS
exec 8>"${ROOT}/logs/semantic_residual_v2/runner.lock"
flock 8
state CHECKING_RESIDUAL_COMPLETION
CUDA_VISIBLE_DEVICES='' .venv/bin/python - <<'PY'
import json
from pathlib import Path
root = Path('reports')
run = 'teacher_prefix_residual_r16_30k'
checkpoint = Path('/data/openpi_checkpoints/pi0_libero_goal_teacher_semantic_residual') / run / '29999/_CHECKPOINT_METADATA'
assert checkpoint.is_file(), 'Residual training did not finish; refuse to bypass it'
for name, steps in ((run, (10000, 20000, 29999)), (run + '_residual_off', (29999,))):
    for step in steps:
        base = root / 'validation' / name / f'step_{step}'
        for path in (base / 'libero_goal_seed7.json', base / 'libero_para_seed7/summary.json'):
            data = json.loads(path.read_text())
            assert data['total_episodes'] == 100, f'Incomplete residual evaluation: {path}'
    path = root / 'libero_plus_language' / name / 'seed7/summary.json'
    data = json.loads(path.read_text())
    assert data['complete'] and data['total_episodes'] == 410, f'Incomplete Plus: {path}'
print('Residual training, 10k/20k/final validation, final Plus and residual-off control complete')
PY
sha256sum --check "${LOG}/queued_sources.sha256"
while ! nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
    awk '$1>1500 {busy=1} END {exit busy}'; do
  state WAITING_FOR_FREE_GPUS_AFTER_RESIDUAL
  sleep 60
done
state RUNNING_FM_FIRST_ATTENTION_STUDY
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/summarize_language_experiments.py
bash scripts/run_attention_study.sh
state COMPLETED
# Only the generated study reports and small summaries may be auto-published.
# Never stage other worktree changes or force-push.
if [[ "$(git branch --show-current)" == libero-para-consistency ]]; then
  git add -- reports/ALL_EXPERIMENTS_METHODS_AND_RESULTS.md reports/all_language_experiments.csv
  git add -- reports/attention/v1/selection.json reports/attention/v1/COMPLETED
  while IFS= read -r path; do git add -- "${path}"; done < <(find reports/attention/v1 -name summary.json -type f)
  for run in teacher_prefix_residual_r16_30k teacher_prefix_residual_r16_30k_residual_off \
             attention_fm_selected_v1 attention_teacher_selected_v1; do
    if [[ -d "reports/validation/${run}" ]]; then
      while IFS= read -r path; do git add -- "${path}"; done < <(find "reports/validation/${run}" \
        -type f \( -name libero_goal_seed7.json -o -name summary.json \))
    fi
    path="reports/libero_plus_language/${run}/seed7/summary.json"
    if [[ -f "${path}" ]]; then git add -- "${path}"; fi
  done
  if ! git diff --cached --quiet; then
    # Abort publication if another job staged unrelated changes.
    if git diff --cached --name-only | awk '$0 !~ /^reports\/(attention\/v1\/|validation\/(teacher_prefix_residual_r16_30k|teacher_prefix_residual_r16_30k_residual_off|attention_fm_selected_v1|attention_teacher_selected_v1)\/step_[0-9]+\/(libero_goal_seed7.json$|libero_para_seed7\/summary.json$)|libero_plus_language\/(teacher_prefix_residual_r16_30k|teacher_prefix_residual_r16_30k_residual_off|attention_fm_selected_v1|attention_teacher_selected_v1)\/seed7\/summary.json$|ALL_EXPERIMENTS_METHODS_AND_RESULTS.md$|all_language_experiments.csv$)/ {bad=1} END {exit bad}'; then
      git commit -m 'Report FM-first attention intervention results'
      git push cold_jeans HEAD:main || state COMPLETED_LOCAL_REMOTE_PUSH_PENDING
    else
      state COMPLETED_LOCAL_UNRELATED_STAGED_FILES_SKIP_COMMIT
    fi
  fi
fi
