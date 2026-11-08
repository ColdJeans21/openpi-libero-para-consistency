# OpenPI evaluation on LIBERO-Plus

The first supported track is the official LIBERO-Plus **Language Instructions**
factor for `libero_goal`.  It is intentionally evaluated without the other six
perturbation factors: prompts come from the official LIBERO-Plus BDDL files,
while observations, initial state zero, and task success predicates come from
the original LIBERO-Goal environments.

This isolates instruction robustness and complements LIBERO-Para:

- LIBERO-Para diagnoses lexical, pragmatic, structural, object, action, and
  compositional rewrites.
- LIBERO-Plus provides 410 official language instances stratified into five
  empirical difficulty levels.

The checkout used for the controlled evaluation is the upstream commit
`4976dc30028e805ff8094b55501d532c48fec182`.

Run a smoke test through the four-checkpoint orchestrator:

```bash
MAX_TASKS=5 bash scripts/run_libero_plus_language_eval.sh
```

Smoke-test outputs are isolated under
`reports/libero_plus_language_smoke_<N>/` and never reused by the full run.

Run the complete 410-instance comparison:

```bash
bash scripts/run_libero_plus_language_eval.sh
```

`RUN_SPECS_OVERRIDE` can restrict the evaluator to newline-delimited custom
entries of the form `run_name|config_name|checkpoint_dir`. The teacher-lambda
ablation uses this interface through
`scripts/run_teacher_lambda_plus_followup.sh`: after the active lambda queue
finishes matched Goal/Para validation, it evaluates the final lambda=0.1 and
lambda=10 checkpoints on all 410 language instances. Lambda=1 reuses its
already completed result in the same report root.

Each model writes an append-only `episodes.jsonl` and an atomic `summary.json`
under `reports/libero_plus_language/<run>/seed7/`.  Interrupted runs resume by
skipping already recorded official task names.  The orchestrator runs
canonical-only, matched paraphrase augmentation, symmetric consistency at
lambda one, and stop-gradient teacher consistency at lambda one sequentially.

This protocol uses the official task classification and the official one
rollout per task, but fixes simulator seed 7 to match the existing controlled
LIBERO-Goal/LIBERO-Para study.  Results should therefore be described as a
paired controlled evaluation, not copied directly into the upstream
leaderboard without first verifying its current seed convention.
