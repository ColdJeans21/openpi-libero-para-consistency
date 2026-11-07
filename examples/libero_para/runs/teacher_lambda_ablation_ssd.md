# Stop-gradient teacher lambda ablation on the new SSD

## Scientific question

Does the behavioral gain come from the directed canonical-teacher objective,
and is it robust to the consistency-loss scale rather than a lucky choice of
`lambda=1`?

## Objective

```text
L_total = L_FM(v_canonical, u)
        + lambda * MSE(stop_gradient(v_canonical), v_paraphrase)
```

Both branches share the observation, action label, image augmentation, noise,
flow time, and flow coordinate.  Only the canonical branch receives direct
flow-matching supervision.

## Runs

| Run | Lambda | Role |
|---|---:|---|
| `canonical_30k` | 0 | Existing no-consistency lower bound |
| `teacher_consistency_lam01_30k` | 0.1 | New weak-consistency run |
| `teacher_consistency_lam1_30k` | 1 | Existing provisional best |
| `teacher_consistency_lam10_30k` | 10 | New over-constraint/stability run |

The two new runs use the same PI0 base checkpoint, 60 train-only paraphrases,
LoRA adapters on the VLM and action expert, global batch four, two-device FSDP,
seed 42, optimizer, action horizon ten, and 30k updates as the lambda-one run.

## Storage and identification

The real checkpoint roots are:

```text
/data/openpi_checkpoints/pi0_libero_goal_lora_consistency/teacher_consistency_lam01_30k
/data/openpi_checkpoints/pi0_libero_goal_lora_consistency/teacher_consistency_lam10_30k
```

Repository-local paths with the same run names are symbolic links to the new
SSD.  W&B, logs, validation reports, and checkpoint directories all use the
exact run name above.  The orchestrator refuses to overwrite either target.

At launch on 2026-11-07, the configured W&B API returned HTTP 403 before the
first training step.  Training therefore defaults to W&B offline mode: all
loss components, gradient/parameter norms, initial-batch images, run id, and
step history are retained locally and can be uploaded with `wandb sync` after
authentication is restored.  This does not change training computation.

## Validation and decision rule

Checkpoints 10000, 20000, and 29999 receive the same seed-7 matched
LIBERO-Goal and fixed 100-instruction LIBERO-Para evaluation.  A final model is
eligible only if Goal success is at least 87%.  Among eligible settings,
LIBERO-Para is the development metric.  LIBERO-Plus remains an external
language-generalization evaluation and is not used for repeated tuning.

If both boundary settings underperform lambda one, the sweep stops.  If a
boundary is best, add only the adjacent interpolation point (`0.3` or `3`).
