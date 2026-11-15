# Preserved PaliGemma language prefix + semantic residual: matched Teacher experiment

## Question

Does a zero-initialized rank-16 semantic residual improve language robustness
when the original PaliGemma instruction pathway is retained? This experiment
tests an architectural hypothesis; it does not presume success or novelty.

## Locked runs

| Run | Config | Architecture |
| --- | --- | --- |
| teacher_prefix_control_30k | pi0_libero_goal_teacher_prefix_control | Original Pi0, no extra semantic head |
| teacher_prefix_residual_r16_30k | pi0_libero_goal_teacher_semantic_residual | Original Pi0 + rank16 velocity residual |

Both independently start from the official pi0_base, not the earlier failed
factorized policy or a finetuned Teacher. Seed42, global batch4, two-device
FSDP, 30k updates, identical optimizer/LR schedule, Goal428 demonstrations,
60 train-only paraphrases, norm statistics/delta transforms, horizon10 and
EMA disabled. The primary objective in BOTH runs is
`FM(v_c,u) + MSE(stop_gradient(v_c), v_p)`, lambda1.

Train complete SigLIP, Gemma-2B LoRA, Expert LoRA and interface projections.
Control trainable count468,039,440; residual count468,580,640. Transformer
base weights remain frozen. Neither config is strict pure LoRA.

## Architecture and initialization

- Actual instruction tokens, masks and positions are retained in PaliGemma.
- Original vision/state/time/action paths and action_out_proj are unchanged.
- Frozen T5-large contextual mean-pool features1024 -> trainable Linear16 ->
  L2 normalization produce z(L). Reuse the exact training-only feature cache.
- U(h) is Linear(1024,32*16) reshaped per future action token. Its kernel and
  bias initialize to zero; the language projection initializes normally.
- Velocity is `action_out_proj(h) + scale * U(h) z(L)`, with fixed scale1.
  At initialization, every original parameter and the velocity/action mapping
  match the original policy. There is no extra zero-valued learnable gate.
- Unlike v1, h depends on the real instruction. Canonical and paraphrase use
  a combined 2B forward after image augmentation; images, noise/time/target
  match, while token inputs and frozen text features differ by branch.
- Teacher supervises FINAL velocity. A separate frozen/EMA teacher is not used.
- Original/neutral-prefix v1 checkpoints keep their previous default behavior.

## Preflight and records

Runner: `scripts/run_semantic_residual_ssd.sh`, phases smoke/full/all.
Both real-data20step smoke tests, checkpoint reloads and inference must pass
before either formal run. Each GPU must stay below23,500MiB; batch is never
silently changed. Shared-parameter/zero-output equivalence, original-language
conditioning, shared flow coordinates, semantic-head gradients and residual
switch behavior are tested separately.

Checkpoints are independently named under `/data/openpi_checkpoints/<config>/<run>`
on the new SSD, with repository symlinks. No existing directory is overwritten.
Code commit, launch patch, manifest, cache SHA256, memory trace and labels are
retained. Model assets are not committed. Remote: cold_jeans/main.

## Evaluation

After each30k training: Goal/Para at10k,20k,29999, seed7,100episodes each;
official Plus Language at29999,410cases. Evaluations run AFTER training and
backfill checkpoint-axis metrics to the training W&B run.

After the residual run, repeat FINAL Goal/Para/Plus using the SAME weights and
`pi0_libero_goal_teacher_semantic_residual_off` (fixed residual scale0). Reports
use `teacher_prefix_residual_r16_30k_residual_off`, and W&B metrics use
`validation_residual_off/*`. These cannot overwrite primary `validation/*`.
This on/off diagnostic does not replace the separately trained control,
because trunk and head may co-adapt during training.

The development criterion stays Goal>=87%, then Para; Plus is external and
not used for repeated selection. All final-step results are reported even if
the new branch hurts. Multiple training seeds and negative-language losses
are deferred. No counterfactual target-switch claim follows from these tests.

## Code verification before GPU smoke (2026-11-15)

50 targeted Python tests passed; the unrelated uncached real ALOHA download
test was deselected. Coverage includes original/v1 compatibility, exact
original-parameter/initial-loss/Euler equivalence at zero residual, real-token
conditioning, shared augmented images and flow coordinates in a2B forward,
semantic-head gradients, inference switch behavior and W&B namespace isolation.
Ruff lint/format and shell syntax checks also passed.
