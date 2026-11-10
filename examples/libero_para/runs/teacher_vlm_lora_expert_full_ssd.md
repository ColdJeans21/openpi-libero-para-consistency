# Teacher consistency: VLM LoRA + full Action Expert

## Question

Does language-side LoRA adaptation recover the performance lost when the
complete Action Expert is finetuned, or is the Expert's low-rank constraint
itself necessary for language generalization?

## Controlled setup

- Objective: `FM(v_canonical, u) + MSE(stop_gradient(v_canonical), v_paraphrase)`.
- Both language branches share exactly the same flow noise and time.
- Frozen: complete SigLIP image encoder and Gemma-2B base weights.
- Trainable: Gemma-2B LoRA adapters, full Gemma-300M Action Expert, and all
  action/state/time projections.
- Trainable parameters: 342,582,304 (27,869,184 more than Expert-only).
- Lambda 1, seed 42, two-device FSDP, data, optimizer, schedule, norm stats,
  and 30k-step budget match the preceding Teacher experiments.

This scope is intentionally stricter than OpenPI's default LoRA freeze filter,
which leaves SigLIP trainable. Freezing SigLIP isolates language adaptation
from visual-encoder capacity.

## Memory gate

A 20-step global-batch-4 smoke test runs first. It must complete below 23,500
MiB per GPU. If it fails or exceeds the threshold, the runner performs one
global-batch-2 smoke test and records the selected batch size. No full run is
started without a passing smoke test.

## Evaluation

The 10k, 20k, and 29999 checkpoints receive matched seed-7 LIBERO-Goal and
fixed 100-instruction LIBERO-Para evaluation. The final checkpoint also
receives the official 410-case LIBERO-Plus Language evaluation.
