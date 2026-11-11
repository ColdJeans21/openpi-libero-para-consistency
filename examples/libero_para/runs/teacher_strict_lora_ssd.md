# Teacher consistency: strict LoRA control

## Question

When SigLIP and both transformer bases are held fixed, does constraining the
Action Expert to LoRA recover the language generalization lost by full Expert
finetuning?

## Controlled setup

- Loss: `FM(v_canonical, u) + MSE(stop_gradient(v_canonical), v_paraphrase)`.
- Canonical and paraphrase branches share flow noise and time.
- Frozen: complete SigLIP, Gemma-2B base, and Gemma-300M Action Expert base.
- Trainable: Gemma-2B LoRA, Action Expert LoRA, action/state/time projections.
- Trainable parameters: 53,235,744.
- Lambda 1, seed 42, global batch 4, two-GPU FSDP, optimizer, schedule,
  normalization, data, and 30k steps match the parameter-scope ablations.

This is the direct control for `VLM LoRA + full Action Expert`: the only
material change is replacing full Expert updates with Expert LoRA updates.

## Evaluation

Checkpoints 10k, 20k, and 29999 receive matched seed-7 LIBERO-Goal and the
fixed 100-instruction LIBERO-Para evaluation. The final checkpoint also runs
the official 410-case LIBERO-Plus Language evaluation.
