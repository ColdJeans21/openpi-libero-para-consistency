# Teacher consistency: full Action Expert parameter-scope ablation

## Question

Can semantic paraphrase invariance be absorbed entirely by the action expert,
without adapting SigLIP or the Gemma-2B vision-language prefix?

## Controlled setup

- Objective: `FM(v_canonical, u) + 1.0 * MSE(stop_gradient(v_canonical), v_paraphrase)`.
- Shared flow coordinate: both prompt branches use identical noise and time.
- Frozen: complete SigLIP image encoder and complete Gemma-2B/PaliGemma prefix.
- Trainable: full Gemma-300M action expert, `action_in_proj`,
  `action_out_proj`, `state_proj`, and both action-time MLP projections.
- No LoRA modules are instantiated in either transformer.
- Trainable parameters: 314,713,120.
- Data, seed 42, global batch 4, two-device FSDP, optimizer, schedule,
  normalization, and 30k-step budget match the teacher-lambda study.
- Checkpoints live on `/data/openpi_checkpoints`; repository paths are symlinks.

## Safety gate

A 20-step smoke run must finish at global batch 4 and peak below 23,500 MiB
per GPU before the 30k run starts. The sampled GPU memory trace is retained
with the smoke checkpoint. A failed gate stops the queue rather than silently
changing batch size.

## Evaluation

Checkpoints 10k, 20k, and 29999 receive matched seed-7 LIBERO-Goal and the
fixed 100-instruction LIBERO-Para evaluation. The final 29999 checkpoint then
receives the full 410-case LIBERO-Plus Language evaluation. The primary
comparison is against teacher consistency lambda=1 with the prior LoRA recipe.
