# `teacher_consistency_lam1_30k` planned run

## Question

Can explicit prompt invariance improve held-out paraphrase behavior when it is
isolated from the implicit consistency term created by duplicate FM targets?

## Objective

```text
L_total = L_FM(v_canonical, u)
        + lambda * MSE(stop_gradient(v_canonical), v_paraphrase)
```

The canonical prediction remains anchored to the expert flow target. The
paraphrase branch receives no direct duplicate FM target and learns from the
stop-gradient canonical velocity at the same observation, image augmentation,
noise, flow time, and flow coordinate. The first run uses lambda 1.0.

## Controlled variables

The dataset, 60 train-only paraphrases, π0 base initialization, LoRA adapters
on both experts, source batch four, two-device FSDP, optimizer, seed 42,
action horizon ten, and 30k updates match the symmetric paired runs. Steps
10000, 20000, and 29999 will use the same seed-7 100-episode Goal and fixed
100-instruction Para validation.

The run is queued after lambda=10 training and validation; it must not overlap
that job on the two RTX 3090 Ti GPUs.
