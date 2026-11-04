# `consistency_lam10_30k` run record

## Identity

- Started: 2026-11-04 17:11 Asia/Shanghai
- Git commit at launch: `3cf59f9`
- Base config: `pi0_libero_goal_lora_consistency`
- CLI override: `model.consistency_loss_weight=10.0`
- Experiment name: `consistency_lam10_30k`
- W&B run: `e8n9mfle`
- W&B URL: <https://wandb.ai/xzzsryouxiwang-shenzhen-university/openpi/runs/e8n9mfle>

## Purpose

This is the final strong-weight stress test of the original symmetric paired-FM
objective. Because paired FM already contains an implicit coefficient 0.25 on
the velocity difference, lambda 10 raises the effective coefficient to 10.25,
or 41x the lambda-zero control. All data, seed, optimizer, LoRA, FSDP, batch,
horizon, and 30k-step settings match the earlier paired runs.

The run keeps steps 10000, 20000, and 29999 for the fixed seed-7 Goal/Para
validation. After validation, the sequence proceeds to the isolated
`teacher_consistency_lam1_30k` objective only if this run completes cleanly.
