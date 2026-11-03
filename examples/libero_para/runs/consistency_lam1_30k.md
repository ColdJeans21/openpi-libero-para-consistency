# `consistency_lam1_30k` run record

## Identity

- Started: 2026-11-03 14:00 Asia/Shanghai
- Git commit at launch: `d6ebf3a`
- Base config: `pi0_libero_goal_lora_consistency`
- CLI override: `model.consistency_loss_weight=1.0`
- Experiment name: `consistency_lam1_30k`
- W&B run: `sqvt7xqw`
- W&B URL: <https://wandb.ai/xzzsryouxiwang-shenzhen-university/openpi/runs/sqvt7xqw>

## Purpose

The first controlled experiment found that paired-prompt augmentation at
`lambda=0` reached 89% Goal / 50% Para, while `lambda=0.1` reached 84% / 43%.
The diagnostic consistency loss was about two orders of magnitude below the FM
loss, so the weighted consistency contribution was only about 0.1% of the
total objective. This run increases the explicit weight by 10x to test whether
the previous negative result was caused by insufficient loss scale.

All other variables match the λ=0 and λ=0.1 paired-prompt runs: π0 base,
428 LIBERO-Goal demonstrations over ten tasks, source batch four, two language
branches with shared flow coordinates, LoRA on both experts, two-device FSDP,
action horizon ten, seed 42, optimizer, and 30k updates.

## Launch command

```bash
CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_libero_goal_lora_consistency \
  --model.consistency-loss-weight 1.0 \
  --exp-name consistency_lam1_30k \
  --keep-period 10000 \
  --overwrite
```

`keep_period=10000` retains steps 10000, 20000, and 29999 for matched
closed-loop Goal/Para validation. The existing π0 test suite passed 8/8 before
launch. The run was started as a detached process with training output at
`logs/train_consistency_lam1_30k.log`.

## Evaluation plan

After training, run the same seed-7 evaluation used by A/B/C:

- LIBERO-Goal: 10 trials per task, 100 episodes total;
- LIBERO-Para: checked-in fixed 100-instruction manifest;
- backfill Goal SR, Para SR, and Goal−Para gap to W&B at each checkpoint step.

The λ=10 run will start only after this run and its validation finish, avoiding
GPU contention and allowing λ=1 to act as the decision point for the stronger
constraint.
