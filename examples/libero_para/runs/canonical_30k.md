# `canonical_30k` run record

## Identity

- Started: 2026-11-02 19:59 Asia/Shanghai
- Git commit at launch: `541377f`
- Config: `pi0_libero_goal_lora_canonical`
- Experiment name: `canonical_30k`
- W&B run: `qc7y5gh0`
- W&B URL: <https://wandb.ai/xzzsryouxiwang-shenzhen-university/openpi/runs/qc7y5gh0>

## Purpose

This is the canonical-prompt-only baseline for the first LIBERO-Para
consistency experiment. It measures the natural paraphrase generalization of
the same π0 LoRA recipe before either training-time paraphrase augmentation or
an explicit flow-field consistency term is added.

It matches the paired-prompt runs in:

- the same 428 LIBERO-Goal episodes across ten tasks;
- source batch size 4, seed 42, and 30k optimizer steps;
- π0 base initialization and LoRA parameterization of both language and action
  experts;
- two-device FSDP, action horizon 10, optimizer, and Goal-only norm stats.

The only intended training difference is that each trajectory uses its single
canonical task prompt. Consequently, this baseline has half the number of
language-conditioned examples per step and less sample-level compute than the
paired runs; it is the correct control for the total effect of paraphrase
augmentation, while the `lambda=0` versus `lambda=0.1` comparison isolates the
explicit consistency term.

## Launch command

```bash
CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py \
  pi0_libero_goal_lora_canonical \
  --exp-name canonical_30k \
  --keep-period 10000 \
  --overwrite
```

`keep_period=10000` retains steps 10000, 20000, and 29999 for direct comparison
with both paired-prompt runs while limiting additional storage to roughly 27GB.

## Evaluation plan

After training, evaluate all three retained checkpoints with the same seed-7
protocol:

- LIBERO-Goal: 10 trials per task, 100 episodes total.
- LIBERO-Para: the checked-in fixed 100-instruction manifest.
- Backfill Goal SR, Para SR, and Goal−Para gap to W&B with checkpoint step as
  the validation x-axis.

The primary A/B comparison is canonical-only versus `para_aug_lam0_30k`; the
B/C comparison is `para_aug_lam0_30k` versus `consistency_lam01_30k`.

## Training completion

- Completed: 2026-11-03 08:33 Asia/Shanghai.
- Final checkpoint: step 29999; retained checkpoints: 10000, 20000, 29999.
- Final logged metrics at step 29900: total/original FM loss 0.0229,
  gradient norm 0.5958, and parameter norm 1380.5803.
- No OOM, NaN, checkpoint-save error, or abnormal termination was observed.

Matched closed-loop validation ran with
`scripts/run_libero_goal_para_validation.sh`. Each checkpoint used seed 7,
100 LIBERO-Goal episodes, and the same fixed 100-instruction LIBERO-Para
manifest. All results were backfilled to W&B run `qc7y5gh0`.

## Matched closed-loop validation results

| Step | Goal SR | Para SR | Goal−Para gap |
|---:|---:|---:|---:|
| 10000 | 35% | 11% | 24 pp |
| 20000 | 50% | 7% | 43 pp |
| 29999 | **84%** | **21%** | **63 pp** |

At 30k, the matched `lambda=0` paired-prompt augmentation control reaches
89% Goal / 50% Para, improving this canonical-only baseline by +5 pp Goal and
+29 pp Para while reducing the paraphrase gap by 24 pp. The `lambda=0.1`
consistency run reaches 84% Goal / 43% Para, which is 5 pp Goal and 7 pp Para
below the augmentation-only control. Thus the first controlled experiment
supports training-time paraphrase augmentation, but provides no evidence that
the tested explicit consistency weight adds value beyond shared action
supervision. These are single-seed, 100-episode validation estimates and are
not a substitute for multi-seed full-benchmark evaluation.
