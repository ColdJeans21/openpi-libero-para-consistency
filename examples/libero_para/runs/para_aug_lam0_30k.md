# `para_aug_lam0_30k` run record

## Identity

- Started: 2026-11-01 15:24 Asia/Shanghai
- Git commit at launch: `14dcc51`
- Config: `pi0_libero_goal_lora_para_aug`
- Experiment name: `para_aug_lam0_30k`
- W&B run: `oad2l00i`
- W&B URL: <https://wandb.ai/xzzsryouxiwang-shenzhen-university/openpi/runs/oad2l00i>

## Purpose

This is the paired-prompt, `lambda_cons=0` control for the first consistency
experiment. It matches the proposed `lambda_cons=0.1` run in dataset, original
and paraphrased prompt branches, shared flow noise/time, source batch size,
optimizer steps, seed, LoRA parameterization, and two-device FSDP. The only
objective difference is that `loss_consistency` is logged but has zero weight.

## Launch command

```bash
CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py \
  pi0_libero_goal_lora_para_aug \
  --exp-name para_aug_lam0_30k \
  --keep-period 10000 \
  --overwrite
```

`keep_period=10000` is a storage-only override. With Orbax `max_to_keep=1`, the
run retains steps 10000, 20000, and 29999 instead of six 5k checkpoints. It does
not change model updates and leaves three steps directly comparable to the
`lambda_cons=0.1` run.

## Launch checks

- Relevant unit tests: 10 passed.
- Selected dataset: 428 LIBERO-Goal episodes across 10 tasks.
- Source batch: 4 trajectories; effective prompt-condition batch: 8.
- Prompt tensors: original `(4, 48)`, paraphrase `(4, 48)`.
- Action target: `(4, 10, 32)`.
- GPU memory after compilation: approximately 22.3 / 22.2 GiB.
- Stable throughput after compilation: approximately 2.3 seconds per step.

First W&B log at step 100:

| Metric | Value |
|---|---:|
| `loss_fm_original` | 0.2610728 |
| `loss_fm_paraphrase` | 0.2522273 |
| `loss_consistency` | 0.0073021 |
| `loss_total` | 0.2566501 |
| `grad_norm` | 9.8935480 |

The logged total equals
`0.5 * (loss_fm_original + loss_fm_paraphrase)` as required for this control.

## Evaluation plan

After training, evaluate retained checkpoints at steps 10000, 20000, and 29999
on the same seed-7 protocols used by the consistency run:

- LIBERO-Goal: 10 trials per task, 100 episodes total.
- LIBERO-Para validation: the checked-in fixed 100-instruction manifest.
- Backfill Goal SR, Para SR, and paraphrase gap into W&B using checkpoint step as
  the validation x-axis.

## Training completion

- Completed: 2026-11-02 11:07 Asia/Shanghai.
- Final checkpoint: step 29999; retained checkpoints: 10000, 20000, 29999.
- Final logged metrics at step 29900: total loss 0.0212, original FM loss
  0.0211, paraphrase FM loss 0.0213, and diagnostic consistency loss 0.0002.
- No OOM, NaN, checkpoint-save error, or abnormal termination was observed.

Matched closed-loop validation was launched after training with the generic
`scripts/run_libero_goal_para_validation.sh` orchestrator. Results are isolated
under `reports/validation/para_aug_lam0_30k/` so they cannot overwrite the
`lambda_cons=0.1` summaries.

## Matched closed-loop validation results

Validation completed on 2026-11-02 at 17:38 Asia/Shanghai. Every checkpoint
used seed 7, 100 LIBERO-Goal episodes, and the same fixed 100-instruction
LIBERO-Para manifest. All metrics were backfilled to W&B run `oad2l00i`.

| Step | Goal SR | Para SR | Goal−Para gap |
|---:|---:|---:|---:|
| 10000 | 36% | 15% | 21 pp |
| 20000 | 41% | 13% | 28 pp |
| 29999 | **89%** | **50%** | **39 pp** |

For the matched `lambda_cons=0.1` run, the corresponding values were:

| Step | Goal SR | Para SR | Goal−Para gap |
|---:|---:|---:|---:|
| 10000 | 39% | 14% | 25 pp |
| 20000 | 44% | 21% | 23 pp |
| 29999 | **84%** | **43%** | **41 pp** |

At the final checkpoint, the `lambda_cons=0` augmentation control is +5 pp on
Goal and +7 pp on Para, with a 2 pp smaller gap. Therefore this single-seed
validation provides no evidence that `lambda_cons=0.1` improves over matched
paraphrase augmentation. The intermediate checkpoints are non-monotonic, and
100 binary episodes per benchmark are insufficient for a definitive small-gap
claim. The result should be treated as a preliminary negative result until the
canonical-only baseline, additional seeds, and stronger consistency weights
are evaluated.

