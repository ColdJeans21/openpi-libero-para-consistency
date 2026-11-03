# Controlled experiment v1: canonical vs augmentation vs consistency

## Question

Does explicit flow-field consistency under semantically equivalent prompts
improve held-out paraphrase robustness beyond the effect of simply training on
both prompt variants?

All three runs use the same π0 base initialization, ten LIBERO-Goal tasks,
428 demonstrations, source batch size four, action horizon ten, LoRA adapters
on both experts, two-device FSDP, optimizer, seed, and 30k update schedule.
Paired runs duplicate the language-conditioned branch while sharing the
observation, action label, image augmentation, flow noise, and flow time.

| Group | Run | Prompt branches | `lambda_cons` | W&B |
|---|---|---:|---:|---|
| A | `canonical_30k` | 1 | — | `qc7y5gh0` |
| B | `para_aug_lam0_30k` | 2 | 0 | `oad2l00i` |
| C | `consistency_lam01_30k` | 2 | 0.1 | `mmc3rn50` |

## Matched checkpoint results

Every cell uses seed 7, 100 LIBERO-Goal episodes, and the same fixed
100-instruction LIBERO-Para manifest.

| Step | A Goal | A Para | A Gap | B Goal | B Para | B Gap | C Goal | C Para | C Gap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10000 | 35% | 11% | 24 pp | 36% | 15% | 21 pp | 39% | 14% | 25 pp |
| 20000 | 50% | 7% | 43 pp | 41% | 13% | 28 pp | 44% | 21% | 23 pp |
| 29999 | **84%** | **21%** | **63 pp** | **89%** | **50%** | **39 pp** | **84%** | **43%** | **41 pp** |

## What the controls establish

At the final checkpoint:

- B−A is +5 pp Goal, +29 pp Para, and −24 pp gap. Training-time
  paraphrase augmentation is the dominant observed robustness intervention.
- C−B is −5 pp Goal, −7 pp Para, and +2 pp gap. The tested explicit
  `lambda_cons=0.1` term does not improve over matched augmentation.
- The diagnostic consistency loss naturally falls to approximately `2e-4`
  while the FM loss is approximately `2e-2`. At weight 0.1, the explicit term
  contributes only about 0.1% of the total objective, so it may be too weak or
  redundant with the shared action target.

The updated insight is therefore narrower and testable: behavior-level
invariance remains the right target, but paired action supervision already
induces substantial implicit invariance. An explicit regularizer can only add
value if its scale and the model's trainable degrees of freedom are sufficient.

## Claim boundary and next ablations

This is one training seed and a fixed 100-episode validation subset. The large
B−A Para difference motivates continued study, but C−B requires more seeds
before a small-gap claim. The next sequence is:

1. train paired LoRA with `lambda_cons=1`;
2. train paired LoRA with `lambda_cons=10`;
3. compare LoRA with action-expert-only and unfrozen/full tuning;
4. rerun selected conditions on multiple seeds and the complete benchmark;
5. add semantic-change sensitivity only after the positive-invariance question
   is resolved.

Raw summaries are stored under `reports/validation/{canonical_30k,
para_aug_lam0_30k}/`; the λ=0.1 summaries are under the original isolated
validation directories. All checkpoint metrics were backfilled to their
training W&B runs with `validation/checkpoint_step` as the custom x-axis.
