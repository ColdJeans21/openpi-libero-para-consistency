# π0 LoRA paired-prompt consistency on LIBERO-Para

This experiment tests whether matching the π0 flow velocity under two independently
authored, semantically equivalent instructions improves paraphrase robustness without
damaging canonical LIBERO-Goal performance.

For each demonstration, both branches share the observation, expert action, image
augmentation, Gaussian noise, flow time, flow coordinate, and target velocity. Only
the language prompt changes:

\[
L_{total}=\frac{1}{2}(L_{FM}^{original}+L_{FM}^{paraphrase})
+\lambda_{cons}L_{cons},\qquad
L_{cons}=\|v_\theta(x_\tau,\tau,l_1)-v_\theta(x_\tau,\tau,l_2)\|_2^2.
\]

The first version intentionally contains no negative/semantic-sensitivity loss.
Therefore it supports claims about invariance to equivalent instructions only.

## Controlled experiment

| Config | Prompt branches | `lambda_cons` | Purpose |
|---|---:|---:|---|
| `pi0_libero_goal_lora_canonical` | 1 | — | canonical-only baseline |
| `pi0_libero_goal_lora_para_aug` | 2 | 0.0 | controls for paraphrase augmentation and 2B compute |
| `pi0_libero_goal_lora_consistency` | 2 | 0.1 | proposed objective |

All configs use π0 LoRA, the same ten LIBERO-Goal tasks selected directly from the
official 256×256 `physical-intelligence/libero` dataset, horizon 10, source batch 4,
30k optimizer steps, and two-device FSDP. Paired configs have an effective condition
batch of 8 but still contain four distinct robot trajectories per step.

## Training

The verified Goal-only statistics are included at
`assets/pi0_libero_goal_lora_canonical/physical-intelligence/libero/norm_stats.json`.
Recompute them after changing data or action transforms; the other two configs reuse
the canonical config's statistics:

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi0_libero_goal_lora_canonical
```

Run the three controlled trainings:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi0_libero_goal_lora_canonical --exp-name canonical_30k --overwrite

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi0_libero_goal_lora_para_aug --exp-name para_aug_30k --overwrite

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi0_libero_goal_lora_consistency --exp-name consistency_lam01_30k --overwrite
```

Before a full run, use a one-step smoke test on the proposed path and confirm all
four loss values are finite and `loss_consistency` contributes a nonzero gradient:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi0_libero_goal_lora_consistency \
  --exp-name smoke_lam01 --num-train-steps 1 --no-wandb-enabled --overwrite
```

Training logs include `loss_fm_original`, `loss_fm_paraphrase`,
`loss_consistency`, and `loss_total`. If paired batch 4 exceeds memory on two 24GB
GPUs, reduce all three configs to batch 2 for the primary comparison; do not reduce
only the proposed method.

## Train/evaluation separation

`train_paraphrases_v1.json` contains 60 training-only paraphrases authored
independently of the benchmark. Audit normalized exact overlap after cloning the
official benchmark:

```bash
git clone https://github.com/cau-hai-lab/LIBERO-Para.git /path/to/LIBERO-Para
uv run scripts/check_libero_para_leakage.py \
  --libero-para-root /path/to/LIBERO-Para
```

The audit must report zero overlap before training results are used.

## Evaluation

First serve a trained checkpoint from this repository:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_libero_goal_lora_consistency \
  --policy.dir=checkpoints/pi0_libero_goal_lora_consistency/consistency_lam01_30k/29999
```

Then, from the official LIBERO-Para client environment, run the thin wrapper. It
delegates to the upstream evaluator so the 4,092 held-out instructions and JSON log
schema remain unchanged:

```bash
python examples/libero_para/eval_openpi.py \
  --libero-para-root /path/to/LIBERO-Para \
  --model-name pi0-consistency-lam01 \
  --seed 7 \
  --output-dir reports/libero_para/pi0-consistency-lam01/seed7
```

Run canonical LIBERO-Goal and LIBERO-Para for every checkpoint using the same seeds.
The primary report should include canonical success rate, paraphrase success rate,
their absolute gap, and category-level LIBERO-Para results. Use at least three seeds;
five seeds (7–11) matches the upstream analysis layout.

### Checkpoint validation and W&B

Closed-loop validation is checkpoint-level rather than an in-process training loss.
Two-device FSDP occupies both GPUs, so running a second policy server concurrently
would either contend for memory or change training throughput. Evaluate saved
checkpoints while training is paused or after it finishes, then attach the results
to the original W&B run using the checkpoint step as a custom x-axis.

For a training curve, use a fixed, predeclared validation subset at every saved
checkpoint (for example 10 trials per LIBERO-Goal task and a fixed LIBERO-Para
subset). Run the final selected checkpoint on 50 trials per Goal task and all 4,092
LIBERO-Para instructions. Do not compare checkpoints evaluated on different subsets.

The checked-in checkpoint-curve subset contains 100 instructions: ten for each
original Goal task, split into seven compositional, two action, and one object
paraphrase. The seed-7 SHA-256 selection is deterministic and recorded in
`validation_subset_v1_seed7.json`. Rebuild or audit it with:

```bash
uv run scripts/build_libero_para_subset.py \
  --libero-para-root /path/to/LIBERO-Para \
  --output examples/libero_para/validation_subset_v1_seed7.json \
  --seed 7
```

Pass the same manifest to every checkpoint evaluation:

```bash
python examples/libero_para/eval_openpi.py \
  --libero-para-root /path/to/LIBERO-Para \
  --task-manifest examples/libero_para/validation_subset_v1_seed7.json \
  --model-name pi0-consistency-lam01-step5000 \
  --seed 7 \
  --output-dir reports/validation/step_5000/libero_para_seed7
```

Write the canonical summary directly from the LIBERO client:

```bash
python examples/libero/main.py \
  --task-suite-name libero_goal \
  --num-trials-per-task 10 \
  --seed 7 \
  --summary-path reports/validation/step_5000/libero_goal_seed7.json
```

The official LIBERO-Para evaluator writes its own JSON summary. Once both clients
have finished, backfill their closed-loop success rates into the same training run:

```bash
uv run scripts/log_validation_to_wandb.py \
  --checkpoint-dir checkpoints/pi0_libero_goal_lora_consistency/consistency_lam01_30k \
  --checkpoint-step 5000 \
  --libero-goal-summary reports/validation/step_5000/libero_goal_seed7.json \
  --libero-para-summary reports/validation/step_5000/libero_para_seed7/summary.json
```

This logs `validation/libero_goal_success_rate`,
`validation/libero_para_success_rate`, and `validation/paraphrase_gap`. Because
`validation/checkpoint_step` is a custom W&B axis, delayed evaluations remain aligned
with their actual checkpoint instead of being assigned to the latest training step.

Official benchmark: <https://github.com/cau-hai-lab/LIBERO-Para>.

## Controlled experiment v1 results

The first A/B/C experiment is complete. Each checkpoint used seed 7,
100 LIBERO-Goal episodes, and the same fixed 100-instruction LIBERO-Para
manifest.

| Step | Canonical Goal / Para | Augmentation (`lambda=0`) | Consistency (`lambda=0.1`) |
|---:|---:|---:|---:|
| 10000 | 35% / 11% | 36% / 15% | 39% / 14% |
| 20000 | 50% / 7% | 41% / 13% | 44% / 21% |
| 29999 | **84% / 21%** | **89% / 50%** | **84% / 43%** |

At 30k, paired paraphrase augmentation improves the canonical-only baseline by
+29 pp Para, whereas `lambda=0.1` is 7 pp below the matched `lambda=0`
control. This preliminary single-seed result supports augmentation but does not
show an added benefit from the tested explicit consistency weight. See
`runs/controlled_experiment_v1.md` for the full table, claim boundary, and
interpretation.

Run the next loss-scale ablations without adding duplicate static configs:

```bash
uv run scripts/train.py pi0_libero_goal_lora_consistency \
  --model.consistency-loss-weight 1.0 \
  --exp-name consistency_lam1_30k --keep-period 10000 --overwrite

uv run scripts/train.py pi0_libero_goal_lora_consistency \
  --model.consistency-loss-weight 10.0 \
  --exp-name consistency_lam10_30k --keep-period 10000 --overwrite
```
