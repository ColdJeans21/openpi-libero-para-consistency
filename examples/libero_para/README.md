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

Official benchmark: <https://github.com/cau-hai-lab/LIBERO-Para>.
