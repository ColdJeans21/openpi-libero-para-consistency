# Verification before queue submission — 2026-11-15

- CPU/JAX regression suite: **43 passed**. Covers instruction-bias boundaries,
  full/cached Euler equivalence and unchanged parameter trees, original Pi0
  freeze/loss rules, semantic-interface v1, preserved-prefix residual v2,
  controlled-noise server adapter, counterfactual protocol, report extraction,
  and existing validation/W&B parsing. No GPU was allocated.
- After the final report/builder edits, the focused protocol/adapter/report
  subset passed again: **8 passed** (a subset, not8 additional independent tests).
- Ruff check passed; Bash syntax passed for both new queue/runner scripts and
  both modified standard benchmark launchers; staged diff whitespace passed.
- A clean manifest rebuild to a temporary path exactly matched the frozen
  manifest SHA256
  `45b1cbdcafb6e1f7dfb76670f7119f4685cb4e72b4799a7b1e75c10a64f3c334`.
  Selected paraphrases are absent from the training paraphrase strings.
- Existing `libero:latest` image
  `sha256:7cd8c1bc5347298f4d476fb05f44e3bec797ac061338df2db00c5c377678eccd`
  ran the new evaluator's `--help` successfully with no GPU attached. The
  evaluator and shared protocol also parse under Python3.8 syntax.
- **Pending, first in GPU queue:** real FM checkpoint inference/reload smoke,
  beta-zero exact-action restoration, frozen-parameter identity, nonzero-beta
  finite actions, and actual MuJoCo physics/image/state matching. These are
  deliberately not marked passed while the current training uses both GPUs.
- The active residual runner itself was not edited or interrupted. The default
  benchmark launchers retain the original server branch unless ATTENTION_MODEL
  is explicitly set. Five unrelated existing Docker-related edits are excluded
  from this commit.

Reproduce the CPU suite:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu .venv/bin/pytest -q \
  src/openpi/models/instruction_attention_test.py \
  src/openpi/evaluation/attention_protocol_test.py \
  src/openpi/models/semantic_residual_test.py \
  src/openpi/models/semantic_interface_test.py \
  src/openpi/models/pi0_test.py \
  scripts/serve_attention_policy_test.py \
  scripts/summarize_language_experiments_test.py \
  scripts/log_validation_to_wandb_test.py
```
