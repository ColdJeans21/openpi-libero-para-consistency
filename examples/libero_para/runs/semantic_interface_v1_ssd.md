# State-dependent rank-16 semantic interface: canonical FM vs Teacher consistency

## Hypothesis and limits

Factorizing velocity as `v_base(o,s,x,t) + U(o,s,x,t) z(L)` may reduce sensitivity
to surface wording while preserving instruction-dependent control. This is an
unvalidated architectural hypothesis, not an established novelty claim.
This first experiment uses LIBERO-Goal and cannot establish cross-task-family
generality or counterfactual success without additional evaluation.

## Locked comparison

| Run | Config | Objective |
| --- | --- | --- |
| semantic_interface_r16_fm_30k | pi0_libero_goal_semantic_fm | FM(canonical) only |
| semantic_interface_r16_teacher_lam1_30k | pi0_libero_goal_semantic_teacher | FM(canonical) + MSE(sg(canonical), paraphrase) |

- Both initialize independently from the same official `pi0_base`, seed 42.
- Both use the same Goal subset, training-only paraphrases, image augmentation,
  normalization and delta transform, batch 4, 30k steps, two-GPU FSDP, optimizer,
  LR schedule, EMA disabled, and rank 16.
- Train **complete SigLIP**, Gemma-2B LoRA, Expert LoRA, existing projection
  layers, and the two semantic heads. Transformer base weights stay frozen.
- Trainable parameters: 468,580,640, including 541,200 newly initialized head parameters.
- Action horizon 10; internal dimension 32; environment dimension 7.
- T5-large is frozen, eval mode, contextual mask-mean pooling, float32, max 256
  tokens with an error instead of truncation. Its 1024-dimensional feature is
  projected linearly to 16 dimensions and L2-normalized.
- `U` is Linear(1024, 32*16) on each final action hidden token, zero initialized.
- Neutral prefix IDs `[2,34688,573,6911,108]` encode `perform the task` plus BOS
  and newline using the repository PaliGemma tokenizer. Model code ignores
  supplied task token IDs, masks, and lengths to exclude raw-language bypasses.
- Canonical and paraphrase share one B-sized trunk forward, identical augmented
  observations, flow noise/time/target, and one backward pass. FM control also
  computes paraphrase diagnostics but gives them zero training weight.
- Paired Teacher retains its existing stop-gradient semantics, including student
  gradients through shared trunk and U. Gradients are not restricted to text heads.

## Frozen language assets and checkpoint portability

The cache builder reads only `train_paraphrases_v1.json` (70 training texts),
never evaluation instructions. Encoder files, source text, and feature assets
are SHA-256 identified. Each checkpoint contains `assets/semantic_features.npz`.
Unseen deployment instructions are encoded locally on CPU and cached in RAM;
they are never appended to the training cache. Set `OPENPI_TEXT_MODEL_DIR` to
the exact hash-matching T5 directory when moving a checkpoint to another host.
Encoder weights are stored on the SSD, not committed to Git.

## Safety and records

Runner: `scripts/run_semantic_interface_ssd.sh`; phases are `smoke`, `full`, `all`.
The `all` phase requires both real-data smoke runs and policy reload tests to
pass before starting either 30k run. Existing experiment directories are never
overwritten. Run labels, config, code commit, frozen asset hashes, smoke GPU
peaks, and launch command are recorded under each run on `/data`.

## Evaluation and metrics

Goal/Para: checkpoints 10k, 20k, 29999, matched seed 7, 100 episodes each.
Plus Language: final checkpoint, fixed official 410 cases, same protocol as
earlier ablations. Checkpoint evaluations run after each training run, and
Goal/Para results are backfilled to W&B using `validation/checkpoint_step`.
They are not concurrent live rollout validation during gradient updates.
Final Plus metrics and per-difficulty rates are also appended to the same W&B run.

W&B training: loss, canonical/para FM diagnostics, consistency, gradient/parameter
norms; `semantic/code_distance`, `semantic/base_rms`, `semantic/residual_rms`,
and `semantic/basis_rms`. A successful smoke test is not evidence of task success.
Inferential results remain single-seed exploratory results.
