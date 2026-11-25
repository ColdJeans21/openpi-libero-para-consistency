# Verified CFD v2 — frozen R0/R1/R2 protocol (2026-11-25)

## Hypothesis and controls

Retain complete LIBERO-Goal demonstrations (10 tasks, 428 episodes) and the
existing 60 training paraphrases. Add independently executed, both-successful
same-initial-state teacher pairs. All runs start from pi0_base, same original
architecture, full SigLIP + Gemma LoRA + expert LoRA + projections, batch 4
logical pairs, seed 42, 30k updates, FSDP=2, EMA off, existing normalization.
No new language head. A pair is expanded to 2B branches. Whole trajectories
are not aligned. The pseudo-labels are ten ACTUALLY EXECUTED initial controls.

Dataset: /data/openpi_triplets/verified_initial_pairs_v2. Collection excludes
counterfactual dev/test states; checks exact initial image/state/physics hashes;
both rollouts must finish the right goal without triggering the wrong goal.
Keep original demonstrations; sample seven demonstration/paraphrase pairs per
one counterfactual pair, reversing counterfactual orientation equally. Require
at least 20 accepted states spanning both task families before training. This
is an engineering minimum, not a statistical sufficiency guarantee. Underlying
original demonstrations are not certified state-disjoint from all evaluations.

## Objective

L0 = 0.5*(FM_anchor + FM_alt), averaged over all sampled pairs/time/dimensions.
R0: L0.
R1: L0 + 1.0 * positive-only MSE(v_alt, sg(v_anchor)).
R2: R1 + 0.1 * negative-only MSE(v_alt(eps,1), sg(v_anchor(eps,1)+A-A_alt)).
The MSE spans all padded 32 dimensions to match historical scaling. Labels A
and A_alt are transformed/normalized with the same pipeline as FM targets.
Auxiliaries are zero on the other edge type; no random batch-subset division.

R1 differs from historical E07: it RETAINS direct paraphrase FM. R2 adds an
endpoint forward; compare equal samples/updates and report extra compute,
not equal FLOPs. Random-tau negative FM uses two different interpolated inputs;
only the endpoint differential has identical x_tau. Teacher and differential
stop gradients on the target prediction only, not on shared model parameters.

## Execution and reporting

1. Run collection and verify manifest/hashes/splits/finite labels.
2. Unit tests: signs, common-offset FM anchor, gradient direction, edge masks,
   normalized-label routing, 7:1 mixture, regression tests.
3. Real-data 20-step smoke for each config; require finite loss/gradients and a
   valid final checkpoint. Never treat CPU formula tests as a GPU smoke.
4. R0, R1, R2 30k in sequence. Store checkpoints under /data/openpi_checkpoints;
   mirror paths with local symlinks for existing evaluators. Log configuration,
   code revision/patch, immutable dataset manifest, source hashes and GPU info.
5. At final checkpoint29999 run Goal100, Para100, Plus language410 (same seed7
   and historical protocol). Update local Markdown/CSV after each complete run.
   Missing/incomplete results remain blank. W&B uploads enabled as previously
   authorized. Reports/results/paper are local only; source code may be pushed.
6. Save training metrics (FM original/alt/mean, positive teacher, negative
   differential, negative fraction, total, gradients, learning rate as existing
   trainer supports). No conclusions about goal switching without paired
   closed-loop counterfactual testing beyond these historical benchmarks.

Fixed names: pi0_libero_goal_verified_r{0,1,2}; verified_r{0,1,2}_30k.
Do not tune on Plus. Lambda-0.1 is an initial auxiliary weight, not a known
optimum. Record negative results and cost. No multi-seed extension in this run.
