# LIBERO same-state language/action triplets

This pilot dataset is generated from the frozen legal counterfactual task pairs
in `examples/libero_attention/manifest_v1.json`.

For every retained teacher rollout state it stores:

- `(I, L, a)`: the executed canonical instruction and teacher action chunk;
- `(I, L+, a)`: an independently authored paraphrase with the same label;
- `(I, L-, a-)`: the other valid goal in the exact same physics state, queried
  from the same frozen teacher with the same flow noise.

Only successful anchor rollouts that never satisfy the wrong goal are retained.
The negative action is a same-state teacher pseudo-label, not a human
demonstration; all reports and papers must disclose this distinction.

The 30 frozen dev/test initial states per task pair are excluded. The remaining
20 states are training-only. This is a mechanism-validation pilot, not yet a
broad benchmark.
