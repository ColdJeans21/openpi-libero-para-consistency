# Same-state triplet language-control pilot

Frozen protocol, 2026-11-16.

## Data

Two legal counterfactual task families share an identical XML, initial state,
camera observation, and robot state. The 30 initial states reserved by the
counterfactual dev/test manifest are never used for training. The other 20
states per family are labeled by the frozen official `pi05_libero` policy.
Only successful anchor rollouts that never satisfy the wrong goal are kept.

Each record is a logical triplet:

1. `(I, L, a)`: canonical instruction and teacher action chunk;
2. `(I, L+, a)`: paraphrase with the same action label;
3. `(I, L-, a-)`: alternate legal goal and its same-state teacher action.

The two action queries use identical initial flow noise. During B/C training,
the three members are flattened into ordinary FM samples. Thus no pairwise
loss and no effective 3B forward are introduced.

## Experiments

| ID | Architecture | Data | Objective | Question |
|---|---|---|---|---|
| A-triplet | Original Pi0 | Same triplets | FM | Matched-data control |
| B | Original + fixed language-only readout | Same triplets | FM | Does separate normalization help? |
| C | Original + dynamically gated language-only readout | Same triplets | FM | Does control-state-dependent language use help? |
| D1 | Best of B/C | Same triplets | FM + unified CFD | Does action-difference supervision add value? |

The language-only readout reuses per-layer Q/K/V but has an independent
softmax over instruction content tokens and a separate zero-initialized output
projection. The original joint visual-language-action attention remains intact.

## Evaluation

- LIBERO-Goal canonical success;
- LIBERO-Para success and paraphrase gap;
- LIBERO-Plus language variants;
- frozen legal counterfactual dev/test: canonical, paraphrase, both-target and
  wrong-goal completion rates;
- offline velocity/action separation diagnostics on the held-out exact-state
  pairs.

No claim beyond this two-family pilot is permitted until more task families and
multiple seeds are added.
