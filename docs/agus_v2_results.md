# AGUS v2 Results

## Counterfactual Setup

AGUS v2 adds **counterfactual branching episodes** on top of AGUS interactive evaluation. One base episode is expanded into a small set of closely related branches where only one decisive factor changes.

Current supported branching families are:

- `hidden_rule`
- `shift_transfer`
- `social_miniworlds`

Example branch differences include:

- contradiction appears versus does not appear
- one transfer codebook versus a nearby alternative remapping
- one agent has private information versus everyone sees the same event

## What AGUS v2 Measures

The first branch-aware metrics are:

- `counterfactual_update_fidelity`
- `invariant_preservation_score`
- `branch_belief_coherence`
- `cross_branch_consistency`
- `counterfactual_confidence_calibration`

These are intended to measure whether a model carries a coherent task model across nearby alternate futures, not just whether it can follow one observed trajectory.

## Current Local Results

| Model | Update Fidelity | Invariant Preservation | Branch Belief Coherence | Cross-Branch Consistency | Confidence Calibration |
| --- | ---: | ---: | ---: | ---: | ---: |
| `llama3.1:8b` | 0.7222 | 0.7500 | 0.7037 | 1.0000 | 0.9561 |
| `qwen2.5:7b` | 0.8333 | 0.8438 | 0.8542 | 1.0000 | 0.9717 |

The current AGUS v2 pattern is consistent with AGUS v1:

- Llama remains stronger on frozen-task accuracy
- Qwen remains stronger on adaptive and coherence-oriented metrics

## Why This Matters

Ordinary interactive evaluation asks whether a model updates along one path. AGUS v2 asks whether the same model:

- updates only when contradiction truly appears
- preserves invariants across nearby transfer variants
- keeps world state stable while changing only the appropriate belief state

That is a tighter test of coherent adaptive reasoning.

## Limitations

- These are still local-model results from a small two-model comparison.
- The AGUS v2 result set should be used as evidence of benchmark promise, not as a broad leaderboard claim.
- `cross_branch_consistency` being `1.0` for both models is encouraging, but it should not be read as proof that the branching problem is solved. The more informative differences are in update fidelity, invariant preservation, and branch belief coherence.
