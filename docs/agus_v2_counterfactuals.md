# AGUS v2: Counterfactual Branching Episodes

## Why Branching Counterfactuals Matter

AGUS v1 already tests whether a model can adapt along one observed trajectory. AGUS v2 asks a harder question:

**does the model maintain a coherent internal model of the task across nearby alternate futures?**

This matters because a model can sometimes look competent on a single trajectory by reacting locally to prompts or by memorizing a shallow revision pattern. Counterfactual branching raises the bar. The model must handle several closely related continuations that share the same base episode but vary one critical factor.

## What AGUS v2 Adds Beyond Ordinary Interactive Evaluation

In AGUS v2, one base task can generate multiple branches such as:

- contradiction appears versus does not appear
- one representation shift versus a nearby alternative remapping
- one agent has private information versus everyone sees the same event

These branches stay tightly controlled. Most of the episode is shared. Only one decisive factor changes.

That makes it possible to measure:

- whether the model updates when it should
- whether it preserves invariants when the underlying structure stays the same
- whether nearby alternate futures produce coherent but distinct belief states
- whether confidence tracks branch difficulty and ambiguity

## Supported Branching Families

The initial AGUS v2 implementation supports:

- `hidden_rule`
- `shift_transfer`
- `social_miniworlds`

### Hidden Rule

The same induction phase can lead to:

- a confirming continuation where the original rule still holds
- a contradicting continuation where the rule must be revised

This tests whether the model over-revises or under-revises when contradiction pressure changes.

### Shift and Transfer

The same learned source rule can continue through:

- one transfer codebook
- a nearby alternative transfer codebook

This tests whether the model preserves latent structure across different surface remappings rather than anchoring to one familiar transfer pattern.

### Social Miniworlds

The same social setup can continue through:

- a private-information branch where one agent sees the move
- a public-information branch where everyone sees the move

This tests whether the model keeps world state fixed while changing only the appropriate belief state.

## New Metrics

AGUS v2 adds five branch-aware metrics:

- `counterfactual_update_fidelity`
- `invariant_preservation_score`
- `branch_belief_coherence`
- `cross_branch_consistency`
- `counterfactual_confidence_calibration`

These are intended to measure coherent adaptation across nearby alternate futures, not just performance on one branch in isolation.

## Why This Is More AGI-Relevant

A generally capable system should not merely survive one trajectory. It should carry a structured model of the task that remains stable when it should remain stable and changes when it should change.

That is the clearest AGUS v2 claim:

**counterfactual branching evaluates whether a model reasons from a coherent task model rather than merely following one observed path.**

## Current Artifacts

Branch bundles are written under:

- `data/generated/branches/`

Per-run evaluation outputs are written under:

- `data/evals/<run_name>/counterfactual_summary.json`
- `data/evals/<run_name>/counterfactual_highlights.md`

The demo CLI is:

```bash
PYTHONPATH=. python -m src.cli.run_counterfactual_branches \
  --adapter mock-shallow \
  --run-name agus_v2_counterfactual_demo \
  --max-per-family 1
```
