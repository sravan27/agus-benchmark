# Research Note: AGUS v1

## Working Hypothesis

Frontier models are often strong on static benchmark questions but weaker when a task requires **online model-building**: learning from few examples, detecting that a latent rule changed, revising beliefs after contradiction, and transferring structure under representational shift.

AGUS is built to probe that gap.

## Research Claim

If a benchmark is intended to speak to cognitive ability rather than pattern familiarity, it should score not only final answers but also:

- how quickly a system adapts
- whether it knows when it is uncertain
- whether it revises instead of rationalizing
- whether it can separate signal from distractor noise

## v1 Contribution

AGUS v1 contributes a coherent synthetic benchmark family organized around **adaptive generalization under shift**. The suite is compact enough to inspect manually, but structured enough to support systematic evaluation.

Implemented families:

1. hidden rule induction with shift
2. cross-representation transfer
3. metacognitive revision under ambiguity
4. distractor-robust inference
5. social miniworlds with false belief and incentive inference

## Why It Could Matter for Kaggle

This framing is useful competitively because it creates a benchmark identity that is not reducible to one more reasoning dataset. It supports a stronger narrative:

- the benchmark targets dynamic cognition
- the task families are cognitively differentiated
- the metrics go beyond answer-only scoring
- the synthetic design is scalable and extensible

## Highest-Leverage Next Research Step

The strongest next move is to add a **lightweight interaction protocol** that lets models receive feedback during evaluation. This would turn AGUS from a static export suite into a more faithful online-learning benchmark while preserving inspectability.
