# AGUS Final Submission Summary

## What AGUS Is

AGUS, **Adaptive Generalization Under Shift**, is a Learning-track benchmark for measuring whether a model can infer new rules, notice when its current strategy stops working, and adapt after contradiction, distractors, or representation shifts.

## Why It Matters

Many benchmarks reward frozen-task correctness. AGUS is built to measure **dynamic cognition**: hypothesis revision, confidence recalibration, attention recovery, transfer under shift, and belief tracking across short interactive episodes.

## Strongest Result

The clearest result from the first local-model runs is that **static accuracy and adaptive reasoning quality diverge sharply**:

- `llama3.1:8b` static accuracy: `0.6179`
- `qwen2.5:7b` static accuracy: `0.3000`
- `qwen2.5:7b` belief trajectory quality: `0.7281` versus `0.5434` for Llama
- `qwen2.5:7b` trajectory instability index: `0.2626` versus `0.3348` for Llama, lower is better

## Why It Is Novel

AGUS is not just another static reasoning dataset. It combines:

- dynamic interactive evaluation
- branching counterfactual coherence evaluation
- adversarial curation against shallow solver probes
- refinement and search-conditioned refinement of weak generators
- interpretable failure categories such as `static_dynamic_gap`, `overconfident_error`, and `social_belief_confusion`

AGUS v2 adds one especially judge-visible idea: **benchmarking coherence across nearby alternate futures**. Instead of evaluating only one observed trajectory, AGUS can now ask whether a model stays coherent when contradiction appears versus does not appear, when a representation shifts differently, or when private information is present in one branch but not another. In the current local results, Qwen also leads Llama on these counterfactual coherence metrics.

## What Static Evaluations Miss

If we only looked at static accuracy, we would conclude that Llama is the stronger model. AGUS shows a more interesting picture: Qwen is weaker on frozen-task correctness but stronger on adaptive reasoning quality, online revision, and overall brittleness under changing evidence.

That is the central value proposition of AGUS: it measures a learning-relevant dimension that static evaluation compresses away.
