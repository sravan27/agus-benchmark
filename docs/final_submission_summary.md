# AGUS Final Submission Summary

## What AGUS Is

AGUS, **Adaptive Generalization Under Shift**, is a Learning-track benchmark for measuring whether a model can infer new rules, notice when its current strategy stops working, and adapt after contradiction, distractors, or representation shifts.

## Why It Matters

Many benchmarks reward frozen-task correctness. AGUS is built to measure **dynamic cognition**: hypothesis revision, confidence recalibration, attention recovery, transfer under shift, and belief tracking across short interactive episodes.

## Strongest Result

The clearest result from the current local-model runs is that **static accuracy, adaptive reasoning quality, and counterfactual coherence diverge rather than collapsing to one ranking**:

- `llama3.1:8b` static accuracy: `0.6179`
- `qwen2.5:7b` belief trajectory quality: `0.7281` and episode cognitive flexibility: `0.7361`
- `mistral-nemo:12b` contradiction blindness rate: `0.56`, lower than Qwen `0.64` and Llama `0.66`
- `mistral-nemo:12b` also posts the strongest current `counterfactual_update_fidelity`: `0.8889`

That central AGUS signal also replicated on a fresh deterministic balanced slice. On the original slice, Llama beat Qwen on static accuracy (`0.6179` vs `0.3000`) while Qwen beat Llama on `belief_trajectory_quality` (`0.7281` vs `0.5434`). On the replication slice, the same pattern held: Llama static accuracy `0.4857` vs Qwen `0.2857`, while Qwen `belief_trajectory_quality` `0.7494` vs Llama `0.5606`.

## Why It Is Novel

AGUS is not just another static reasoning dataset. It combines:

- dynamic interactive evaluation
- branching counterfactual coherence evaluation
- adversarial curation against shallow solver probes
- refinement and search-conditioned refinement of weak generators
- interpretable failure categories such as `static_dynamic_gap`, `overconfident_error`, and `social_belief_confusion`

AGUS v2 adds one especially judge-visible idea: **benchmarking coherence across nearby alternate futures**. Instead of evaluating only one observed trajectory, AGUS can now ask whether a model stays coherent when contradiction appears versus does not appear, when a representation shifts differently, or when private information is present in one branch but not another. In the current local results, Mistral is strongest on counterfactual update fidelity and invariant preservation, while Qwen is strongest on branch belief coherence and counterfactual confidence calibration.

## What Static Evaluations Miss

If we only looked at static accuracy, we would conclude that Llama is the stronger model and Mistral is the weakest. AGUS shows a more interesting picture: Qwen is stronger on adaptive reasoning quality, and Mistral is materially better on contradiction handling and counterfactual coherence than its static score alone would suggest.

That is the central value proposition of AGUS: it measures a learning-relevant dimension that static evaluation compresses away, and the core signal is not just a one-slice fluke inside the current local-model scope.
