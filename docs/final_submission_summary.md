# AGUS Final Submission Summary

## What AGUS Is

AGUS, **Adaptive Generalization Under Shift**, is the umbrella benchmark identity in this repository. The current Kaggle submission packages the **Learning Core** slice: `hidden_rule`, `shift_transfer`, and `metacog_revision`.

## Why It Matters

Many benchmarks reward frozen-task correctness. AGUS is built to measure **dynamic cognition**: hypothesis revision, confidence recalibration, attention recovery, transfer under shift, and belief tracking across short interactive episodes.

## Strongest Result

The clearest result from the current local-model runs is that **static accuracy, adaptive reasoning quality, and counterfactual coherence diverge rather than collapsing to one ranking**:

- `llama3.1:8b` static accuracy: `0.6179`
- `qwen2.5:7b` belief trajectory quality: `0.7281` and episode cognitive flexibility: `0.7361`
- `mistral-nemo:12b` contradiction blindness rate: `0.56`, lower than Qwen `0.64` and Llama `0.66`
- `mistral-nemo:12b` also posts the strongest current `counterfactual_update_fidelity`: `0.8889`

That central AGUS signal also survived a three-slice deterministic robustness check. On the original slice, Llama beat Qwen on static accuracy (`0.6179` vs `0.3000`) while Qwen beat Llama on `belief_trajectory_quality` (`0.7281` vs `0.5434`). On the first fresh replication slice, Llama still led on static accuracy (`0.4857` vs `0.2857`) while Qwen still led on `belief_trajectory_quality` (`0.7494` vs `0.5606`). The same directional split held on `replication_2` and `replication_3` as well.

That result is now stronger than a one-off replication. Across `replication`, `replication_2`, and `replication_3`, the same Llama-versus-Qwen split held on `3/3` deterministic balanced slices, and the main weakness-proxy directions held on `3/3` as well.

## Why It Is Novel

AGUS is not just another static reasoning dataset. It combines:

- dynamic interactive evaluation
- branching counterfactual coherence evaluation
- adversarial curation against shallow solver probes
- refinement and search-conditioned refinement of weak generators
- interpretable failure categories such as `static_dynamic_gap`, `overconfident_error`, and `social_belief_confusion`

AGUS v2 adds one especially judge-visible idea: **benchmarking coherence across nearby alternate futures**. Instead of evaluating only one observed trajectory, AGUS can now ask whether a model stays coherent when contradiction appears versus does not appear, when a representation shifts differently, or when private information is present in one branch but not another. In the current local results, Mistral is strongest on counterfactual update fidelity and invariant preservation, while Qwen is strongest on branch belief coherence and counterfactual confidence calibration.

AGUS v2 is also less preliminary than before. On an expanded Llama-versus-Qwen branch set with `12` bundles and `24` branches per model, Qwen still beats Llama on every discriminative counterfactual metric: update fidelity (`0.8611` vs `0.7222`), invariant preservation (`0.8281` vs `0.6875`), branch belief coherence (`0.8090` vs `0.6157`), and confidence calibration (`0.9681` vs `0.9511`).

## What Static Evaluations Miss

If we only looked at static accuracy, we would conclude that Llama is the stronger model and Mistral is the weakest. AGUS shows a more interesting picture: Qwen is stronger on adaptive reasoning quality, and Mistral is materially better on contradiction handling and counterfactual coherence than its static score alone would suggest.

The validation bundle reinforces that this is not just a custom-score trick. On matched-composition runs, static and interactive rankings diverge, adversarial refinement improves retained tasks from `325` to `423`, and even a matched `mock_shallow` baseline can post high static accuracy (`0.7214`) without leading the adaptive metrics.

That is the central value proposition of AGUS: it measures a learning-relevant dimension that static evaluation compresses away, and the core signal is not just a one-slice fluke inside the current local-model scope.

## Why This Submission Exists In This Form

This benchmark was developed under tight practical constraints: I built and evaluated it locally on a **MacBook Pro M2 Max** and did not have budget for paid API tokens, so the first serious empirical pass had to be done with open-source local models. That constraint pushed AGUS toward a benchmark that is light enough to run, reproducible enough to inspect, and transparent enough to verify in Kaggle and GitHub rather than only in proprietary evaluation stacks.

Questions or clarifications: `sridharsravan@icloud.com`
