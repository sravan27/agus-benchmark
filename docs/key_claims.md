# Key Claims

## Three Strongest Claims We Can Safely Make

1. Across three local models, AGUS separates frozen-task correctness from adaptive reasoning quality, and the core Llama-versus-Qwen split holds on three fresh deterministic balanced replication slices. In the original runs, Llama leads static accuracy (`0.6179`) while Qwen leads `belief_trajectory_quality` (`0.7281`). The same directional split holds on `replication`, `replication_2`, and `replication_3`, and the three judge-facing weakness proxies also replicate directionally on `3/3` slices.
2. AGUS v2 adds counterfactual coherence as another separable axis. In the compact three-model branching runs, `mistral-nemo:12b` leads on `counterfactual_update_fidelity` (`0.8889`) and `invariant_preservation_score` (`0.9375`), while `qwen2.5:7b` leads on `branch_belief_coherence` (`0.8542`) and `counterfactual_confidence_calibration` (`0.9717`), even though Llama remains the strongest frozen-task model. A focused expanded Llama-versus-Qwen run with `12` bundles and `24` branches per model preserved the same directional result, with Qwen ahead on all discriminative branch metrics.
3. Instability is not reducible to static score. `mistral-nemo:12b` has the lowest current `contradiction_blindness_rate` (`0.56`) and `trajectory_instability_index` (`0.2513`), `qwen2.5:7b` is close behind on overall instability (`0.2626`), and `llama3.1:8b` is clearly most brittle overall (`0.3348`).

## Three Claims We Should Not Overstate Yet

1. We should not claim that any one of the three local models is the strongest overall. The whole point of the current evidence is that the ordering changes by metric family.
2. We should not claim that AGUS has already validated dynamic cognition at frontier scale. The current evidence is from three local open models, not a broad frontier-model comparison.
3. We should not claim that the best current instability or counterfactual results imply strong absolute robustness. Even the lowest `contradiction_blindness_rate`, `0.56`, is still high, and AGUS v2 still has only one expanded pairwise run rather than a broad large-bundle study.

## Three Important Limitations

1. The evidence comes from a small local-model set with one primary inference stack per model, and the robustness check is three deterministic balanced slices rather than a broad multi-seed or multi-stack study.
2. AGUS is synthetic and interactive, which is a strength for controllability but still leaves open questions about transfer to more naturalistic tasks.
3. The current results packet is strongest on comparative patterns across static, interactive, instability, and counterfactual metrics, not on broad claims about real-world deployment competence.
