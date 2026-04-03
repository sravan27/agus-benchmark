# Final Results Table

## Core Comparison

| Model | Static Accuracy | Belief Trajectory Quality | Episode Cognitive Flexibility | Contradiction Blindness Rate | Trajectory Instability Index |
| --- | ---: | ---: | ---: | ---: | ---: |
| `llama3.1:8b` | 0.6179 | 0.5434 | 0.6348 | 0.66 | 0.3348 |
| `qwen2.5:7b` | 0.3000 | 0.7281 | 0.7361 | 0.64 | 0.2626 |
| `mistral-nemo:12b` | 0.2714 | 0.5828 | 0.6982 | 0.56 | 0.2513 |

`contradiction_blindness_rate` and `trajectory_instability_index` are lower-is-better.

## AGUS v2 Counterfactual Coherence

| Model | Counterfactual Update Fidelity | Invariant Preservation Score | Branch Belief Coherence | Cross-Branch Consistency | Counterfactual Confidence Calibration |
| --- | ---: | ---: | ---: | ---: | ---: |
| `llama3.1:8b` | 0.7222 | 0.7500 | 0.7037 | 1.0000 | 0.9561 |
| `qwen2.5:7b` | 0.8333 | 0.8438 | 0.8542 | 1.0000 | 0.9717 |
| `mistral-nemo:12b` | 0.8889 | 0.9375 | 0.8333 | 1.0000 | 0.9697 |

## Current Judge-Facing Failure Types

The most legible AGUS failure categories in the current distilled evidence are:

- `overconfident_error`
- `static_dynamic_gap`
- `social_belief_confusion`

## Takeaway

Llama is strongest on frozen-task accuracy. Qwen is strongest on adaptive trajectory quality. Mistral reinforces the AGUS thesis by pairing weak static accuracy with lower contradiction blindness, lower overall instability, and the strongest current counterfactual update and invariant-preservation metrics. AGUS is valuable because it makes those separable dimensions visible instead of collapsing them into one score.
