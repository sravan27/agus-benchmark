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

## Multi-Slice Robustness For The Main Split

| Slice | Llama Static | Llama BTQ | Qwen Static | Qwen BTQ | Core Pattern Held |
| --- | ---: | ---: | ---: | ---: | ---: |
| `original` | 0.6179 | 0.5434 | 0.3000 | 0.7281 | Yes |
| `replication` | 0.4857 | 0.5606 | 0.2857 | 0.7494 | Yes |
| `replication_2` | 0.6429 | 0.5767 | 0.3143 | 0.7257 | Yes |
| `replication_3` | 0.6429 | 0.4879 | 0.3286 | 0.7329 | Yes |

Held on `3/3` replication slices:

- static accuracy ranking
- `belief_trajectory_quality` ranking
- static-vs-dynamic divergence
- the main weakness-proxy directions

## Expanded AGUS v2 Pair Check

| Model | Bundles | Branches | Update Fidelity | Invariant Preservation | Branch Belief Coherence | Counterfactual Confidence Calibration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `llama3.1:8b` | 12 | 24 | 0.7222 | 0.6875 | 0.6157 | 0.9511 |
| `qwen2.5:7b` | 12 | 24 | 0.8611 | 0.8281 | 0.8090 | 0.9681 |

## Current Judge-Facing Failure Types

The most legible AGUS failure categories in the current distilled evidence are:

- `overconfident_error`
- `static_dynamic_gap`
- `social_belief_confusion`

## Takeaway

Llama is strongest on frozen-task accuracy. Qwen is strongest on adaptive trajectory quality and still leads Llama on the expanded AGUS v2 branch set. Mistral reinforces the AGUS thesis by pairing weak static accuracy with lower contradiction blindness, lower overall instability, and the strongest current compact-run counterfactual update and invariant-preservation metrics. AGUS is valuable because it makes those separable dimensions visible instead of collapsing them into one score.
