# Final Results Table

## Core Comparison

| Model | Static Accuracy | Belief Trajectory Quality | Episode Cognitive Flexibility | Trajectory Instability Index |
| --- | ---: | ---: | ---: | ---: |
| `llama3.1:8b` | 0.6179 | 0.5434 | 0.6348 | 0.3348 |
| `qwen2.5:7b` | 0.3000 | 0.7281 | 0.7361 | 0.2626 |

`trajectory_instability_index` is lower-is-better.

## Most Separating Weakness Types

| Weakness Type | Llama Count | Qwen Count | Separation |
| --- | ---: | ---: | ---: |
| `overconfident_error` | 89 | 68 | +21 Llama |
| `static_dynamic_gap` | 20 | 1 | +19 Llama |
| `social_belief_confusion` | 13 | 1 | +12 Llama |

## Supporting Dynamic Metrics

| Metric | Llama | Qwen | Direction |
| --- | ---: | ---: | --- |
| `online_adaptation_gain` | 0.3150 | 0.5250 | higher is better |
| `hypothesis_update_score` | 0.5800 | 0.8100 | higher is better |
| `belief_state_consistency` | 0.7937 | 0.9510 | higher is better |
| `confidence_recalibration_score` | 0.5995 | 0.8650 | higher is better |
| `attention_recovery_score` | 0.5750 | 0.7812 | higher is better |

## Takeaway

Llama is stronger on frozen-task accuracy. Qwen is stronger on adaptive reasoning quality and less brittle overall. AGUS is valuable because it makes that tradeoff visible.
