# Local Model Findings

## Scope

This note summarizes the current AGUS evidence from the local Ollama runs:

- `llama31_balanced_static100`
- `llama31_balanced_interactive100`
- `qwen25_balanced_static100`
- `qwen25_balanced_interactive100`

The goal is not to make sweeping model-quality claims. The goal is to show what AGUS measures that static accuracy alone would miss.

## Top-Line Comparison

| Model | Static Accuracy | Belief Trajectory Quality | Episode Cognitive Flexibility | Online Adaptation Gain |
| --- | ---: | ---: | ---: | ---: |
| `llama3.1:8b` | 0.6179 | 0.5434 | 0.6348 | 0.3150 |
| `qwen2.5:7b` | 0.3000 | 0.7281 | 0.7361 | 0.5250 |

The important pattern is the inversion:

- Llama wins clearly on static accuracy
- Qwen wins clearly on dynamic revision quality

## Static Versus Interactive Gap

AGUS interactive metrics widen the difference between the two models in ways that static accuracy does not predict.

Qwen leads on:

- `belief_trajectory_quality`: `0.7281` versus `0.5434`
- `hypothesis_update_score`: `0.81` versus `0.58`
- `belief_state_consistency`: `0.951` versus `0.7937`
- `confidence_recalibration_score`: `0.865` versus `0.5995`
- `attention_recovery_score`: `0.7812` versus `0.575`

Llama still retains some dynamic strengths:

- `trust_revision_score`: `0.7667` versus `0.7333`
- `deception_sensitivity`: `0.4667` versus `0.3000`

This is a useful AGUS result because it suggests that dynamic cognition is not a thin wrapper around static correctness.

## Instability Findings

Overall instability favors Qwen:

| Model | Trajectory Instability Index | Unnecessary Revision Rate | Brittle Reversal Rate | Contradiction Blindness Rate |
| --- | ---: | ---: | ---: | ---: |
| `llama3.1:8b` | 0.3348 | 0.30 | 0.32 | 0.66 |
| `qwen2.5:7b` | 0.2626 | 0.11 | 0.11 | 0.64 |

The cleanest safe interpretation is:

- Llama is more brittle overall under AGUS interactive conditions
- Qwen is less brittle overall, but both models remain highly contradiction-blind

One nuance should stay visible in any writeup:

- Qwen has higher `confidence_volatility`: `0.2772` versus `0.1012`

So "less brittle overall" should not be rewritten as "uniformly more stable on every dimension."

## Strongest Separating Weaknesses

The clearest category-level differences in the current local runs are:

| Weakness Type | Llama Count | Qwen Count | Separation |
| --- | ---: | ---: | ---: |
| `overconfident_error` | 89 | 68 | +21 Llama |
| `static_dynamic_gap` | 20 | 1 | +19 Llama |
| `social_belief_confusion` | 13 | 1 | +12 Llama |

These are useful because they are legible to judges:

- `overconfident_error` shows harmful certainty, not just wrongness
- `static_dynamic_gap` shows failure only when adaptation is required
- `social_belief_confusion` shows trouble separating world state from agent belief state

## Family-Level Pattern

Interactive family averages show a broad Qwen advantage except in some socially targeted submetrics:

- `attention_distractors`: `0.8125` versus `0.525`
- `hidden_rule`: `0.9000` versus `0.675`
- `shift_transfer`: `1.0000` versus `0.8000`
- `social_miniworlds`: `1.0000` versus `0.9500`
- `metacog_revision`: Llama leads `0.6500` versus `0.4500`

This suggests the current benchmark is measuring more than a single generic capability. Different modules expose different strengths.

## Why These Findings Matter For AGUS

AGUS is already doing useful scientific work if it can separate:

- frozen-task correctness
- online revision quality
- instability under new evidence
- social belief tracking
- confidence behavior under contradiction

The local-model result set does exactly that. It does not yet establish a frontier-model leaderboard, but it does establish that AGUS can reveal interpretable structure that ordinary static benchmarks would flatten away.
