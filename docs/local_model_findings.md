# Local Model Findings

## Scope

This note summarizes the current AGUS evidence from the local Ollama runs:

- `llama31_balanced_static100`
- `llama31_balanced_interactive100`
- `qwen25_balanced_static100`
- `qwen25_balanced_interactive100`
- `llama31_balanced_interactive25_replication_fixed`
- `qwen25_balanced_interactive25_replication_fixed`
- `mistralnemo_balanced_static100`
- `mistralnemo_balanced_interactive100`
- `llama31_counterfactual_v2`
- `qwen25_counterfactual_v2`
- `mistralnemo_counterfactual_v2`

The goal is not to make sweeping model-quality claims. The goal is to show what AGUS measures that static accuracy alone would miss.

## Top-Line Comparison

| Model | Static Accuracy | Belief Trajectory Quality | Episode Cognitive Flexibility | Contradiction Blindness Rate | Trajectory Instability Index |
| --- | ---: | ---: | ---: | ---: | ---: |
| `llama3.1:8b` | 0.6179 | 0.5434 | 0.6348 | 0.66 | 0.3348 |
| `qwen2.5:7b` | 0.3000 | 0.7281 | 0.7361 | 0.64 | 0.2626 |
| `mistral-nemo:12b` | 0.2714 | 0.5828 | 0.6982 | 0.56 | 0.2513 |

The important pattern is no longer a simple two-model inversion. It is a three-model separation:

- Llama wins clearly on static accuracy
- Qwen wins clearly on adaptive trajectory quality
- Mistral is weak on static accuracy but stronger on dynamic and instability metrics than that score alone would predict

## Fresh-Slice Replication

The central AGUS result also held on a fresh deterministic balanced slice for the main Llama-versus-Qwen comparison.

| Slice | Model | Static Accuracy | Belief Trajectory Quality |
| --- | --- | ---: | ---: |
| original | `llama3.1:8b` | 0.6179 | 0.5434 |
| original | `qwen2.5:7b` | 0.3000 | 0.7281 |
| replication | `llama3.1:8b` | 0.4857 | 0.5606 |
| replication | `qwen2.5:7b` | 0.2857 | 0.7494 |

What replicated:

- static accuracy ranking: Llama remained ahead of Qwen
- `belief_trajectory_quality` ranking: Qwen remained ahead of Llama
- the main static-versus-dynamic divergence persisted
- the main weakness-proxy directions remained the same:
  - `overconfident_error`
  - `static_dynamic_gap`
  - `social_belief_confusion`

This does not prove broad robustness. It does support a narrower and important claim: the core AGUS signal is not just a one-slice accident in the current local-model setup.

## Static Versus Dynamic Separation

AGUS interactive metrics produce a different ranking than static accuracy.

Qwen leads on the main adaptive-trajectory metrics:

- `belief_trajectory_quality`: `0.7281`
- `episode_cognitive_flexibility_score`: `0.7361`

Mistral lands in between on those same trajectory metrics:

- `belief_trajectory_quality`: `0.5828`
- `episode_cognitive_flexibility_score`: `0.6982`

Llama remains the static leader:

- `accuracy`: `0.6179`

This is a useful AGUS result because it suggests that dynamic cognition is not a thin wrapper around static correctness.

## Instability Findings

Instability adds another ranking:

- Mistral is least contradiction-blind and least unstable overall
- Qwen is close behind and still much less brittle than Llama
- Llama is the most brittle overall despite its static lead

| Model | Trajectory Instability Index | Unnecessary Revision Rate | Brittle Reversal Rate | Contradiction Blindness Rate |
| --- | ---: | ---: | ---: | ---: |
| `llama3.1:8b` | 0.3348 | 0.30 | 0.32 | 0.66 |
| `qwen2.5:7b` | 0.2626 | 0.11 | 0.11 | 0.64 |
| `mistral-nemo:12b` | 0.2513 | 0.13 | 0.14 | 0.56 |

The cleanest safe interpretation is:

- Llama is more brittle overall under AGUS interactive conditions
- Qwen and Mistral are both less brittle overall, but all three models remain highly contradiction-blind in absolute terms

One nuance should stay visible in any writeup:

- Qwen has the highest `confidence_volatility`: `0.2772` versus `0.1167` for Mistral and `0.1012` for Llama

So "less brittle overall" should not be rewritten as "uniformly more stable on every dimension."

## AGUS v2 Counterfactual Coherence

Counterfactual branching adds a second dynamic layer beyond ordinary interactive revision.

| Model | Counterfactual Update Fidelity | Invariant Preservation Score | Branch Belief Coherence | Cross-Branch Consistency | Counterfactual Confidence Calibration |
| --- | ---: | ---: | ---: | ---: | ---: |
| `llama3.1:8b` | 0.7222 | 0.7500 | 0.7037 | 1.0000 | 0.9561 |
| `qwen2.5:7b` | 0.8333 | 0.8438 | 0.8542 | 1.0000 | 0.9717 |
| `mistral-nemo:12b` | 0.8889 | 0.9375 | 0.8333 | 1.0000 | 0.9697 |

The current v2 pattern is:

- Mistral is strongest on update fidelity and invariant preservation
- Qwen is strongest on branch belief coherence and confidence calibration
- Llama still leads only on the frozen-task axis, not the counterfactual one

All three models reached `cross_branch_consistency` of `1.0`, so the more discriminative branch metrics are update fidelity, invariant preservation, branch belief coherence, and confidence calibration.

## Why These Findings Matter For AGUS

AGUS is already doing useful scientific work if it can separate:

- frozen-task correctness
- online revision quality
- instability under new evidence
- contradiction blindness
- counterfactual coherence
- social belief tracking
- confidence behavior under contradiction

The local-model result set does exactly that. It does not yet establish a frontier-model leaderboard, but it does establish that AGUS can reveal interpretable structure that ordinary static benchmarks would flatten away.
