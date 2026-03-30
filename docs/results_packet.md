# AGUS Results Packet v1

## Benchmark Thesis

AGUS is designed to measure **adaptive generalization under shift**, not just frozen-task correctness. The key question is whether a model can stay coherent when the task changes, new evidence arrives, or a previously successful strategy stops working.

The first local-model result set already supports a clear benchmark claim:

**static accuracy and dynamic reasoning quality can come apart sharply.**

## Model Comparison Summary

Current local runs compare:

- `llama3.1:8b` via Ollama
- `qwen2.5:7b` via Ollama

Both were evaluated on balanced 100-task AGUS runs with static and interactive settings.

| Model Run | Static Accuracy | Belief Trajectory Quality | Episode Cognitive Flexibility | Trajectory Instability Index |
| --- | ---: | ---: | ---: | ---: |
| `llama31_balanced_interactive100` | 0.6179 | 0.5434 | 0.6348 | 0.3348 |
| `qwen25_balanced_interactive100` | 0.3000 | 0.7281 | 0.7361 | 0.2626 |

## Strongest Quantitative Findings

1. `llama3.1:8b` led on static accuracy: `0.6179` versus `0.3000` for `qwen2.5:7b`.
2. `qwen2.5:7b` led on dynamic reasoning quality:
   - `belief_trajectory_quality`: `0.7281` versus `0.5434`
   - `episode_cognitive_flexibility_score`: `0.7361` versus `0.6348`
   - `online_adaptation_gain`: `0.525` versus `0.315`
   - `hypothesis_update_score`: `0.81` versus `0.58`
3. `qwen2.5:7b` was less brittle overall on AGUS interactive traces:
   - `trajectory_instability_index`: `0.2626` versus `0.3348`
   - `unnecessary_revision_rate`: `0.11` versus `0.30`
   - `brittle_reversal_rate`: `0.11` versus `0.32`
4. The largest separating weakness categories in the current local runs were:
   - `overconfident_error`: `89` for Llama versus `68` for Qwen
   - `static_dynamic_gap`: `20` for Llama versus `1` for Qwen
   - `social_belief_confusion`: `13` for Llama versus `1` for Qwen

## Key Interpretation

AGUS is already surfacing a non-trivial tradeoff:

- `llama3.1:8b` is much stronger on frozen-task correctness
- `qwen2.5:7b` is meaningfully stronger on interactive revision, belief tracking, and adaptation quality

That is exactly the kind of separation AGUS was built to expose. A benchmark focused only on static accuracy would rank these models very differently from a benchmark that also measures hypothesis revision, confidence recalibration, and multi-turn cognitive flexibility.

## Why AGUS Reveals What Static Accuracy Misses

Several AGUS interactive metrics move in Qwen's favor despite its much lower static score:

- `attention_recovery_score`: `0.7812` versus `0.575`
- `belief_state_consistency`: `0.951` versus `0.7937`
- `confidence_recalibration_score`: `0.865` versus `0.5995`
- `hypothesis_update_score`: `0.81` versus `0.58`

At the same time, AGUS preserves nuance rather than collapsing everything into one winner:

- Llama remained stronger on `trust_revision_score`: `0.7667` versus `0.7333`
- Llama also led on `deception_sensitivity`: `0.4667` versus `0.3000`

So the current evidence is not "Qwen is simply better." The stronger claim is:

**AGUS exposes different cognitive strengths than static accuracy does, and those strengths do not align perfectly with frozen-task performance.**

## Important Caveats

- These are local-model results from two open models, not frontier closed-model evaluations.
- Both models still show high `contradiction_blindness_rate`:
  - Llama: `0.66`
  - Qwen: `0.64`
- Qwen is less brittle overall, but not uniformly more stable on every submetric. Its `confidence_volatility` is higher: `0.2772` versus `0.1012`.

## Recommended Use In A Submission

The safest headline for AGUS today is:

**AGUS distinguishes static competence from adaptive reasoning quality, and the first local-model runs already reveal a measurable tradeoff between them.**

Supporting materials:

- [local_model_findings.md](./local_model_findings.md)
- [failure_gallery.md](./failure_gallery.md)
- [key_claims.md](./key_claims.md)
