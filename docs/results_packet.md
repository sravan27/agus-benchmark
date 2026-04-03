# AGUS Results Packet v1

## Benchmark Thesis

AGUS is designed to measure **adaptive generalization under shift**, not just frozen-task correctness. The key question is whether a model can stay coherent when the task changes, new evidence arrives, or a previously successful strategy stops working.

The current local-model result set already supports a stronger benchmark claim:

**static correctness, adaptive reasoning quality, and counterfactual coherence do not collapse to one ranking.**

## Model Comparison Summary

Current local runs compare:

- `llama3.1:8b` via Ollama
- `qwen2.5:7b` via Ollama
- `mistral-nemo:12b` via Ollama

All three were evaluated on balanced 100-task AGUS runs with static and interactive settings.

| Model | Static Accuracy | Belief Trajectory Quality | Episode Cognitive Flexibility | Contradiction Blindness Rate | Trajectory Instability Index |
| --- | ---: | ---: | ---: | ---: | ---: |
| `llama3.1:8b` | 0.6179 | 0.5434 | 0.6348 | 0.66 | 0.3348 |
| `qwen2.5:7b` | 0.3000 | 0.7281 | 0.7361 | 0.64 | 0.2626 |
| `mistral-nemo:12b` | 0.2714 | 0.5828 | 0.6982 | 0.56 | 0.2513 |

## Fresh-Slice Replication

The main Llama-versus-Qwen AGUS pattern also held on a **fresh deterministic balanced slice**, which matters because it reduces the strongest obvious dismissal risk: that the result is only a one-slice artifact.

| Slice | Model | Static Accuracy | Belief Trajectory Quality |
| --- | --- | ---: | ---: |
| original | `llama3.1:8b` | 0.6179 | 0.5434 |
| original | `qwen2.5:7b` | 0.3000 | 0.7281 |
| replication | `llama3.1:8b` | 0.4857 | 0.5606 |
| replication | `qwen2.5:7b` | 0.2857 | 0.7494 |

What replicated:

- static accuracy ranking replicated: Llama stayed ahead of Qwen
- `belief_trajectory_quality` ranking replicated: Qwen stayed ahead of Llama
- the core static-versus-dynamic divergence replicated
- the main weakness-proxy directions also replicated

This is still a lightweight robustness check, not a large multi-seed study. But it is enough to support the narrower claim that the central AGUS signal is **not just one lucky balanced slice**.

## Strongest Quantitative Findings

1. `llama3.1:8b` is still the strongest frozen-task model in the current local set, with static accuracy `0.6179` versus `0.3000` for Qwen and `0.2714` for Mistral.
2. `qwen2.5:7b` is still the strongest adaptive-trajectory model:
   - `belief_trajectory_quality`: `0.7281`
   - `episode_cognitive_flexibility_score`: `0.7361`
3. `mistral-nemo:12b` reinforces the static-versus-dynamic separation rather than collapsing it:
   - static accuracy is only `0.2714`
   - but `belief_trajectory_quality` is `0.5828`
   - and `episode_cognitive_flexibility_score` is `0.6982`
4. Mistral also has the lowest current contradiction blindness and overall instability:
   - `contradiction_blindness_rate`: `0.56` versus `0.64` for Qwen and `0.66` for Llama
   - `trajectory_instability_index`: `0.2513` versus `0.2626` for Qwen and `0.3348` for Llama

## Key Interpretation

The three-model evidence makes the AGUS story much stronger than a simple Llama-versus-Qwen contrast:

- Llama leads on frozen-task correctness
- Qwen leads on interactive adaptation quality
- Mistral adds a third anchor by showing weak static accuracy but materially stronger dynamic behavior than its static score would predict
- and the original Llama-versus-Qwen split also survives a fresh deterministic balanced slice

That is exactly the kind of separation AGUS was built to expose. A benchmark focused only on static accuracy would rank these models very differently from a benchmark that also measures hypothesis revision, confidence recalibration, and multi-turn cognitive flexibility.

## Why AGUS Reveals What Static Accuracy Misses

The clearest example is Mistral. If we only looked at its static score, we would classify it as the weakest model in the set. AGUS shows a more nuanced picture:

- Mistral static accuracy: `0.2714`
- Mistral `belief_trajectory_quality`: `0.5828`
- Mistral `episode_cognitive_flexibility_score`: `0.6982`
- Mistral `contradiction_blindness_rate`: `0.56`, the lowest of the three

Qwen shows a different version of the same pattern:

- much lower static accuracy than Llama
- but clearly stronger trajectory quality and lower overall brittleness

So the current evidence is not that one model is simply better. The stronger claim is:

**AGUS exposes multiple learning-relevant axes that static accuracy compresses away.**

## AGUS v2: Counterfactual Branching

AGUS v2 extends the benchmark beyond ordinary interactive evaluation by creating **multiple nearby alternate futures from the same base episode**. Instead of asking whether a model can survive one observed trajectory, AGUS v2 asks whether the model stays coherent when one critical factor changes:

- contradiction appears versus does not appear
- the representation shifts through one codebook versus a nearby alternate codebook
- one agent has private information versus everyone sees the same event

This matters because a model can sometimes look competent on a single trajectory by reacting locally to prompts. Branching counterfactuals are a stronger test of whether the model is carrying a coherent internal task model across nearby possibilities.

### Current AGUS v2 Results

| Model | Update Fidelity | Invariant Preservation | Branch Belief Coherence | Cross-Branch Consistency | Counterfactual Confidence Calibration |
| --- | ---: | ---: | ---: | ---: | ---: |
| `llama3.1:8b` | 0.7222 | 0.7500 | 0.7037 | 1.0000 | 0.9561 |
| `qwen2.5:7b` | 0.8333 | 0.8438 | 0.8542 | 1.0000 | 0.9717 |
| `mistral-nemo:12b` | 0.8889 | 0.9375 | 0.8333 | 1.0000 | 0.9697 |

AGUS v2 adds another separable axis:

- Mistral is strongest on current `counterfactual_update_fidelity` and `invariant_preservation_score`
- Qwen is strongest on current `branch_belief_coherence` and `counterfactual_confidence_calibration`
- Llama remains strongest on frozen-task accuracy

That makes AGUS more compelling as a Learning-track benchmark. It now measures not only revision along one path, but also whether revision behavior stays coherent across tightly controlled alternate continuations.

### Why This Goes Beyond Ordinary Interactive Evaluation

Ordinary interactive evaluation can tell us whether a model updates after one stream of evidence. AGUS v2 can tell us whether the same model:

- revises when contradiction truly appears but does not over-revise when it does not
- preserves the latent rule across nearby remapping branches
- keeps world state fixed while updating only the appropriate belief state in social branches

That is a more AGI-relevant notion of coherence than answer quality on one path alone.

## Important Caveats

- These are local-model results from three open models, not a broad frontier-model comparison.
- The replication evidence is one fresh deterministic balanced slice, not a broad multi-seed robustness study.
- All three models are still substantially contradiction-blind in absolute terms. Even the best current value, Mistral's `0.56`, is still high.
- Qwen and Mistral are less brittle overall than Llama, but that does not mean they are uniformly more stable on every submetric. Qwen's `confidence_volatility`, for example, is still the highest of the three at `0.2772`.
- AGUS v2 currently uses a compact counterfactual bundle set, so those metrics are useful evidence but not yet a large-scale leaderboard.

## Recommended Use In A Submission

The safest headline for AGUS today is:

**Across three local models, AGUS shows that frozen-task accuracy, adaptive reasoning quality, and counterfactual coherence are separable dimensions, and the core static-versus-dynamic split replicates on a fresh balanced slice.**

Supporting materials:

- [local_model_findings.md](./local_model_findings.md)
- [failure_gallery.md](./failure_gallery.md)
- [key_claims.md](./key_claims.md)
- [agus_v2_results.md](./agus_v2_results.md)
