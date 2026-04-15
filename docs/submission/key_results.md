# Key Results

## Main Learning Core Result

The central AGUS result is:

**static correctness and adaptive reasoning quality can diverge sharply.**

This table uses existing Learning Core run artifacts from `data/evals/`.

| Model | Static accuracy | Belief trajectory quality | Episode cognitive flexibility |
| --- | ---: | ---: | ---: |
| `llama3.1:8b` | 0.6179 | 0.5434 | 0.6348 |
| `qwen2.5:7b` | 0.3000 | 0.7281 | 0.7361 |
| `mistral-nemo:12b` | 0.2714 | 0.5828 | 0.6982 |

Interpretation:

- Llama is strongest on frozen-task accuracy.
- Qwen is strongest on adaptive trajectory quality.
- Mistral reinforces the separation by remaining weak on static accuracy while landing closer to the dynamic side of the comparison than the static side.

## Replication Result

The main Llama-versus-Qwen split was checked on three fresh deterministic balanced slices.

| Slice | Llama static accuracy | Llama BTQ | Qwen static accuracy | Qwen BTQ | Core pattern held |
| --- | ---: | ---: | ---: | ---: | --- |
| `original` | 0.6179 | 0.5434 | 0.3000 | 0.7281 | Yes |
| `replication` | 0.4857 | 0.5606 | 0.2857 | 0.7494 | Yes |
| `replication_2` | 0.6429 | 0.5767 | 0.3143 | 0.7257 | Yes |
| `replication_3` | 0.6429 | 0.4879 | 0.3286 | 0.7329 | Yes |

Most important replication point:

- On the first fresh deterministic slice, Llama still led on static accuracy (`0.4857` vs `0.2857`) while Qwen still led on `belief_trajectory_quality` (`0.7494` vs `0.5606`).

Replication summary:

- static accuracy ranking held on `3/3`
- `belief_trajectory_quality` ranking held on `3/3`
- the static-vs-dynamic divergence held on `3/3`
- the main weakness-proxy directions held on `3/3`

## Supporting AGUS v2 Result

AGUS v2 is a broader supporting module, not the current Kaggle package, but it strengthens the same underlying story.

Expanded counterfactual branch runs show:

| Model | Counterfactual update fidelity | Invariant preservation | Branch belief coherence |
| --- | ---: | ---: | ---: |
| `llama3.1:8b` | 0.7222 | 0.6875 | 0.6157 |
| `qwen2.5:7b` | 0.8611 | 0.8281 | 0.8090 |
| `mistral-nemo:12b` | 0.8889 | 0.9375 | 0.8333 |

That means the static-vs-dynamic separation does not just appear in one interactive metric. It also persists when AGUS asks whether a model remains coherent across nearby alternate futures.

## Figure

The clearest single visual summary is:

- [static_vs_adaptive_divergence.png](./static_vs_adaptive_divergence.png)
- [Figure source (SVG)](./static_vs_adaptive_divergence.svg)

It plots static accuracy against `belief_trajectory_quality` using the existing Learning Core run artifacts:

- `data/evals/llama31_balanced_interactive100/aggregate_summary.json`
- `data/evals/qwen25_balanced_interactive100/aggregate_summary.json`
- `data/evals/mistralnemo_balanced_interactive100/aggregate_summary.json`
