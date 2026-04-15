# Reviewer Guide

## Fastest 5-Minute Path

If you only want the essential submission story:

1. Read [Executive Summary](./executive_summary.md)
2. Read [Benchmark Scope](./benchmark_scope.md)
3. Read [Key Results](./key_results.md)
4. Open [static_vs_adaptive_divergence.svg](./static_vs_adaptive_divergence.svg)
5. If you want the package itself, open [kaggle_benchmark/README.md](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/kaggle_benchmark/README.md)

## What To Keep In Mind

- **AGUS** is the umbrella benchmark identity for the repository.
- **Learning Core** is the currently submitted Kaggle benchmark slice.
- Learning Core includes exactly:
  - `hidden_rule`
  - `shift_transfer`
  - `metacog_revision`

Do not assume the broader five-family AGUS research suite is the same thing as the current Kaggle benchmark package.

## What The Repo Is Claiming

The repo is making one primary claim:

**Static correctness and adaptive reasoning quality can diverge sharply.**

The strongest evidence for that claim is:

- Llama leads on static accuracy
- Qwen leads on adaptive trajectory quality
- Mistral reinforces the separation rather than collapsing it
- the Llama-vs-Qwen split held on `3/3` fresh deterministic replication slices

## Where To Verify The Evidence

Learning Core run artifacts:

- [llama31_balanced_interactive100](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/data/evals/llama31_balanced_interactive100/aggregate_summary.json)
- [qwen25_balanced_interactive100](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/data/evals/qwen25_balanced_interactive100/aggregate_summary.json)
- [mistralnemo_balanced_interactive100](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/data/evals/mistralnemo_balanced_interactive100/aggregate_summary.json)

Replication summary:

- [llama_qwen_multi_slice_robustness_v1](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/data/evals/comparisons/llama_qwen_multi_slice_robustness_v1/robustness_summary.json)

Supporting AGUS v2 comparison:

- [llama31_counterfactual_v2_expanded](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/data/evals/llama31_counterfactual_v2_expanded/counterfactual_summary.json)
- [qwen25_counterfactual_v2_expanded](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/data/evals/qwen25_counterfactual_v2_expanded/counterfactual_summary.json)
- [mistralnemo_counterfactual_v2](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/data/evals/mistralnemo_counterfactual_v2/counterfactual_summary.json)

## What Is Broader Supporting Scope

The repo also includes:

- `attention_distractors`
- `social_miniworlds`
- adversarial curation
- refinement and search-conditioned refinement
- failure distillation
- instability analysis

Those materials support the research case, but they are not required to understand the current Kaggle submission.
