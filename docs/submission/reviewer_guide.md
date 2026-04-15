# Reviewer Guide

## If You Want The Fastest Read

1. [Executive Summary](./executive_summary.md)
2. [Key Results](./key_results.md)
3. [static_vs_adaptive_divergence.svg](./static_vs_adaptive_divergence.svg)

## If You Want To Check Scope First

- **AGUS** = umbrella benchmark identity
- **Learning Core** = submitted Kaggle slice
- Learning Core includes exactly:
  - `hidden_rule`
  - `shift_transfer`
  - `metacog_revision`

Do not read the broader five-family AGUS research suite as if it were the same thing as the current Kaggle benchmark package.

## What The Submission Is Claiming

One main claim:

**Static correctness and adaptive reasoning quality can diverge sharply.**

Current evidence for that claim:

- Llama leads on static accuracy
- Qwen leads on adaptive trajectory quality
- Mistral reinforces the separation rather than collapsing it
- the Llama-vs-Qwen split held on `3/3` fresh deterministic replication slices

## Where To Verify The Evidence

Learning Core run artifacts:

- [llama31_balanced_interactive100](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/data/evals/llama31_balanced_interactive100/aggregate_summary.json)
- [qwen25_balanced_interactive100](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/data/evals/qwen25_balanced_interactive100/aggregate_summary.json)
- [mistralnemo_balanced_interactive100](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/data/evals/mistralnemo_balanced_interactive100/aggregate_summary.json)

Replication artifact:

- [llama_qwen_multi_slice_robustness_v1](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/data/evals/comparisons/llama_qwen_multi_slice_robustness_v1/robustness_summary.json)

Supporting AGUS v2 artifacts:

- [llama31_counterfactual_v2_expanded](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/data/evals/llama31_counterfactual_v2_expanded/counterfactual_summary.json)
- [qwen25_counterfactual_v2_expanded](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/data/evals/qwen25_counterfactual_v2_expanded/counterfactual_summary.json)
- [mistralnemo_counterfactual_v2](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/data/evals/mistralnemo_counterfactual_v2/counterfactual_summary.json)

## What Is Supporting Scope

The repo also includes broader AGUS research materials:

- `attention_distractors`
- `social_miniworlds`
- adversarial curation and refinement
- failure distillation
- instability analysis

Those materials are useful context, but they are not required to evaluate the current Learning Core submission.
