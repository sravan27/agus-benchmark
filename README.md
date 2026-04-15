# AGUS

**AGUS** stands for **Adaptive Generalization Under Shift**.

It is the umbrella benchmark identity for this repository. The **current Kaggle submission** is the **Learning Core** slice: a narrow, submission-ready benchmark for the Learning track.

## Start Here

- [Kaggle benchmark package](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/kaggle_benchmark/README.md)
- [Final writeup draft](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/final_learning_track_submission.md)
- [Submission overview](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/submission/README.md)
- [Key results](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/submission/key_results.md)
- [Key figure](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/submission/static_vs_adaptive_divergence.svg)
- [Reviewer guide](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/submission/reviewer_guide.md)

## Main Finding

**Static correctness and adaptive reasoning quality can diverge sharply.**

Learning Core results from existing repo artifacts:

| Model | Static accuracy | Belief trajectory quality |
| --- | ---: | ---: |
| `llama3.1:8b` | 0.6179 | 0.5434 |
| `qwen2.5:7b` | 0.3000 | 0.7281 |
| `mistral-nemo:12b` | 0.2714 | 0.5828 |

That split is not confined to the original slice. On the first fresh deterministic replication slice:

- Llama static accuracy: `0.4857`
- Qwen static accuracy: `0.2857`
- Llama `belief_trajectory_quality`: `0.5606`
- Qwen `belief_trajectory_quality`: `0.7494`

And the same Llama-vs-Qwen pattern held on **3/3 deterministic replication slices**.

![AGUS Learning Core static-vs-adaptive divergence](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/submission/static_vs_adaptive_divergence.svg)

## Submission Scope

- **AGUS** = the umbrella benchmark identity.
- **Learning Core** = the currently submitted Kaggle slice.
- Learning Core packages exactly:
  - `hidden_rule`
  - `shift_transfer`
  - `metacog_revision`

Broader AGUS materials remain in the repo as **supporting or extended scope**, not as the current Kaggle benchmark package. That broader scope includes:

- `attention_distractors`
- `social_miniworlds`
- AGUS v2 counterfactual branching
- adversarial curation and refinement
- replication, instability, and failure-analysis artifacts

## Why AGUS Matters

Static benchmarks are good at telling you whether a model can answer a frozen task. They are much worse at telling you whether the model can:

- infer a rule from sparse evidence
- notice that the old rule no longer fits
- revise after corrective evidence
- keep structure stable across a representation shift

AGUS is designed to measure that adaptive layer directly. Learning Core is the narrowest version of that claim that is still strong enough to matter.

## Review Paths

**3 minutes**

- [Executive summary](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/submission/executive_summary.md)
- [Key results](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/submission/key_results.md)
- [Key figure](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/submission/static_vs_adaptive_divergence.svg)

**10 minutes**

- [Submission overview](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/submission/README.md)
- [Benchmark scope](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/submission/benchmark_scope.md)
- [Reviewer guide](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/submission/reviewer_guide.md)
- [Final writeup draft](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/final_learning_track_submission.md)

**Deeper review**

- [Kaggle benchmark package](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/kaggle_benchmark/README.md)
- [Kaggle submission audit](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/kaggle_submission_audit.md)
- [Hostile review defense](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/hostile_review_defense.md)
- [Broader docs folder](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs)

## Repo Map

- [docs/submission/](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/submission): canonical reviewer path
- [kaggle_benchmark/](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/kaggle_benchmark): submitted Learning Core package
- [data/evals/](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/data/evals): result and replication artifacts
- [docs/](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs): writeups, audits, and supporting research materials
- [src/](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/src): benchmark generation and evaluation code
- [tests/](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/tests): regression and unit tests
