# AGUS

**AGUS** stands for **Adaptive Generalization Under Shift**.

It is the umbrella benchmark identity for this repository: a research benchmark family for testing whether models can infer rules, detect change, revise hypotheses, and stay coherent when evidence shifts.

## Current Kaggle Submission: Learning Core

The **currently submitted Kaggle benchmark slice** is **Learning Core**.

Learning Core packages exactly **three** AGUS task families:

- `hidden_rule`
- `shift_transfer`
- `metacog_revision`

These three families are the current Kaggle benchmark because they are the cleanest expression of the Learning-track thesis: **can a model learn from sparse evidence, notice when its current strategy is no longer enough, and adapt after new evidence or a representation shift?**

## Core Empirical Claim

AGUS’s main empirical claim is:

**static correctness and adaptive reasoning quality can diverge sharply.**

In the current local Learning Core results:

- `llama3.1:8b` leads on static accuracy: `0.6179`
- `qwen2.5:7b` leads on `belief_trajectory_quality`: `0.7281` versus Llama `0.5434`
- `mistral-nemo:12b` reinforces the split with weak static accuracy (`0.2714`) but stronger adaptive quality (`belief_trajectory_quality 0.5828`)

That central Llama-versus-Qwen split also held on **3/3 deterministic replication slices**:

- Llama stayed ahead on static accuracy
- Qwen stayed ahead on `belief_trajectory_quality`
- the main weakness-proxy directions also replicated

AGUS v2 adds a supporting counterfactual layer where coherence across nearby alternate futures becomes another separable axis.

## Submission Scope

**Kaggle submission scope:** Learning Core only.

That means the Kaggle benchmark package currently includes:

- `hidden_rule`
- `shift_transfer`
- `metacog_revision`

The broader repository still contains the wider AGUS research program as **supporting or extended scope**, including:

- `attention_distractors`
- `social_miniworlds`
- AGUS v2 counterfactual branching
- adversarial curation and refinement tooling
- local-model evaluation, replication, and failure-analysis artifacts

Those broader materials stay in the repo because they strengthen the research story, but they are **not** the same thing as the currently submitted Kaggle benchmark package.

## Start Here

If you are reviewing the submission, start in this order:

1. [Submission Overview](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/submission/README.md)
2. [Executive Summary](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/submission/executive_summary.md)
3. [Benchmark Scope](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/submission/benchmark_scope.md)
4. [Key Results](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/submission/key_results.md)
5. [Reviewer Guide](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/submission/reviewer_guide.md)
6. [Kaggle Benchmark Package](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/kaggle_benchmark/README.md)

## Repo Map

- [docs/submission/](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/submission)
  Submission-facing reading set for judges and reviewers.
- [kaggle_benchmark/](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/kaggle_benchmark)
  Canonical Kaggle benchmark package for the Learning Core slice.
- [data/evals/](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/data/evals)
  Local-model result artifacts, replication outputs, and comparison summaries.
- [docs/](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs)
  Broader AGUS design notes, results packet drafts, audits, and supporting research materials.
- [src/](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/src)
  Benchmark generation, scoring, evaluation, curation, refinement, and analysis code.
- [tests/](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/tests)
  Unit and regression tests.

## Reviewer Notes

- The **canonical submission-facing docs** are now under [docs/submission/](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/submission).
- Older docs elsewhere in [docs/](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs) may describe the broader AGUS research scope, including five-family and AGUS v2 materials.
- The **submitted Kaggle benchmark package** remains the **Learning Core** slice only.

## Validation

Repo tests currently pass with:

```bash
PYTHONPATH=. pytest -q
```
