# Benchmark Scope

## Benchmark Identity

- **AGUS** = the umbrella benchmark identity in this repository.
- **Learning Core** = the currently submitted Kaggle benchmark slice.

## Submitted Kaggle Slice

The current Kaggle benchmark packages exactly three Learning Core families:

1. `hidden_rule`
2. `shift_transfer`
3. `metacog_revision`

These are the families that define the submission-facing benchmark claim:

- infer a latent rule from sparse evidence
- transfer a learned rule across a representation shift
- answer with a hypothesis and confidence, then revise after corrective evidence

## Broader Repo Scope

The broader repository contains additional AGUS research materials beyond the current Kaggle package:

- `attention_distractors`
- `social_miniworlds`
- AGUS v2 counterfactual branching
- adversarial curation and refinement tooling
- local-model evaluation and replication artifacts
- failure distillation and instability analysis

These materials are retained in the repo because they support the broader AGUS research program, but they are **not** the same thing as the current Kaggle Learning Core package.

## Why The Slice Is Narrower Than The Repo

The submitted Kaggle slice is narrower on purpose.

Learning Core is the cleanest expression of the competition-facing Learning-track thesis:

**Can a model learn from sparse evidence, notice that its old strategy no longer fits, and revise successfully after new evidence or a representation change?**

That makes the benchmark easier to explain, easier to package in Kaggle, and easier to judge without losing the core AGUS idea.

## What Reviewers Should Treat As Canonical

If there is any ambiguity:

- treat **Learning Core** as the current Kaggle benchmark
- treat the wider AGUS materials as supporting research artifacts

The canonical Kaggle package lives in:

- [kaggle_benchmark/](../../kaggle_benchmark/)
