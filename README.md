# AGUS: Adaptive Generalization Under Shift

AGUS is a **Learning-track benchmark** for the Kaggle competition **"Measuring Progress Toward AGI - Cognitive Abilities."** Its core claim is simple:

**static correctness and adaptive reasoning quality can diverge sharply.**

In the current local-model results:

- `llama3.1:8b` scores higher on static accuracy: `0.6179`
- `qwen2.5:7b` scores higher on `belief_trajectory_quality`: `0.7281` versus `0.5434`
- `qwen2.5:7b` is also less brittle overall on `trajectory_instability_index`: `0.2626` versus `0.3348`, lower is better

AGUS is not just another static reasoning dataset. It is designed to test **adaptive generalization under shift**: whether a model can infer a rule, detect change, revise its hypothesis, recalibrate confidence, resist distractors, and maintain coherent belief updates across short interactive episodes.

Submission-oriented docs:

- [final_learning_track_submission.md](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/final_learning_track_submission.md)
- [final_submission_summary.md](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/final_submission_summary.md)
- [final_track_positioning.md](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/final_track_positioning.md)
- [final_results_table.md](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/final_results_table.md)
- [final_failure_gallery.md](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/final_failure_gallery.md)
- [final_submission_checklist.md](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/final_submission_checklist.md)
- [final_requirements_audit.md](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/final_requirements_audit.md)

The design goal is to test **dynamic cognition**, not static recall.

Instead of rewarding benchmark familiarity, AGUS asks whether a model can:

- infer new rules from sparse evidence
- notice when the old strategy stops working
- adapt after a shift
- transfer structure across surface representations
- express uncertainty and revise after contradiction
- resist distractors that look salient but are irrelevant

## What Is Implemented in v1

Five task families are included:

1. `hidden_rule`: infer a latent rule, then adapt after a rule change
2. `shift_transfer`: preserve a learned rule across representation remapping
3. `metacog_revision`: answer with confidence, then revise after corrective evidence
4. `attention_distractors`: recover the true rule while ignoring structured decoys
5. `social_miniworlds`: infer beliefs, knowledge access, and trust under incomplete information

The repo also includes:

- deterministic synthetic generators
- JSON and JSONL export
- a model-agnostic scoring harness
- an interactive evaluation loop for dynamic revision
- preview and scoring demo notebooks
- benchmark design docs
- unit tests

## Project Structure

```text
measuring_agi_project/
  README.md
  requirements.txt
  pyproject.toml
  .gitignore
  data/
    samples/
    generated/
  docs/
    benchmark_design.md
    research_note.md
  notebooks/
    01_task_preview.ipynb
    02_scoring_demo.ipynb
  src/
    config.py
    utils/
    generators/
    curation/
    scoring/
    eval/
    schemas/
    cli/
  tests/
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run tests:

```bash
PYTHONPATH=. pytest -q
```

## Generate Tasks

Generate 100 tasks per family and export JSON plus JSONL:

```bash
PYTHONPATH=. python -m src.cli.generate_tasks --project-root . --count-per-family 100
```

Outputs are written to `data/generated/`:

- `hidden_rule.json`
- `shift_transfer.json`
- `metacog_revision.json`
- `attention_distractors.json`
- `social_miniworlds.json`
- `agus_v1_all.json`

## Adversarial Curation

Generation alone gives coverage, but it does not guarantee that retained tasks are diagnostic. AGUS Curation runs a deterministic suite of weak baseline probes over generated tasks and keeps the ones that are less solvable by shallow shortcuts.

Run the curation pass:

```bash
PYTHONPATH=. python -m src.cli.run_curation --project-root .
```

Outputs are written to `data/generated/curation/`:

- `curated_tasks.json`
- `rejected_tasks.json`
- `curation_report.json`

The curation report summarizes:

- baseline solve rates across weak heuristic solvers
- shortcut vulnerability
- revision and trajectory signal
- per-family retention rates
- top high-signal tasks by family

## Adversarial Refinement Loop

AGUS now closes the loop between curation and generation. Weak-solver wins are not treated as the end of the story. They become signals for how future samples should change.

Run one refinement cycle:

```bash
PYTHONPATH=. python -m src.cli.run_refinement_cycle --project-root .
```

This cycle:

1. inspects `curation_report.json` and `rejected_tasks.json`
2. summarizes the main generator weaknesses by family
3. regenerates stronger `attention_distractors` and `shift_transfer` samples
4. reruns curation on the refreshed tasks
5. saves before/after comparison artifacts

Outputs are written to `data/generated/refinement/`:

- `refinement_summary.json`
- `pre_post_curation_comparison.json`
- `refined_tasks.json`
- `refined_curation_report.json`

## Probe-Conditioned Generator Search

AGUS now uses weak-solver probes not only to filter tasks and refine generators manually, but also to search over generator settings directly. This makes benchmark construction more systematic: candidate generator configurations are evaluated under adversarial curation, ranked, and compared using diagnostic signal rather than arbitrary tuning.

Run the search:

```bash
PYTHONPATH=. python -m src.cli.run_search --count-per-config 24 --emit-best-batches
```

Outputs are written to `data/generated/search/`:

- `search_results.json`
- `best_generator_configs.json`
- `search_summary.json`

Optional best-batch exports are written under `data/generated/search/best_batches/`.

The current search focuses on:

- `attention_distractors`
  - `distractor_diversity_level`
  - `cue_delay_level`
  - `anti_template_strength`
  - `adversarial_query_mode`
- `shift_transfer`
  - `remap_composition_depth`
  - `bridge_representation_mode`
  - `anti_anchor_strength`
  - `latent_rule_mix`

## Search-Conditioned Refinement

AGUS now closes the loop one step further: refinement can promote winning search configurations directly into the next generation pass.

Run search-conditioned refinement:

```bash
PYTHONPATH=. python -m src.cli.run_refinement_cycle --mode search_conditioned
```

This mode:

1. inspects curation failures
2. loads existing search winners, or runs search if needed
3. promotes the winning generator configs for targeted families
4. regenerates refined tasks with those promoted configs
5. reruns curation
6. compares search-conditioned refinement against both the original benchmark and the prior manual refinement pass

Additional outputs are written to `data/generated/refinement/`:

- `search_conditioned_refinement_summary.json`
- `search_promoted_configs.json`
- `search_conditioned_pre_post_curation_comparison.json`

This makes AGUS more adaptive as a benchmark-development pipeline because the same weak-solver pressure used for curation now also shapes future refinement decisions.

## Scoring Workflow

The scoring harness supports:

- `accuracy`
- `adaptation_speed`
- `transfer_score`
- `calibration_score`
- `revision_quality`
- `distractor_robustness`

Run the demo evaluator:

```bash
PYTHONPATH=. python -m src.cli.evaluate_demo --tasks data/generated/agus_v1_all.json --predictions data/samples/demo_predictions.json
```

If `data/samples/demo_predictions.json` does not exist yet, the demo CLI creates a deterministic baseline prediction file automatically.

## AGUS Interactive

AGUS Interactive upgrades the benchmark from static answer checking to lightweight 2-turn and 3-5 turn micro-episodes:

1. observe examples
2. state a hypothesis, answer, and confidence
3. receive new evidence, contradiction, or representation shift
4. revise the hypothesis, answer, and confidence
5. continue through additional turns when the task requires attention recovery or social belief tracking

Run the interactive demo:

```bash
PYTHONPATH=. python -m src.cli.run_interactive_demo --project-root .
```

The interactive demo saves a JSON artifact at `data/samples/interactive_demo_results.json` with session traces and dynamic metrics.

## AGUS v2 Counterfactual Branching

AGUS v2 adds a tighter coherence test on top of interactive evaluation. Instead of following only one observed trajectory, a model can now be evaluated across **2-4 nearby counterfactual continuations** of the same base episode.

Current branching support includes:

- `hidden_rule`
- `shift_transfer`
- `social_miniworlds`

This makes it possible to ask whether the model:

- updates only when contradiction truly appears
- preserves invariants across nearby representation shifts
- keeps world state stable while changing only the right agent beliefs

Run the counterfactual demo:

```bash
PYTHONPATH=. python -m src.cli.run_counterfactual_branches \
  --adapter mock-shallow \
  --run-name agus_v2_counterfactual_demo \
  --max-per-family 1
```

See [agus_v2_counterfactuals.md](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/agus_v2_counterfactuals.md) for the benchmark rationale.

## Local-First Model Evaluation

AGUS Model Evaluation v1 is designed to work without paid APIs. The harness supports:

- deterministic mock adapters for validation and CI
- local adapters for Ollama-style inference
- resumable run directories for overnight local benchmarking
- structured artifacts for scores, traces, progress, and failures

Run a local mock baseline:

```bash
PYTHONPATH=. python -m src.cli.run_model_eval --adapter mock-noisy --run-name mock_noisy_demo
```

Run a balanced smoke test across families:

```bash
PYTHONPATH=. python -m src.cli.run_model_eval \
  --adapter mock-noisy \
  --run-name mock_balanced_smoke \
  --profile balanced25
```

Run against a local Ollama model:

```bash
PYTHONPATH=. python -m src.cli.run_model_eval \
  --adapter ollama \
  --model llama3.1:8b \
  --run-name ollama_llama31 \
  --keep-alive 30m \
  --resume
```

Available convenience profiles:

- `smoke`
- `balanced25`
- `overnight100`

Profiles only prefill existing options such as `--max-tasks`, `--balanced`, and `--per-family-max`. Direct flags still work normally.

Each run writes a dedicated folder under `data/evals/<run_name>/` with:

- `config.json`
- `predictions.jsonl`
- `interactive_sessions.jsonl`
- `static_summary.json`
- `interactive_summary.json`
- `aggregate_summary.json`
- `failure_cases.json`
- `progress.json`

Balanced runs report planned and completed work by family, and `progress.json` is driven by real completed static evaluations plus real completed interactive sessions. This keeps the benchmark usable on an M2 Max for overnight local runs while preserving a clean path for future optional API adapters.

Compare completed runs:

```bash
PYTHONPATH=. python -m src.cli.compare_eval_runs \
  data/evals/mock_noisy_full \
  data/evals/mock_shallow_full \
  --comparison-name mock_noisy_vs_shallow
```

Comparison artifacts are written to `data/evals/comparisons/<name>/`:

- `comparison_summary.json`
- `comparison_table.md`
- `top_insights.md`

Distill the most revealing failures from one or more completed runs:

```bash
PYTHONPATH=. python -m src.cli.distill_failures data/evals/mock_noisy_full
```

For multi-run weakness comparison:

```bash
PYTHONPATH=. python -m src.cli.distill_failures \
  data/evals/mock_noisy_full \
  data/evals/mock_shallow_full \
  --comparison-name noisy_vs_shallow_weaknesses
```

This writes per-run `distilled_failures.json` and `signature_weaknesses.md`, plus cross-run weakness comparison artifacts under `data/evals/comparisons/<name>/`.

Analyze trajectory instability from completed runs:

```bash
PYTHONPATH=. python -m src.cli.analyze_instability data/evals/mock_noisy_full
```

For cross-run instability comparison:

```bash
PYTHONPATH=. python -m src.cli.analyze_instability \
  data/evals/mock_noisy_full \
  data/evals/mock_shallow_full \
  --comparison-name noisy_vs_shallow_instability
```

This writes per-run `instability_summary.json` and `instability_highlights.md`, plus comparison-time `instability_comparison.json` and `instability_insights.md`.

Supported interactive families now include:

- `hidden_rule`
- `shift_transfer`
- `metacog_revision`
- `attention_distractors`
- `social_miniworlds`

## First Local Findings

The first balanced local-model runs already show why AGUS is useful. Static accuracy and dynamic reasoning quality do not line up cleanly:

- `llama3.1:8b` leads on static accuracy: `0.6179`
- `qwen2.5:7b` leads on `belief_trajectory_quality`: `0.7281` versus `0.5434`
- `qwen2.5:7b` also has lower overall `trajectory_instability_index`: `0.2626` versus `0.3348`

This suggests AGUS is capturing a real tradeoff between frozen-task correctness and adaptive reasoning quality. The current local runs also show that the biggest separating weakness types are:

- `overconfident_error`
- `static_dynamic_gap`
- `social_belief_confusion`

The strongest safe headline is not that one local model is simply better. It is that AGUS reveals cognitive differences that a static benchmark would compress into a single accuracy score.

Judge-facing summaries are in:

- [results_packet.md](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/results_packet.md)
- [local_model_findings.md](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/local_model_findings.md)
- [failure_gallery.md](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/failure_gallery.md)
- [key_claims.md](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/key_claims.md)

### Why AGUS Interactive Is More Novel Than Static Reasoning Datasets

Static reasoning datasets mostly reward one-shot competence on a frozen task. AGUS Interactive instead measures whether a model can revise a live belief state after new evidence arrives and sustain that revision over a short episode.

This makes the benchmark more diagnostic of dynamic cognition because it explicitly scores:

- online hypothesis revision
- uncertainty trajectory
- contradiction sensitivity
- adaptation quality after evidence
- attention recovery after distractor capture
- trust and belief revision under social conflict
- cognitive flexibility under shift

AGUS now measures several distinct forms of revision inside the same benchmark family:

- rule revision
- representation revision
- confidence revision
- attention recovery
- trust revision
- false-belief and belief-state revision

The emphasis is not on chat UX or product interaction. The emphasis is on a minimal, reproducible protocol for evaluating dynamic cognition in a research setting.

## Why Adversarial Curation Improves Benchmark Legitimacy

Synthetic generation is useful for coverage, but synthetic generation alone is not enough. If too many episodes are easy for shallow heuristics, the benchmark becomes easier to game and harder to defend as a measurement of adaptive cognition.

AGUS Curation addresses this by probing each task with lightweight shortcut agents that intentionally fail in different ways:

- surface pattern matching without genuine abstraction
- over-anchoring on the most obvious local rule
- salience-following behavior under distractors
- failure to revise after contradiction
- trust-naive social reasoning
- failure to transfer across representation shifts

The goal is not leaderboard optimization. The goal is to retain tasks that better separate shallow patterning from online adaptation, revision, transfer, and belief tracking.

## Why The Refinement Loop Matters

This makes AGUS iterative rather than one-shot. Instead of freezing an initial synthetic generator and hoping it remains diagnostic, the benchmark can now:

- identify where shallow heuristics still succeed
- harden the weakest families with targeted generator changes
- compare retention and signal quality before and after refinement

In practice, this reduces templatedness, improves transfer depth, and makes the benchmark development process itself easier to defend to Kaggle judges and researchers.

## Why Probe-Conditioned Search Matters

This search layer makes AGUS less dependent on one-off manual refinement. Generator settings are now evaluated by how much diagnostic pressure they create under the benchmark’s own weak-solver probes.

That matters because it means AGUS development is guided by:

- shortcut resistance
- adaptive-cognition signal
- curation-aware retention tradeoffs
- family-specific discriminative value

rather than by arbitrary intuition or one static generator configuration.

## Benchmark Philosophy

AGUS is designed around **adaptive generalization under shift**:

- performance should depend on online updating, not static pattern recall
- confidence should matter when evidence is underdetermined
- transfer should preserve latent structure under representation change
- distractors should punish shallow salience-following behavior
- social reasoning should require belief tracking and incentive inference
- interactive evaluation should reward belief change when the evidence warrants it

For more detail, see [benchmark_design.md](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/benchmark_design.md) and [research_note.md](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/docs/research_note.md).
