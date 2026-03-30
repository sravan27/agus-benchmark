# AGUS: Adaptive Generalization Under Shift

AGUS is a synthetic benchmark suite for the Kaggle competition **"Measuring Progress Toward AGI - Cognitive Abilities."** The design goal is to test **dynamic cognition**, not static recall.

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

Supported interactive families now include:

- `hidden_rule`
- `shift_transfer`
- `metacog_revision`
- `attention_distractors`
- `social_miniworlds`

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

## Benchmark Philosophy

AGUS is designed around **adaptive generalization under shift**:

- performance should depend on online updating, not static pattern recall
- confidence should matter when evidence is underdetermined
- transfer should preserve latent structure under representation change
- distractors should punish shallow salience-following behavior
- social reasoning should require belief tracking and incentive inference
- interactive evaluation should reward belief change when the evidence warrants it

For more detail, see [benchmark_design.md](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/measuring_agi_project/docs/benchmark_design.md) and [research_note.md](/Users/sravansridhar/Documents/Codex/Kaggle-benchmarks/measuring_agi_project/docs/research_note.md).
