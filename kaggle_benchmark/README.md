# AGUS Kaggle Benchmark Package

This folder is the Kaggle-side packaging bundle for **AGUS: Adaptive Generalization Under Shift**.

It is designed for the Kaggle competition **"Measuring Progress Toward AGI - Cognitive Abilities"** and is intentionally scoped to a **minimal Learning-track benchmark slice** first:

- `hidden_rule`
- `shift_transfer`
- `metacog_revision`

That slice is the right first package because it is the cleanest expression of the AGUS thesis:

**Can a model infer a rule, detect that its current strategy is no longer enough, and adapt after new evidence or a representation shift?**

## Files

- `agus_learning_track_notebook.py`
  Notebook-ready runner for Kaggle Benchmarks.
- `benchmark_tasks.py`
  Kaggle Benchmarks SDK task definitions and aggregate benchmark assembly.
- `packaging.py`
  Helpers that build the packaged AGUS JSONL slice from repo datasets.
- `prompts.py`
  Prompt renderers for the Learning-track task families.
- `data/learning_core_v1.jsonl`
  Packaged deterministic benchmark slice.
- `data/manifest.json`
  Package manifest with family counts and source references.
- `KAGGLE_UI_CHECKLIST.md`
  Final manual Kaggle-side checklist.

## What Codex Automated

- Packaged a deterministic AGUS Learning-track slice for Kaggle.
- Implemented real `kaggle_benchmarks` tasks for:
  - hidden-rule adaptation
  - shift-transfer
  - metacognitive revision
- Added an aggregate benchmark task named `agus_learning_track_v1`.
- Added a notebook-ready script that runs the benchmark and produces the Kaggle task/run files.

## What Must Still Be Done In Kaggle

- Create the benchmark notebook in Kaggle.
- Run the notebook in Kaggle so the official task/run files are generated in the Kaggle environment.
- Use Kaggle’s benchmark notebook flow to keep the main AGUS task artifact.
- Publish the benchmark notebook / benchmark entity in Kaggle.
- Attach the resulting benchmark project link to the competition writeup.

## Minimal Kaggle Workflow

1. In Kaggle, open [https://www.kaggle.com/benchmarks/tasks/new](https://www.kaggle.com/benchmarks/tasks/new).
2. Upload or copy this `kaggle_benchmark/` folder into the notebook workspace.
3. Run:

```python
!python kaggle_benchmark/agus_learning_track_notebook.py
```

4. After the run completes, keep the main benchmark artifact:

```python
%choose agus_learning_track_v1
```

5. Verify that the notebook run produced the AGUS `.task.json` and `.run.json` files in the working directory.
6. Publish the notebook/benchmark through Kaggle’s UI.
7. Copy the Kaggle benchmark project link and attach it to the AGUS writeup submission.

## Notes

- This package intentionally prioritizes **submission correctness** over full-suite breadth.
- It does **not** assume any paid API usage.
- It uses Kaggle’s official `kaggle-benchmarks` notebook flow, as documented in the official Kaggle Benchmarks library:
  - Kaggle README: [https://github.com/Kaggle/kaggle-benchmarks](https://github.com/Kaggle/kaggle-benchmarks)
  - Notebook entrypoint: [https://www.kaggle.com/benchmarks/tasks/new](https://www.kaggle.com/benchmarks/tasks/new)
