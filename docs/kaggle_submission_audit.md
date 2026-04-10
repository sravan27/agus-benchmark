# AGUS Kaggle Submission Audit

## Scope

This audit covers the entire repository, with primary focus on the Kaggle submission surface:

- `kaggle_benchmark/`
- the packaged Learning-track slice in `kaggle_benchmark/data/`
- the benchmark task definitions and notebook entrypoint
- tests that validate Kaggle packaging behavior

The goal was operational reliability, not benchmark redesign.

## What Submission-Ready Means

For AGUS, submission-ready means:

1. The Learning-core benchmark package can be mounted directly from a Kaggle dataset.
2. The Kaggle notebook can run one canonical entrypoint without ad hoc patching.
3. The task JSONL schema matches the Kaggle task function signatures.
4. The Kaggle Benchmarks SDK can infer task result types and execute tasks cleanly.
5. The package does not depend on repo-local source files that will not exist in Kaggle.
6. The run path to create the benchmark and attach it to the writeup is unambiguous.

## Highest-Risk Failure Points Found

### 1. Package-only Kaggle runs could still depend on repo-level generated data

Why this was risky:
- `kaggle_benchmark/packaging.py` originally built slices from repo files under `data/generated/`.
- If only `kaggle_benchmark/` were mounted in Kaggle, those source files would not exist.

Fix applied:
- `build_slice()` now falls back to the packaged slice in `kaggle_benchmark/data/learning_core_v1.jsonl`.
- This makes the Learning-core package self-contained.

Status:
- Answered.

### 2. Aggregate task execution could break on extra dataframe columns

Why this was risky:
- The Kaggle Benchmarks SDK forwards dataframe row fields into task calls.
- If evaluation data included extra columns not present in a task function signature, task execution could fail with unexpected keyword argument errors.

Fix applied:
- Added column filtering in `kaggle_benchmark/benchmark_tasks.py`.
- Family-specific evaluation frames are now reduced to exactly the parameters expected by each task function.
- Missing required columns now raise a clear error early.

Status:
- Answered.

### 3. Notebook entrypoint was too script-centric

Why this was risky:
- The previous flow leaned on a script execution path and did not expose one clean callable notebook entrypoint.
- That makes Kaggle notebook usage more fragile and harder to explain.

Fix applied:
- `kaggle_benchmark/agus_learning_track_notebook.py` now exposes `run_notebook_entrypoint()` and `resolve_slice_path()`.
- Script-path import handling only runs when executed as a script.

Status:
- Partially answered.

Notes:
- The canonical Kaggle path is still script execution from the mounted dataset because that is the least fragile operationally.
- A callable import path also exists now for copied-workspace scenarios.

### 4. Kaggle SDK task return types were not auto-inferable

Why this was risky:
- The Kaggle Benchmarks SDK expects task functions to expose a registered result type for auto-inference.
- `-> None` annotations caused SDK compatibility failures.

Fix applied:
- All `@kbench.task` episode functions now return `bool`.
- Each task ends with `return True` after assertions succeed.
- The deferred annotation behavior that hid `bool` from the SDK was removed from that file.

Status:
- Answered.

### 5. Generated Kaggle artifacts could leak into the repo root

Why this was risky:
- Local benchmark dry runs can emit `.task.json` and `.run.json` files in the repo root.
- If not ignored, they create confusion about what is source and what is generated output.

Fix applied:
- Added `*.task.json` and `*.run.json` to `.gitignore`.

Status:
- Answered.

## Schema Alignment Audit

The packaged Learning-core slice in `kaggle_benchmark/data/learning_core_v1.jsonl` was checked against the three Kaggle task definitions:

- `hidden_rule_episode`
- `shift_transfer_episode`
- `metacog_revision_episode`

Result:
- Each family row contains exactly the parameter set expected by its corresponding task function.
- No missing fields were found in the packaged slice.
- Extra-column crashes are now prevented during benchmark assembly by explicit filtering.

## Packaging Integrity Audit

Verified:

- `kaggle_benchmark/data/learning_core_v1.jsonl` contains the packaged deterministic benchmark rows.
- `kaggle_benchmark/data/manifest.json` matches the packaged Learning-core slice:
  - 30 total rows
  - 10 each for `hidden_rule`, `shift_transfer`, `metacog_revision`
- `kaggle_benchmark/prompts.py` is consumed by the Kaggle task definitions only through package-local imports.
- `kaggle_benchmark/__init__.py` exports a notebook-safe entrypoint.
- The package can be copied out of the repo and imported on its own.

## Tests Added Or Strengthened

Focused Kaggle packaging tests now cover:

- package-only import without repo-level generated sources
- exact JSONL-to-task-schema alignment
- filtering of extra evaluation columns
- notebook-safe callable entrypoint
- Kaggle SDK bool-return compatibility

Validation status:

- `pytest -q tests/test_kaggle_benchmark_packaging.py` -> `8 passed`
- `pytest -q` -> `72 passed`

## Canonical Kaggle Entry Point

The canonical Kaggle execution path is:

```python
!python /kaggle/input/<your-dataset-slug>/kaggle_benchmark/agus_learning_track_notebook.py
```

Why this is the canonical path:

- it runs directly from the mounted dataset
- it avoids copying files into the notebook workspace
- it avoids manual `sys.path` editing in notebook cells
- it uses the packaged Learning-core slice already bundled with the Kaggle package

## Canonical Kaggle Checklist

1. Upload the `kaggle_benchmark/` folder as a Kaggle dataset.
2. Open a benchmark notebook from `https://www.kaggle.com/benchmarks/tasks/new`.
3. Attach the dataset that contains `kaggle_benchmark/`.
4. Run:

```python
!python /kaggle/input/<your-dataset-slug>/kaggle_benchmark/agus_learning_track_notebook.py
```

5. Keep the benchmark artifact:

```python
%choose agus_learning_track_v1
```

6. Confirm the notebook produced `.task.json` and `.run.json` artifacts in the Kaggle working directory.
7. Publish the benchmark notebook / benchmark entity.
8. Copy the benchmark project link.
9. Attach that link to the competition writeup.

## Final Audit Verdict

The Kaggle package is now import-and-run ready for the Learning-core submission slice.

Most important improvements:
- self-contained package behavior
- notebook-safe entrypoint
- SDK-compatible task return types
- explicit schema alignment protection
- one canonical Kaggle execution path

Remaining manual work is Kaggle-side only:
- uploading the package as a dataset
- running the benchmark notebook
- publishing the benchmark
- attaching the benchmark link to the writeup
