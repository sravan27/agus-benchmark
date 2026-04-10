# AGUS Kaggle UI Checklist

## Manual Steps In Kaggle

1. Open a benchmark notebook from:
   [https://www.kaggle.com/benchmarks/tasks/new](https://www.kaggle.com/benchmarks/tasks/new)
2. Attach the Kaggle dataset that contains the `kaggle_benchmark/` folder.
3. Run the canonical notebook entrypoint directly from the mounted dataset:

```python
!python /kaggle/input/<your-dataset-slug>/kaggle_benchmark/agus_learning_track_notebook.py
```

4. Keep the main benchmark artifact:

```python
%choose agus_learning_track_v1
```

5. Confirm the notebook created AGUS benchmark task/run artifacts in the Kaggle working directory.
6. Publish the benchmark notebook or benchmark entity in Kaggle.
7. Copy the resulting Kaggle benchmark project link.
8. Open the competition writeup submission.
9. Attach the benchmark project link to the writeup.
10. Confirm the writeup plus attached benchmark satisfy the competition’s mandatory benchmark requirement.

## What Is Still Manual

- Creating the Kaggle benchmark notebook
- Uploading the `kaggle_benchmark/` package as a Kaggle dataset
- Publishing the benchmark through Kaggle’s UI
- Attaching the benchmark project link to the writeup
- Final UI verification before submission

## What To Verify Before Final Submit

- The AGUS benchmark notebook ran without errors
- `agus_learning_track_v1` is the selected benchmark artifact
- The task data came from the packaged Learning-track slice in `kaggle_benchmark/data/learning_core_v1.jsonl`
- The writeup links to the correct benchmark project
- The writeup and benchmark are both published/visible as required by Kaggle
