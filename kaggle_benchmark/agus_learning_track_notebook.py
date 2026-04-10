from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:  # pragma: no cover - script execution path for Kaggle notebooks.
    PACKAGE_DIR = Path(__file__).resolve().parent
    REPO_PARENT = PACKAGE_DIR.parent
    if str(REPO_PARENT) not in sys.path:
        sys.path.insert(0, str(REPO_PARENT))

from kaggle_benchmark.benchmark_tasks import default_slice_path, run_learning_track_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Notebook-ready entrypoint for the AGUS Learning-track Kaggle benchmark."
    )
    parser.add_argument(
        "--slice-path",
        default=str(default_slice_path()),
        help="Path to the packaged AGUS benchmark JSONL slice.",
    )
    return parser.parse_args()


def resolve_slice_path(slice_path: str | Path | None = None) -> Path:
    resolved = Path(slice_path) if slice_path else default_slice_path()
    if not resolved.exists():
        raise FileNotFoundError(
            f"AGUS packaged slice not found at {resolved}. "
            "Mount the kaggle_benchmark package and make sure `data/learning_core_v1.jsonl` is present."
        )
    return resolved


def run_notebook_entrypoint(slice_path: str | Path | None = None) -> dict[str, str | float]:
    resolved = resolve_slice_path(slice_path)
    run = run_learning_track_benchmark(slice_path=resolved)
    payload = {
        "benchmark_name": "agus_learning_track_v1",
        "slice_path": str(resolved),
        "score": float(run.result),
    }
    print(f"AGUS Learning-track score: {payload['score']:.4f}")
    print("Kaggle Benchmarks task/run files were written to the active working directory.")
    print("Finish in the Kaggle notebook with: %choose agus_learning_track_v1")
    return payload


def main() -> None:
    args = parse_args()
    run_notebook_entrypoint(args.slice_path)


if __name__ == "__main__":
    main()
