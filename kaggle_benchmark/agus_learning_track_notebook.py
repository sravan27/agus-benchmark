from __future__ import annotations

import argparse
from pathlib import Path
import sys


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


def main() -> None:
    args = parse_args()
    slice_path = Path(args.slice_path)
    run = run_learning_track_benchmark(slice_path=slice_path)
    print(f"AGUS Learning-track score: {run.result:.4f}")
    print("Kaggle Benchmarks task/run files were written to the active working directory.")
    print("If you are in a Kaggle notebook, finish with: %choose agus_learning_track_v1")


if __name__ == "__main__":
    main()
