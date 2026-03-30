"""CLI for comparing completed AGUS evaluation runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.run_comparison import compare_evaluation_runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare completed AGUS evaluation runs.")
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Completed run directories to compare.")
    parser.add_argument("--output-root", type=Path, default=Path("data/evals/comparisons"))
    parser.add_argument("--comparison-name", type=str, default=None)
    args = parser.parse_args()

    comparison_name = args.comparison_name or "_vs_".join(run_dir.name for run_dir in args.run_dirs[:3])
    output_dir = args.output_root / comparison_name
    summary = compare_evaluation_runs(args.run_dirs, output_dir)
    print(
        {
            "comparison_dir": str(output_dir),
            "runs": [row["run_name"] for row in summary["runs"]],
            "top_insights": summary["top_insights"][:3],
        }
    )


if __name__ == "__main__":
    main()
