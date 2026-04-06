"""CLI for comparing AGUS counterfactual evaluation runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.counterfactual_comparison import compare_counterfactual_runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare completed AGUS counterfactual runs.")
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Completed counterfactual run directories.")
    parser.add_argument("--output-root", type=Path, default=Path("data/evals/comparisons"))
    parser.add_argument("--comparison-name", type=str, default="counterfactual_comparison")
    args = parser.parse_args()

    output_dir = args.output_root / args.comparison_name
    summary = compare_counterfactual_runs(args.run_dirs, output_dir)
    print(
        {
            "comparison_dir": str(output_dir),
            "runs": [row["run_name"] for row in summary["runs"]],
            "top_insights": summary["top_insights"][:3],
        }
    )


if __name__ == "__main__":
    main()
