"""CLI for multi-slice AGUS robustness comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.robustness_analysis import compare_robustness_runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare AGUS runs across original and replication slices.")
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Completed run directories.")
    parser.add_argument("--output-root", type=Path, default=Path("data/evals/comparisons"))
    parser.add_argument("--comparison-name", type=str, default="robustness_check")
    args = parser.parse_args()

    output_dir = args.output_root / args.comparison_name
    summary = compare_robustness_runs(args.run_dirs, output_dir)
    print(
        {
            "comparison_dir": str(output_dir),
            "models_compared": summary["models_compared"],
            "slice_names": summary["slice_names"],
            "core_pattern_held_on": summary["core_pattern_held_on"],
            "static_accuracy_ranking_held_on": summary["replication_counts"]["static_accuracy_ranking"],
            "belief_trajectory_quality_ranking_held_on": summary["replication_counts"][
                "belief_trajectory_quality_ranking"
            ],
        }
    )


if __name__ == "__main__":
    main()
