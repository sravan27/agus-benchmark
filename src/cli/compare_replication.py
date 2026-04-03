"""CLI for AGUS replication-slice robustness checks."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.replication_analysis import compare_replication_runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare original and replication-slice AGUS runs.")
    parser.add_argument("--original", action="append", type=Path, required=True, help="Original-slice run directory.")
    parser.add_argument("--replication", action="append", type=Path, required=True, help="Replication-slice run directory.")
    parser.add_argument("--output-root", type=Path, default=Path("data/evals/comparisons"))
    parser.add_argument("--comparison-name", type=str, default="replication_check")
    args = parser.parse_args()

    output_dir = args.output_root / args.comparison_name
    summary = compare_replication_runs(
        original_run_dirs=args.original,
        replication_run_dirs=args.replication,
        output_dir=output_dir,
    )
    print(
        {
            "comparison_dir": str(output_dir),
            "models_compared": summary["models_compared"],
            "core_pattern_replicated": summary["core_pattern_replicated"],
            "static_accuracy_ranking_replicated": summary["ranking_checks"]["static_accuracy_ranking"][
                "ranking_replicated"
            ],
            "belief_trajectory_quality_ranking_replicated": summary["ranking_checks"][
                "belief_trajectory_quality_ranking"
            ]["ranking_replicated"],
            "static_vs_dynamic_divergence_replicated": summary["ranking_checks"]["static_vs_dynamic_divergence"][
                "divergence_replicated"
            ],
        }
    )


if __name__ == "__main__":
    main()
