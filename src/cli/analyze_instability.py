"""CLI for AGUS instability analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.instability_analysis import analyze_run_instability, compare_run_instability


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze AGUS trajectory instability from completed runs.")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--comparison-name", type=str, default=None)
    args = parser.parse_args()

    summaries = [analyze_run_instability(run_dir) for run_dir in args.run_dirs]
    payload = {
        "runs": [summary["run_name"] for summary in summaries],
        "single_run_outputs": [str(Path(run_dir) / "instability_summary.json") for run_dir in args.run_dirs],
    }

    if len(args.run_dirs) > 1:
        comparison_name = args.comparison_name or "_vs_".join(run_dir.name for run_dir in args.run_dirs[:3])
        comparison_dir = Path("data/evals/comparisons") / comparison_name
        comparison = compare_run_instability(args.run_dirs, comparison_dir)
        payload["comparison_dir"] = str(comparison_dir)
        payload["most_brittle_model"] = comparison["most_brittle_model"]
        payload["most_stable_under_contradiction"] = comparison["most_stable_under_contradiction"]

    print(payload)


if __name__ == "__main__":
    main()
