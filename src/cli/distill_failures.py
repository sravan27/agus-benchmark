"""CLI for AGUS failure distillation."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.failure_distillation import compare_distilled_failures, distill_run_failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill AGUS failure cases into benchmark-ready evidence.")
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Completed evaluation run directories.")
    parser.add_argument("--comparison-name", type=str, default=None)
    args = parser.parse_args()

    summaries = [distill_run_failures(run_dir) for run_dir in args.run_dirs]

    payload = {
        "runs": [summary["run_name"] for summary in summaries],
        "single_run_outputs": [str(Path(summary["run_dir"]) / "distilled_failures.json") for summary in summaries],
    }

    if len(args.run_dirs) > 1:
        comparison_name = args.comparison_name or "_vs_".join(run_dir.name for run_dir in args.run_dirs[:3])
        comparison_dir = Path("data/evals/comparisons") / comparison_name
        comparison_summary = compare_distilled_failures(args.run_dirs, comparison_dir)
        payload["comparison_dir"] = str(comparison_dir)
        payload["top_separating_weaknesses"] = comparison_summary["most_separating_weaknesses"][:3]

    print(payload)


if __name__ == "__main__":
    main()
