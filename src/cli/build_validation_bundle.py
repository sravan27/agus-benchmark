"""CLI for building an AGUS validation bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.validation_bundle import build_validation_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an AGUS hostile-review validation bundle.")
    parser.add_argument(
        "--curation-comparison",
        type=Path,
        default=Path("data/generated/refinement/pre_post_curation_comparison.json"),
    )
    parser.add_argument(
        "--search-conditioned-comparison",
        type=Path,
        default=Path("data/generated/refinement/search_conditioned_pre_post_curation_comparison.json"),
    )
    parser.add_argument(
        "--shallow-run",
        type=Path,
        default=Path("data/evals/mock_shallow_balanced_interactive100"),
    )
    parser.add_argument(
        "--adaptive-run",
        action="append",
        type=Path,
        default=None,
        help="Adaptive run directory. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--rank-shift-run",
        action="append",
        type=Path,
        default=None,
        help="Run directory used for the static-vs-dynamic rank-shift check. Repeat for multiple runs.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/evals/comparisons"))
    parser.add_argument("--bundle-name", type=str, default="validation_bundle_v1")
    args = parser.parse_args()

    adaptive_runs = args.adaptive_run or [
        Path("data/evals/llama31_balanced_interactive100"),
        Path("data/evals/qwen25_balanced_interactive100"),
        Path("data/evals/mistralnemo_balanced_interactive100"),
    ]
    rank_shift_runs = args.rank_shift_run or adaptive_runs
    search_conditioned = (
        args.search_conditioned_comparison if args.search_conditioned_comparison.exists() else None
    )

    output_dir = args.output_root / args.bundle_name
    summary = build_validation_bundle(
        curation_comparison_path=args.curation_comparison,
        search_conditioned_comparison_path=search_conditioned,
        shallow_run_dir=args.shallow_run,
        adaptive_run_dirs=adaptive_runs,
        rank_shift_run_dirs=rank_shift_runs,
        output_dir=output_dir,
    )
    print(
        {
            "comparison_dir": str(output_dir),
            "rank_shift_present": summary["rank_shift"]["rank_shift_present"],
            "matching_composition": summary["rank_shift"]["matching_composition"],
            "top_insights": summary["top_insights"][:3],
        }
    )


if __name__ == "__main__":
    main()
