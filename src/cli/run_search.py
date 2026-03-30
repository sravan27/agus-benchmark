"""CLI for probe-conditioned generator search in AGUS."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.search.probe_conditioned_search import run_probe_conditioned_search


def _parse_families(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run probe-conditioned generator search for AGUS.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--families",
        type=str,
        default="attention_distractors,shift_transfer",
        help="Comma-separated family list.",
    )
    parser.add_argument("--count-per-config", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated/search"))
    parser.add_argument("--emit-best-batches", action="store_true")
    parser.add_argument("--best-batch-size", type=int, default=12)
    args = parser.parse_args()

    families = _parse_families(args.families)
    results = run_probe_conditioned_search(
        families=families,
        count_per_config=args.count_per_config,
        emit_best_batches=args.emit_best_batches,
        best_batch_size=args.best_batch_size,
        output_dir=args.output_dir,
    )

    for family in families:
        winner = results["search_summary"]["families"][family]["winner"]
        fallback = results["search_summary"]["families"][family]["fallback"]
        print(
            f"{family}: winner={winner['config_id']} score={winner['search_metrics']['curation_weighted_score']} "
            f"kept_rate={winner['search_metrics']['kept_rate']}"
        )
        if fallback:
            print(
                f"{family}: fallback={fallback['config_id']} score={fallback['search_metrics']['curation_weighted_score']} "
                f"kept_rate={fallback['search_metrics']['kept_rate']}"
            )
    print(f"saved search artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
