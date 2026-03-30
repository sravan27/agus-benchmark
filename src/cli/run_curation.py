"""CLI for AGUS adversarial curation."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.cli.generate_tasks import generate_all
from src.curation.adversarial_curation import curate_tasks
from src.utils.io_utils import load_json, save_json


def _load_or_generate_tasks(project_root: Path, tasks_path: Path) -> list[dict]:
    if tasks_path.exists():
        return load_json(tasks_path)

    datasets = generate_all(project_root, count_per_family=100)
    combined: list[dict] = []
    for rows in datasets.values():
        combined.extend(rows)
    save_json(tasks_path, combined)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adversarial curation over AGUS tasks.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--tasks", type=Path, default=Path("data/generated/agus_v1_all.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated/curation"))
    args = parser.parse_args()

    tasks = _load_or_generate_tasks(args.project_root, args.tasks)
    results = curate_tasks(tasks)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    curated_path = args.output_dir / "curated_tasks.json"
    rejected_path = args.output_dir / "rejected_tasks.json"
    report_path = args.output_dir / "curation_report.json"

    save_json(curated_path, results["curated_tasks"])
    save_json(rejected_path, results["rejected_tasks"])
    save_json(report_path, results["curation_report"])

    report = results["curation_report"]
    print(f"processed: {report['total_tasks_processed']}")
    print(f"kept: {report['kept_count']}")
    print(f"review: {report['review_count']}")
    print(f"rejected: {report['rejected_count']}")
    if report["most_common_rejection_reasons"]:
        top_reason, top_count = next(iter(report["most_common_rejection_reasons"].items()))
        print(f"top_rejection_reason: {top_reason} ({top_count})")
    print(f"saved curated tasks to {curated_path}")
    print(f"saved rejected tasks to {rejected_path}")
    print(f"saved curation report to {report_path}")


if __name__ == "__main__":
    main()

