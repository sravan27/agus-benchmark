"""CLI for a lightweight AGUS Interactive demo."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.cli.generate_tasks import generate_all
from src.eval.interactive_runner import (
    SUPPORTED_INTERACTIVE_FAMILIES,
    make_placeholder_responder,
    run_interactive_sessions,
    summarize_interactive_sessions,
)
from src.utils.io_utils import load_json, save_json


def _load_or_generate_tasks(project_root: Path, tasks_path: Path) -> list[dict]:
    if tasks_path.exists():
        tasks = load_json(tasks_path)
        present_families = {task["family"] for task in tasks}
        if SUPPORTED_INTERACTIVE_FAMILIES.issubset(present_families):
            return tasks

    datasets = generate_all(project_root, count_per_family=20)
    combined: list[dict] = []
    for rows in datasets.values():
        combined.extend(rows)
    save_json(tasks_path, combined)
    return combined


def _select_tasks(tasks: list[dict], tasks_per_family: int) -> list[dict]:
    selected: list[dict] = []
    for family in sorted(SUPPORTED_INTERACTIVE_FAMILIES):
        family_rows = [task for task in tasks if task["family"] == family][:tasks_per_family]
        selected.extend(family_rows)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight AGUS interactive evaluation demo.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--tasks", type=Path, default=Path("data/generated/agus_v1_all.json"))
    parser.add_argument("--output", type=Path, default=Path("data/samples/interactive_demo_results.json"))
    parser.add_argument("--tasks-per-family", type=int, default=2)
    args = parser.parse_args()

    tasks = _load_or_generate_tasks(args.project_root, args.tasks)
    selected_tasks = _select_tasks(tasks, args.tasks_per_family)
    sessions = run_interactive_sessions(selected_tasks, responder_factory=lambda idx: make_placeholder_responder(200 + idx))
    metrics = summarize_interactive_sessions(sessions)

    artifact = {
        "selected_task_ids": [task["task_id"] for task in selected_tasks],
        "supported_families": sorted(SUPPORTED_INTERACTIVE_FAMILIES),
        "sessions": sessions,
        "interactive_metrics": metrics,
    }
    save_json(args.output, artifact)

    print(f"Ran AGUS Interactive demo on {len(selected_tasks)} tasks.")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")
    print(f"Saved interactive results to {args.output}")


if __name__ == "__main__":
    main()
