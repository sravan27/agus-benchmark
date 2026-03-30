"""Validation helpers for AGUS task records."""

from __future__ import annotations

from typing import Iterable

from src.schemas.task_schema import REQUIRED_TASK_FIELDS


def validate_task_dict(task: dict) -> None:
    """Raise if a task does not satisfy the common schema contract."""
    missing = [field for field in REQUIRED_TASK_FIELDS if field not in task]
    if missing:
        raise ValueError(f"Task {task.get('task_id', '<missing>')} missing fields: {missing}")

    if not isinstance(task["examples"], list):
        raise ValueError(f"Task {task['task_id']} has non-list examples")

    if not isinstance(task["metadata"], dict):
        raise ValueError(f"Task {task['task_id']} has non-dict metadata")


def validate_tasks(tasks: Iterable[dict]) -> None:
    """Validate a full dataset."""
    seen_ids: set[str] = set()
    for task in tasks:
        validate_task_dict(task)
        task_id = task["task_id"]
        if task_id in seen_ids:
            raise ValueError(f"Duplicate task_id detected: {task_id}")
        seen_ids.add(task_id)

