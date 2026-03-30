"""Evaluation harness for AGUS predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.eval.interactive_runner import summarize_interactive_sessions
from src.scoring.metrics import (
    score_accuracy,
    score_adaptation_speed,
    score_calibration,
    score_distractor_robustness,
    score_revision_quality,
    score_transfer,
)
from src.utils.io_utils import load_json, save_json


def index_predictions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index prediction rows by task id."""
    return {row["task_id"]: row for row in rows}


def evaluate_predictions(tasks: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute all benchmark metrics from task records and prediction rows."""
    predictions = index_predictions(prediction_rows)
    accuracy_payload = score_accuracy(tasks, predictions)
    distractor_payload = score_distractor_robustness(tasks, predictions)

    return {
        **accuracy_payload,
        "adaptation_speed": score_adaptation_speed(tasks, predictions),
        "transfer_score": score_transfer(tasks, predictions),
        "calibration_score": score_calibration(tasks, predictions),
        "revision_quality": score_revision_quality(tasks, predictions),
        **distractor_payload,
        "num_tasks": len(tasks),
        "num_predictions": len(prediction_rows),
    }


def evaluate_from_paths(tasks_path: Path, predictions_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    """Load tasks and predictions from disk and optionally save scores."""
    tasks = load_json(tasks_path)
    prediction_rows = load_json(predictions_path)
    results = evaluate_predictions(tasks, prediction_rows)
    if output_path is not None:
        save_json(output_path, results)
    return results


def evaluate_interactive_sessions(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary metrics for interactive session records."""
    return summarize_interactive_sessions(sessions)
