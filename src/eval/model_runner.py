"""Local-first model evaluation runner for AGUS."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import time
from pathlib import Path
from typing import Any

from src.eval.adapters import ModelAdapter
from src.eval.interactive_runner import SUPPORTED_INTERACTIVE_FAMILIES, run_interactive_session
from src.scoring.evaluator import evaluate_interactive_sessions, evaluate_predictions
from src.scoring.metrics import exact_match
from src.utils.io_utils import append_jsonl, load_json, load_jsonl, save_json


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _family_filter(tasks: list[dict[str, Any]], families: set[str] | None = None) -> list[dict[str, Any]]:
    if not families:
        return list(tasks)
    return [task for task in tasks if task["family"] in families]


def _family_order(tasks: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(task["family"] for task in tasks))


def _count_tasks_by_family(tasks: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for task in tasks:
        counts[task["family"]] += 1
    return {family: counts[family] for family in sorted(counts)}


def _index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        indexed[row["task_id"]] = row
    return indexed


def _rows_for_task_ids(rows: list[dict[str, Any]], task_ids: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("task_id") in task_ids]


def _bucket_tasks(tasks: list[dict[str, Any]], family_order: list[str]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {family: [] for family in family_order}
    for task in tasks:
        buckets[task["family"]].append(task)
    return buckets


def _round_robin_take(
    buckets: dict[str, list[dict[str, Any]]],
    family_order: list[str],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    pointers = {family: 0 for family in family_order}
    while True:
        added = False
        for family in family_order:
            family_tasks = buckets.get(family, [])
            pointer = pointers[family]
            if pointer >= len(family_tasks):
                continue
            selected.append(family_tasks[pointer])
            pointers[family] += 1
            added = True
            if limit is not None and len(selected) >= limit:
                return selected
        if not added:
            return selected


def select_evaluation_tasks(
    tasks: list[dict[str, Any]],
    *,
    families: set[str] | None = None,
    max_tasks: int | None = None,
    balanced: bool = False,
    per_family_max: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a deterministic evaluation subset."""
    filtered = _family_filter(tasks, families)
    source_family_order = _family_order(filtered)
    source_counts = _count_tasks_by_family(filtered)

    if balanced:
        buckets = _bucket_tasks(filtered, source_family_order)
        if per_family_max is not None:
            for family in source_family_order:
                buckets[family] = buckets[family][:per_family_max]
        selected = _round_robin_take(buckets, source_family_order, max_tasks)
    else:
        selected = []
        family_counts: dict[str, int] = defaultdict(int)
        for task in filtered:
            family = task["family"]
            if per_family_max is not None and family_counts[family] >= per_family_max:
                continue
            selected.append(task)
            family_counts[family] += 1
            if max_tasks is not None and len(selected) >= max_tasks:
                break

    selected_counts = _count_tasks_by_family(selected)
    requested_families = sorted(families) if families else source_family_order
    families_absent = [family for family in requested_families if family not in source_counts]
    families_skipped = [family for family in source_family_order if selected_counts.get(family, 0) == 0]
    selection_meta = {
        "balanced": balanced,
        "per_family_max": per_family_max,
        "max_tasks": max_tasks,
        "families_requested": requested_families,
        "families_present_in_source": source_family_order,
        "families_planned": sorted(selected_counts),
        "families_absent": families_absent,
        "families_skipped": families_skipped,
        "source_tasks_per_family": source_counts,
        "tasks_planned_per_family": selected_counts,
    }
    return selected, selection_meta


def _score_static_task(task: dict[str, Any], prediction: dict[str, Any] | None) -> dict[str, Any]:
    family = task["family"]
    failure_reasons: list[str] = []
    if prediction is None:
        return {
            "task_id": task["task_id"],
            "family": family,
            "score": 0.0,
            "correct": False,
            "failure_reasons": ["missing_prediction"],
            "prediction": None,
            "expected": task["answer"],
        }

    if family == "hidden_rule":
        checks: list[bool] = []
        induction_predictions = prediction.get("induction_predictions", [])
        shift_predictions = prediction.get("shift_predictions", [])
        for idx, target in enumerate(task["answer"]["induction_targets"]):
            ok = idx < len(induction_predictions) and exact_match(induction_predictions[idx], target)
            checks.append(ok)
        for idx, target in enumerate(task["answer"]["shift_targets"]):
            ok = idx < len(shift_predictions) and exact_match(shift_predictions[idx], target)
            checks.append(ok)
        if not all(checks[: len(task["answer"]["induction_targets"])]):
            failure_reasons.append("induction_errors")
        if not all(checks[len(task["answer"]["induction_targets"]) :]):
            failure_reasons.append("shift_errors")

    elif family == "shift_transfer":
        source_ok = exact_match(prediction.get("source_prediction"), task["answer"]["source_target"])
        transfer_ok = exact_match(prediction.get("transfer_prediction"), task["answer"]["transfer_target"])
        checks = [source_ok, transfer_ok]
        if not source_ok:
            failure_reasons.append("source_rule_miss")
        if not transfer_ok:
            failure_reasons.append("transfer_failure")

    elif family == "metacog_revision":
        initial_ok = any(
            exact_match(prediction.get("initial_answer"), candidate)
            for candidate in task["answer"]["acceptable_initial_targets"]
        )
        revised_ok = exact_match(prediction.get("revised_answer"), task["answer"]["revised_target"])
        checks = [initial_ok, revised_ok]
        if not revised_ok:
            failure_reasons.append("revision_failure")
        if prediction.get("revised_answer") == prediction.get("initial_answer"):
            failure_reasons.append("no_answer_revision")

    elif family == "attention_distractors":
        answer_ok = exact_match(prediction.get("prediction"), task["answer"]["target"])
        checks = [answer_ok]
        if prediction.get("selected_signal") == "distractor_sequence":
            failure_reasons.append("followed_distractor")
        if not answer_ok:
            failure_reasons.append("final_answer_incorrect")

    elif family == "social_miniworlds":
        actual_ok = exact_match(prediction.get("actual_location_prediction"), task["answer"]["actual_location"])
        belief_ok = exact_match(prediction.get("belief_prediction"), task["answer"]["belief_of_false_belief_agent"])
        trust_ok = exact_match(prediction.get("trust_prediction"), task["answer"]["most_reliable_agent"])
        checks = [actual_ok, belief_ok, trust_ok]
        if not actual_ok:
            failure_reasons.append("world_state_miss")
        if not belief_ok:
            failure_reasons.append("belief_state_miss")
        if not trust_ok:
            failure_reasons.append("trust_inference_miss")

    else:
        raise ValueError(f"Unsupported family: {family}")

    score = round(sum(checks) / len(checks), 4) if checks else 0.0
    return {
        "task_id": task["task_id"],
        "family": family,
        "score": score,
        "correct": all(checks),
        "failure_reasons": failure_reasons,
        "prediction": prediction,
        "expected": task["answer"],
    }


def _score_interactive_session(session: dict[str, Any]) -> dict[str, Any]:
    response = session["response"]
    derived = session["derived"]
    expected = session["expected"]
    failure_reasons: list[str] = []
    if not derived["revised_correct"]:
        failure_reasons.append("final_answer_incorrect")
    if expected.get("contradiction_expected") and not response["contradiction_detected"]:
        failure_reasons.append("missed_contradiction")
    if session["family"] == "attention_distractors" and derived.get("recovery_turn") is None:
        failure_reasons.append("failed_attention_recovery")
    if session["family"] == "social_miniworlds":
        reliable = expected["reliable_agent"]
        final_trust = response.get("trust_scores_by_agent", {})
        if final_trust:
            trusted = max(final_trust.items(), key=lambda item: (item[1], item[0]))[0]
            if trusted != reliable:
                failure_reasons.append("trust_not_revised")
        trace = derived.get("belief_consistency_trace", [])
        if trace and trace[-1] < 1.0:
            failure_reasons.append("belief_state_inconsistent")
    score = round(
        sum(float(flag) for flag in derived.get("turn_correctness", []))
        / max(len(derived.get("turn_correctness", [])), 1),
        4,
    )
    return {
        "task_id": session["task_id"],
        "family": session["family"],
        "episode_type": session["episode_type"],
        "score": score,
        "correct": bool(derived["revised_correct"]),
        "failure_reasons": failure_reasons,
        "response": response,
        "expected": expected,
    }


def _extract_failure_cases(
    static_rows: list[dict[str, Any]],
    interactive_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in static_rows:
        if row["correct"]:
            continue
        failures.append(
            {
                "source": "static",
                "task_id": row["task_id"],
                "family": row["family"],
                "severity": round(1.0 - float(row["score"]), 4),
                "failure_reasons": row["failure_reasons"],
                "prediction": row["prediction"],
                "expected": row["expected"],
            }
        )
    for row in interactive_rows:
        if row["correct"] and not row["failure_reasons"]:
            continue
        failures.append(
            {
                "source": "interactive",
                "task_id": row["task_id"],
                "family": row["family"],
                "episode_type": row["episode_type"],
                "severity": round(1.0 - float(row["score"]), 4),
                "failure_reasons": row["failure_reasons"],
                "response": row["response"],
                "expected": row["expected"],
            }
        )
    failures.sort(key=lambda row: (row["severity"], row["task_id"]), reverse=True)
    return failures


def _family_average(rows: list[dict[str, Any]]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        buckets.setdefault(row["family"], []).append(float(row["score"]))
    return {family: round(sum(values) / len(values), 4) for family, values in sorted(buckets.items())}


def _planned_interactive_task_ids(tasks: list[dict[str, Any]], include_interactive: bool, adapter: ModelAdapter) -> set[str]:
    if not include_interactive or not adapter.supports_interactive:
        return set()
    return {
        task["task_id"]
        for task in tasks
        if task["family"] in SUPPORTED_INTERACTIVE_FAMILIES
    }


def _counts_from_ids(tasks: list[dict[str, Any]], task_ids: set[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for task in tasks:
        if task["task_id"] in task_ids:
            counts[task["family"]] += 1
    return {family: counts[family] for family in sorted(counts)}


def _sum_latencies(rows: list[dict[str, Any]]) -> float:
    total = 0.0
    for row in rows:
        total += float(row.get("_meta", {}).get("latency_seconds", 0.0))
    return round(total, 4)


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _build_progress_payload(
    *,
    adapter: ModelAdapter,
    run_name: str,
    run_dir: Path,
    tasks: list[dict[str, Any]],
    selection_meta: dict[str, Any],
    started_at: str,
    prediction_rows: list[dict[str, Any]],
    session_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    include_interactive: bool,
    resume: bool,
    last_task_id: str | None,
) -> dict[str, Any]:
    task_lookup = {task["task_id"]: task for task in tasks}
    interactive_planned_ids = _planned_interactive_task_ids(tasks, include_interactive, adapter)

    static_success_ids = {row["task_id"] for row in prediction_rows if row["task_id"] in task_lookup}
    interactive_success_ids = {row["task_id"] for row in session_rows if row["task_id"] in interactive_planned_ids}
    static_error_ids = {row["task_id"] for row in error_rows if row["stage"] == "static" and row["task_id"] in task_lookup}
    interactive_error_ids = {
        row["task_id"]
        for row in error_rows
        if row["stage"] == "interactive" and row["task_id"] in interactive_planned_ids
    }

    tasks_finished = len(static_success_ids | static_error_ids)
    interactive_finished = len(interactive_success_ids | interactive_error_ids)
    total_work_planned = len(tasks) + len(interactive_planned_ids)
    total_work_completed = tasks_finished + interactive_finished

    elapsed_seconds = round(
        _sum_latencies(prediction_rows) + _sum_latencies(session_rows) + _sum_latencies(error_rows),
        4,
    )
    avg_seconds_per_item = round(elapsed_seconds / total_work_completed, 4) if total_work_completed else 0.0
    avg_seconds_per_task = round(elapsed_seconds / tasks_finished, 4) if tasks_finished else 0.0
    eta_seconds = (
        round(avg_seconds_per_item * (total_work_planned - total_work_completed), 4)
        if total_work_completed
        else None
    )
    percentage = round((total_work_completed / total_work_planned) * 100.0, 2) if total_work_planned else 100.0

    per_family_completed_counts = _counts_from_ids(tasks, static_success_ids | static_error_ids)
    per_family_interactive_completed_counts = _counts_from_ids(tasks, interactive_success_ids | interactive_error_ids)

    adapter_info = adapter.describe()
    payload = {
        "adapter": adapter_info.get("name"),
        "model": adapter_info.get("model"),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "total_tasks_planned": len(tasks),
        "tasks_completed": tasks_finished,
        "tasks_succeeded": len(static_success_ids),
        "interactive_sessions_planned": len(interactive_planned_ids),
        "interactive_sessions_completed": interactive_finished,
        "interactive_sessions_succeeded": len(interactive_success_ids),
        "families_planned": selection_meta["families_planned"],
        "families_skipped": selection_meta["families_skipped"],
        "families_absent": selection_meta["families_absent"],
        "tasks_planned_per_family": selection_meta["tasks_planned_per_family"],
        "interactive_sessions_planned_per_family": _counts_from_ids(tasks, interactive_planned_ids),
        "per_family_completed_counts": per_family_completed_counts,
        "per_family_interactive_completed_counts": per_family_interactive_completed_counts,
        "started_at": started_at,
        "last_updated_at": _now_iso(),
        "elapsed_seconds": elapsed_seconds,
        "avg_seconds_per_item": avg_seconds_per_item,
        "avg_seconds_per_task": avg_seconds_per_task,
        "eta_seconds": eta_seconds,
        "percentage_complete": percentage,
        "work_items_planned": total_work_planned,
        "work_items_completed": total_work_completed,
        "num_errors": len(error_rows),
        "resume_mode": resume,
        "include_interactive": include_interactive and adapter.supports_interactive,
        "balanced_sampling": selection_meta["balanced"],
        "per_family_max": selection_meta["per_family_max"],
        "max_tasks": selection_meta["max_tasks"],
        "last_task_id": last_task_id,
        "run_complete": total_work_completed >= total_work_planned,
    }
    return payload


def _emit_progress(progress: dict[str, Any]) -> None:
    line = (
        f"[progress] {progress['work_items_completed']}/{progress['work_items_planned']} items "
        f"({progress['percentage_complete']:.1f}%) | "
        f"tasks {progress['tasks_completed']}/{progress['total_tasks_planned']} | "
        f"interactive {progress['interactive_sessions_completed']}/{progress['interactive_sessions_planned']} | "
        f"elapsed {_format_duration(progress['elapsed_seconds'])} | "
        f"avg/item {progress['avg_seconds_per_item']:.2f}s | "
        f"eta {_format_duration(progress['eta_seconds'])} | "
        f"errors {progress['num_errors']}"
    )
    print(line, flush=True)


def run_model_evaluation(
    *,
    tasks_path: Path,
    adapter: ModelAdapter,
    run_dir: Path,
    families: set[str] | None = None,
    max_tasks: int | None = None,
    include_interactive: bool = True,
    resume: bool = True,
    balanced: bool = False,
    per_family_max: int | None = None,
) -> dict[str, Any]:
    """Run a resumable local-first evaluation over AGUS tasks."""
    all_tasks = load_json(tasks_path)
    tasks, selection_meta = select_evaluation_tasks(
        all_tasks,
        families=families,
        max_tasks=max_tasks,
        balanced=balanced,
        per_family_max=per_family_max,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    sessions_path = run_dir / "interactive_sessions.jsonl"
    errors_path = run_dir / "errors.jsonl"
    config_path = run_dir / "config.json"
    progress_path = run_dir / "progress.json"

    task_ids = {task["task_id"] for task in tasks}
    existing_predictions = _rows_for_task_ids(load_jsonl(predictions_path), task_ids) if resume else []
    existing_sessions = _rows_for_task_ids(load_jsonl(sessions_path), task_ids) if resume else []
    existing_errors = _rows_for_task_ids(load_jsonl(errors_path), task_ids) if resume else []
    prediction_index = _index_rows(existing_predictions)
    session_index = _index_rows(existing_sessions)
    error_rows = list(existing_errors)
    error_pairs = {(row["task_id"], row["stage"]) for row in error_rows}

    existing_progress = load_json(progress_path) if resume and progress_path.exists() else {}
    started_at = existing_progress.get("started_at", _now_iso())

    save_json(
        config_path,
        {
            "tasks_path": str(tasks_path),
            "adapter": adapter.describe(),
            "families": sorted(families) if families else None,
            "max_tasks": max_tasks,
            "include_interactive": include_interactive,
            "resume": resume,
            "balanced": balanced,
            "per_family_max": per_family_max,
            "selection": selection_meta,
        },
    )

    initial_progress = _build_progress_payload(
        adapter=adapter,
        run_name=run_dir.name,
        run_dir=run_dir,
        tasks=tasks,
        selection_meta=selection_meta,
        started_at=started_at,
        prediction_rows=list(prediction_index.values()),
        session_rows=list(session_index.values()),
        error_rows=error_rows,
        include_interactive=include_interactive,
        resume=resume,
        last_task_id=None,
    )
    save_json(progress_path, initial_progress)
    if initial_progress["work_items_completed"]:
        _emit_progress(initial_progress)

    for task in tasks:
        static_done = task["task_id"] in prediction_index or (task["task_id"], "static") in error_pairs
        interactive_needed = include_interactive and adapter.supports_interactive and task["family"] in SUPPORTED_INTERACTIVE_FAMILIES
        interactive_done = (
            not interactive_needed
            or task["task_id"] in session_index
            or (task["task_id"], "interactive") in error_pairs
        )
        if static_done and interactive_done:
            continue

        if not static_done:
            started = time.perf_counter()
            try:
                prediction = adapter.predict_task(task)
                prediction_row = dict(prediction)
                prediction_row["task_id"] = task["task_id"]
                prediction_row["family"] = task["family"]
                prediction_row["_meta"] = {"latency_seconds": round(time.perf_counter() - started, 4)}
                append_jsonl(predictions_path, prediction_row)
                prediction_index[task["task_id"]] = prediction_row
            except Exception as exc:  # pragma: no cover - defensive overnight runner path
                error_row = {
                    "task_id": task["task_id"],
                    "family": task["family"],
                    "stage": "static",
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                    "_meta": {"latency_seconds": round(time.perf_counter() - started, 4)},
                }
                append_jsonl(errors_path, error_row)
                error_rows.append(error_row)
                error_pairs.add((task["task_id"], "static"))

            progress = _build_progress_payload(
                adapter=adapter,
                run_name=run_dir.name,
                run_dir=run_dir,
                tasks=tasks,
                selection_meta=selection_meta,
                started_at=started_at,
                prediction_rows=list(prediction_index.values()),
                session_rows=list(session_index.values()),
                error_rows=error_rows,
                include_interactive=include_interactive,
                resume=resume,
                last_task_id=task["task_id"],
            )
            save_json(progress_path, progress)
            _emit_progress(progress)

        if interactive_needed and not interactive_done:
            started = time.perf_counter()
            try:
                session = run_interactive_session(task, adapter.respond_turn)
                session["family"] = task["family"]
                session["_meta"] = {"latency_seconds": round(time.perf_counter() - started, 4)}
                append_jsonl(sessions_path, session)
                session_index[task["task_id"]] = session
            except Exception as exc:  # pragma: no cover - defensive overnight runner path
                error_row = {
                    "task_id": task["task_id"],
                    "family": task["family"],
                    "stage": "interactive",
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                    "_meta": {"latency_seconds": round(time.perf_counter() - started, 4)},
                }
                append_jsonl(errors_path, error_row)
                error_rows.append(error_row)
                error_pairs.add((task["task_id"], "interactive"))

            progress = _build_progress_payload(
                adapter=adapter,
                run_name=run_dir.name,
                run_dir=run_dir,
                tasks=tasks,
                selection_meta=selection_meta,
                started_at=started_at,
                prediction_rows=list(prediction_index.values()),
                session_rows=list(session_index.values()),
                error_rows=error_rows,
                include_interactive=include_interactive,
                resume=resume,
                last_task_id=task["task_id"],
            )
            save_json(progress_path, progress)
            _emit_progress(progress)

    prediction_rows = list(prediction_index.values())
    sessions = list(session_index.values())
    static_task_rows = [_score_static_task(task, prediction_index.get(task["task_id"])) for task in tasks]
    interactive_task_rows = [_score_interactive_session(session) for session in sessions]

    run_composition = {
        **selection_meta,
        "interactive_sessions_planned_per_family": _counts_from_ids(
            tasks,
            _planned_interactive_task_ids(tasks, include_interactive, adapter),
        ),
        "tasks_completed_per_family": _counts_from_ids(tasks, set(prediction_index)),
        "interactive_sessions_completed_per_family": _counts_from_ids(tasks, set(session_index)),
    }

    static_summary = evaluate_predictions(tasks, prediction_rows)
    static_summary["run_composition"] = {
        "families_planned": run_composition["families_planned"],
        "families_skipped": run_composition["families_skipped"],
        "families_absent": run_composition["families_absent"],
        "tasks_planned_per_family": run_composition["tasks_planned_per_family"],
        "tasks_completed_per_family": run_composition["tasks_completed_per_family"],
    }

    interactive_summary = evaluate_interactive_sessions(sessions) if sessions else {"num_sessions": 0}
    interactive_summary["run_composition"] = {
        "families_planned": run_composition["families_planned"],
        "families_skipped": run_composition["families_skipped"],
        "families_absent": run_composition["families_absent"],
        "interactive_sessions_planned_per_family": run_composition["interactive_sessions_planned_per_family"],
        "interactive_sessions_completed_per_family": run_composition["interactive_sessions_completed_per_family"],
    }

    failure_cases = _extract_failure_cases(static_task_rows, interactive_task_rows)
    save_json(run_dir / "task_level_static_results.json", static_task_rows)
    save_json(run_dir / "task_level_interactive_results.json", interactive_task_rows)
    save_json(run_dir / "static_summary.json", static_summary)
    save_json(run_dir / "interactive_summary.json", interactive_summary)
    save_json(run_dir / "failure_cases.json", failure_cases)

    final_progress = _build_progress_payload(
        adapter=adapter,
        run_name=run_dir.name,
        run_dir=run_dir,
        tasks=tasks,
        selection_meta=selection_meta,
        started_at=started_at,
        prediction_rows=prediction_rows,
        session_rows=sessions,
        error_rows=error_rows,
        include_interactive=include_interactive,
        resume=resume,
        last_task_id=tasks[-1]["task_id"] if tasks else None,
    )
    save_json(progress_path, final_progress)

    aggregate_summary = {
        "adapter": adapter.describe(),
        "num_tasks_requested": len(tasks),
        "num_static_predictions": len(prediction_rows),
        "num_interactive_sessions": len(sessions),
        "num_errors": len(error_rows),
        "static_summary": static_summary,
        "interactive_summary": interactive_summary,
        "failure_count": len(failure_cases),
        "static_family_average_score": _family_average(static_task_rows),
        "interactive_family_average_score": _family_average(interactive_task_rows) if interactive_task_rows else {},
        "run_composition": run_composition,
        "progress": final_progress,
    }
    save_json(run_dir / "aggregate_summary.json", aggregate_summary)
    _emit_progress(final_progress)
    return aggregate_summary
