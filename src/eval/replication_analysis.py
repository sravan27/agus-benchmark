"""Replication utilities for AGUS core-result robustness checks."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from src.utils.io_utils import load_json, load_jsonl, save_json

WEAKNESS_PROXY_NAMES = (
    "overconfident_error_proxy",
    "static_dynamic_gap_proxy",
    "social_belief_confusion_proxy",
)
SOCIAL_CONFUSION_REASONS = {"belief_state_inconsistent", "belief_state_miss", "trust_not_revised"}


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body_lines])


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    return value


def _safe_slice_name(aggregate: dict[str, Any]) -> str:
    return aggregate.get("run_composition", {}).get("balanced_slice_name", "original")


def _load_run_bundle(run_dir: Path) -> dict[str, Any]:
    aggregate = load_json(run_dir / "aggregate_summary.json")
    failure_cases = load_json(run_dir / "failure_cases.json")
    sessions = load_jsonl(run_dir / "interactive_sessions.jsonl")
    adapter = aggregate.get("adapter", {})
    model = adapter.get("model") or adapter.get("name") or run_dir.name
    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "model": model,
        "slice_name": _safe_slice_name(aggregate),
        "aggregate": aggregate,
        "failure_cases": failure_cases,
        "sessions": sessions,
    }


def _execution_health(bundle: dict[str, Any]) -> dict[str, Any]:
    aggregate = bundle["aggregate"]
    run_composition = aggregate.get("run_composition", {})
    interactive_planned = sum(run_composition.get("interactive_sessions_planned_per_family", {}).values())
    static_planned = int(aggregate.get("num_tasks_requested", 0))
    static_success = int(aggregate.get("num_static_predictions", 0))
    interactive_success = int(aggregate.get("num_interactive_sessions", 0))
    num_errors = int(aggregate.get("num_errors", 0))

    invalid_reasons: list[str] = []
    warnings: list[str] = []
    if static_planned and static_success == 0:
        invalid_reasons.append("no_successful_static_predictions")
    if interactive_planned and interactive_success == 0:
        invalid_reasons.append("no_successful_interactive_sessions")
    if num_errors:
        warnings.append("run_contains_errors")

    return {
        "static_planned": static_planned,
        "static_success": static_success,
        "interactive_planned": interactive_planned,
        "interactive_success": interactive_success,
        "num_errors": num_errors,
        "valid_for_replication": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
        "warnings": warnings,
    }


def _failure_rows_by_task_and_source(failure_cases: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in failure_cases:
        grouped[(row["task_id"], row["source"])].append(row)
    return grouped


def _weakness_proxy_counts(bundle: dict[str, Any]) -> dict[str, int]:
    grouped_failures = _failure_rows_by_task_and_source(bundle["failure_cases"])
    counts = {name: 0 for name in WEAKNESS_PROXY_NAMES}
    for session in bundle["sessions"]:
        task_id = session["task_id"]
        revised_correct = bool(session["derived"]["revised_correct"])
        revised_confidence = float(session["response"].get("revised_confidence", 0.0) or 0.0)
        if not revised_correct and revised_confidence >= 0.75:
            counts["overconfident_error_proxy"] += 1

        static_correct = (task_id, "static") not in grouped_failures
        if static_correct and not revised_correct:
            counts["static_dynamic_gap_proxy"] += 1

        if session["family"] == "social_miniworlds":
            reasons = [
                reason
                for row in grouped_failures.get((task_id, "interactive"), [])
                for reason in row.get("failure_reasons", [])
            ]
            if any(reason in SOCIAL_CONFUSION_REASONS for reason in reasons):
                counts["social_belief_confusion_proxy"] += 1
    return counts


def _pairwise_ranking(
    bundles: dict[str, dict[str, Any]],
    *,
    metric_name: str,
    metric_getter,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    ranking = sorted(
        (
            {
                "model": model,
                "value": _round(metric_getter(bundle)),
            }
            for model, bundle in bundles.items()
        ),
        key=lambda row: ((-row["value"]) if higher_is_better else row["value"], row["model"]),
    )
    return {
        "metric": metric_name,
        "higher_is_better": higher_is_better,
        "ranking": ranking,
        "leader": ranking[0]["model"] if ranking else None,
    }


def _leader(counts: dict[str, int]) -> str | None:
    if not counts:
        return None
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
        return "tie"
    return ordered[0][0]


def _bundle_metric(bundle: dict[str, Any], path: tuple[str, ...]) -> float:
    current: Any = bundle["aggregate"]
    for key in path:
        current = current.get(key, {})
    return float(current or 0.0)


def _core_run_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    aggregate = bundle["aggregate"]
    execution_health = _execution_health(bundle)
    return {
        "run_name": bundle["run_name"],
        "slice_name": bundle["slice_name"],
        "execution_health": execution_health,
        "static_accuracy": _round(_bundle_metric(bundle, ("static_summary", "accuracy"))),
        "belief_trajectory_quality": _round(
            _bundle_metric(bundle, ("interactive_summary", "belief_trajectory_quality"))
        ),
        "episode_cognitive_flexibility_score": _round(
            _bundle_metric(bundle, ("interactive_summary", "episode_cognitive_flexibility_score"))
        ),
        "failure_count": aggregate.get("failure_count", 0),
        "weakness_proxies": _weakness_proxy_counts(bundle),
    }


def compare_replication_runs(
    *,
    original_run_dirs: list[Path],
    replication_run_dirs: list[Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Compare an original slice against a fresh deterministic replication slice."""
    original_loaded = [_load_run_bundle(run_dir) for run_dir in original_run_dirs]
    replication_loaded = [_load_run_bundle(run_dir) for run_dir in replication_run_dirs]
    original_bundles = {bundle["model"]: bundle for bundle in original_loaded}
    replication_bundles = {bundle["model"]: bundle for bundle in replication_loaded}

    shared_models = sorted(set(original_bundles) & set(replication_bundles))
    original_bundles = {model: original_bundles[model] for model in shared_models}
    replication_bundles = {model: replication_bundles[model] for model in shared_models}

    invalid_runs = []
    for label, bundles in (("original", original_bundles), ("replication", replication_bundles)):
        for model, bundle in bundles.items():
            health = _execution_health(bundle)
            if not health["valid_for_replication"]:
                invalid_runs.append(
                    {
                        "slice": label,
                        "model": model,
                        "run_name": bundle["run_name"],
                        "invalid_reasons": health["invalid_reasons"],
                    }
                )

    if invalid_runs:
        details = "; ".join(
            f"{row['run_name']} ({row['slice']}): {', '.join(row['invalid_reasons'])}" for row in invalid_runs
        )
        raise ValueError(
            "Replication comparison refused invalid runs with no successful model outputs: "
            f"{details}"
        )

    original_static = _pairwise_ranking(
        original_bundles,
        metric_name="static_accuracy",
        metric_getter=lambda bundle: _bundle_metric(bundle, ("static_summary", "accuracy")),
        higher_is_better=True,
    )
    replication_static = _pairwise_ranking(
        replication_bundles,
        metric_name="static_accuracy",
        metric_getter=lambda bundle: _bundle_metric(bundle, ("static_summary", "accuracy")),
        higher_is_better=True,
    )
    original_dynamic = _pairwise_ranking(
        original_bundles,
        metric_name="belief_trajectory_quality",
        metric_getter=lambda bundle: _bundle_metric(bundle, ("interactive_summary", "belief_trajectory_quality")),
        higher_is_better=True,
    )
    replication_dynamic = _pairwise_ranking(
        replication_bundles,
        metric_name="belief_trajectory_quality",
        metric_getter=lambda bundle: _bundle_metric(bundle, ("interactive_summary", "belief_trajectory_quality")),
        higher_is_better=True,
    )

    weakness_proxy_checks: dict[str, Any] = {}
    for proxy_name in WEAKNESS_PROXY_NAMES:
        original_counts = {
            model: _core_run_summary(bundle)["weakness_proxies"][proxy_name]
            for model, bundle in original_bundles.items()
        }
        replication_counts = {
            model: _core_run_summary(bundle)["weakness_proxies"][proxy_name]
            for model, bundle in replication_bundles.items()
        }
        weakness_proxy_checks[proxy_name] = {
            "original_counts": original_counts,
            "replication_counts": replication_counts,
            "original_higher": _leader(original_counts),
            "replication_higher": _leader(replication_counts),
            "direction_replicated": _leader(original_counts) == _leader(replication_counts),
        }

    original_divergence = original_static["leader"] != original_dynamic["leader"]
    replication_divergence = replication_static["leader"] != replication_dynamic["leader"]
    divergence_check = {
        "original_static_leader": original_static["leader"],
        "original_dynamic_leader": original_dynamic["leader"],
        "replication_static_leader": replication_static["leader"],
        "replication_dynamic_leader": replication_dynamic["leader"],
        "original_divergence": original_divergence,
        "replication_divergence": replication_divergence,
        "divergence_replicated": (
            original_divergence
            and replication_divergence
            and original_static["leader"] == replication_static["leader"]
            and original_dynamic["leader"] == replication_dynamic["leader"]
        ),
    }

    summary = {
        "models_compared": shared_models,
        "original_slice": {
            "slice_name": next(iter(original_bundles.values()))["slice_name"] if original_bundles else "original",
            "runs": [_core_run_summary(bundle) for bundle in original_bundles.values()],
        },
        "replication_slice": {
            "slice_name": next(iter(replication_bundles.values()))["slice_name"] if replication_bundles else "replication",
            "runs": [_core_run_summary(bundle) for bundle in replication_bundles.values()],
        },
        "ranking_checks": {
            "static_accuracy_ranking": {
                "original": original_static,
                "replication": replication_static,
                "ranking_replicated": [row["model"] for row in original_static["ranking"]]
                == [row["model"] for row in replication_static["ranking"]],
            },
            "belief_trajectory_quality_ranking": {
                "original": original_dynamic,
                "replication": replication_dynamic,
                "ranking_replicated": [row["model"] for row in original_dynamic["ranking"]]
                == [row["model"] for row in replication_dynamic["ranking"]],
            },
            "static_vs_dynamic_divergence": divergence_check,
        },
        "weakness_proxy_checks": weakness_proxy_checks,
    }
    summary["core_pattern_replicated"] = (
        summary["ranking_checks"]["static_accuracy_ranking"]["ranking_replicated"]
        and summary["ranking_checks"]["belief_trajectory_quality_ranking"]["ranking_replicated"]
        and divergence_check["divergence_replicated"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "replication_summary.json", summary)

    overview_headers = ["slice", "model", "static_accuracy", "belief_trajectory_quality", "episode_cognitive_flexibility_score"]
    overview_rows = []
    for label, slice_payload in (
        ("original", summary["original_slice"]),
        ("replication", summary["replication_slice"]),
    ):
        for row in slice_payload["runs"]:
            overview_rows.append(
                [
                    label,
                    row["run_name"],
                    row["static_accuracy"],
                    row["belief_trajectory_quality"],
                    row["episode_cognitive_flexibility_score"],
                ]
            )

    weakness_headers = ["proxy", "original_higher", "replication_higher", "direction_replicated"]
    weakness_rows = [
        [
            proxy_name,
            payload["original_higher"],
            payload["replication_higher"],
            payload["direction_replicated"],
        ]
        for proxy_name, payload in weakness_proxy_checks.items()
    ]

    insights = [
        f"Static accuracy leader on original slice: `{original_static['leader']}`.",
        f"Static accuracy leader on replication slice: `{replication_static['leader']}`.",
        f"Belief trajectory quality leader on original slice: `{original_dynamic['leader']}`.",
        f"Belief trajectory quality leader on replication slice: `{replication_dynamic['leader']}`.",
        (
            "Static-vs-dynamic divergence persisted on the replication slice."
            if divergence_check["divergence_replicated"]
            else "Static-vs-dynamic divergence did not cleanly persist on the replication slice."
        ),
    ]

    markdown = "\n\n".join(
        [
            "# AGUS Replication Summary",
            f"- models_compared: `{', '.join(shared_models)}`",
            f"- core_pattern_replicated: `{summary['core_pattern_replicated']}`",
            "## Slice Overview",
            _markdown_table(overview_headers, overview_rows) if overview_rows else "_No runs supplied._",
            "## Ranking Checks",
            "\n".join(
                [
                    f"- static_accuracy_ranking_replicated: `{summary['ranking_checks']['static_accuracy_ranking']['ranking_replicated']}`",
                    f"- belief_trajectory_quality_ranking_replicated: `{summary['ranking_checks']['belief_trajectory_quality_ranking']['ranking_replicated']}`",
                    f"- static_vs_dynamic_divergence_replicated: `{divergence_check['divergence_replicated']}`",
                ]
            ),
            "## Weakness Proxy Checks",
            _markdown_table(weakness_headers, weakness_rows) if weakness_rows else "_No weakness proxy data._",
            "## Top Insights",
            "\n".join(f"- {insight}" for insight in insights),
        ]
    )
    (output_dir / "replication_summary.md").write_text(markdown, encoding="utf-8")
    return summary
