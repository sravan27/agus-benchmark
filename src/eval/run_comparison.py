"""Cross-run comparison utilities for AGUS evaluation artifacts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from src.utils.io_utils import load_json, save_json


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body_lines])


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    return value


def _load_run_bundle(run_dir: Path) -> dict[str, Any]:
    aggregate = load_json(run_dir / "aggregate_summary.json")
    failure_cases = load_json(run_dir / "failure_cases.json")
    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "aggregate": aggregate,
        "failure_cases": failure_cases,
    }


def _extract_run_composition(bundle: dict[str, Any]) -> dict[str, Any]:
    aggregate = bundle["aggregate"]
    if "run_composition" in aggregate:
        return aggregate["run_composition"]
    static_summary = aggregate.get("static_summary", {})
    if "run_composition" in static_summary:
        return static_summary["run_composition"]
    interactive_summary = aggregate.get("interactive_summary", {})
    if "run_composition" in interactive_summary:
        return interactive_summary["run_composition"]
    return {
        "tasks_planned_per_family": {},
        "interactive_sessions_planned_per_family": {},
        "families_planned": [],
        "families_skipped": [],
        "families_absent": [],
    }


def _top_failure_reason_counts(failure_cases: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in failure_cases:
        for reason in row.get("failure_reasons", []):
            counts[reason] += 1
    return {reason: counts[reason] for reason in sorted(counts)}


def _top_failure_differences(run_bundles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    per_run_counts = {
        bundle["run_name"]: _top_failure_reason_counts(bundle["failure_cases"])
        for bundle in run_bundles
    }
    reasons = sorted({reason for counts in per_run_counts.values() for reason in counts})
    rows: list[dict[str, Any]] = []
    for reason in reasons:
        counts = {run_name: counts.get(reason, 0) for run_name, counts in per_run_counts.items()}
        delta = max(counts.values()) - min(counts.values())
        rows.append({"reason": reason, "counts": counts, "delta": delta})
    rows.sort(key=lambda row: (row["delta"], row["reason"]), reverse=True)
    return rows[:5]


def _run_overview_rows(run_bundles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for bundle in run_bundles:
        aggregate = bundle["aggregate"]
        adapter = aggregate["adapter"]
        rows.append(
            {
                "run_name": bundle["run_name"],
                "adapter": adapter.get("name"),
                "model": adapter.get("model"),
                "tasks": aggregate["num_tasks_requested"],
                "errors": aggregate["num_errors"],
                "accuracy": _round(aggregate["static_summary"]["accuracy"]),
                "belief_trajectory_quality": _round(
                    aggregate["interactive_summary"].get("belief_trajectory_quality", 0.0)
                ),
                "episode_cognitive_flexibility_score": _round(
                    aggregate["interactive_summary"].get("episode_cognitive_flexibility_score", 0.0)
                ),
                "failure_count": aggregate["failure_count"],
            }
        )
    return rows


def _family_comparison_table(run_bundles: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    families = sorted(
        {
            family
            for bundle in run_bundles
            for family in bundle["aggregate"].get(key, {})
        }
    )
    table: dict[str, dict[str, float]] = {}
    for family in families:
        table[family] = {}
        for bundle in run_bundles:
            table[family][bundle["run_name"]] = _round(bundle["aggregate"].get(key, {}).get(family, 0.0))
    return table


def _interactive_metric_table(run_bundles: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    ignore = {"num_sessions", "run_composition"}
    metrics = sorted(
        {
            metric
            for bundle in run_bundles
            for metric, value in bundle["aggregate"]["interactive_summary"].items()
            if metric not in ignore and isinstance(value, (int, float))
        }
    )
    table: dict[str, dict[str, float]] = {}
    for metric in metrics:
        table[metric] = {}
        for bundle in run_bundles:
            table[metric][bundle["run_name"]] = _round(
                bundle["aggregate"]["interactive_summary"].get(metric, 0.0)
            )
    return table


def _composition_mismatch(run_bundles: list[dict[str, Any]]) -> bool:
    compositions = [
        _extract_run_composition(bundle)["tasks_planned_per_family"]
        for bundle in run_bundles
    ]
    return any(composition != compositions[0] for composition in compositions[1:]) if compositions else False


def _top_insights(
    run_overview: list[dict[str, Any]],
    static_family: dict[str, dict[str, float]],
    failure_differences: list[dict[str, Any]],
    composition_mismatch: bool,
) -> list[str]:
    if not run_overview:
        return ["No runs were supplied for comparison."]

    best_accuracy = max(run_overview, key=lambda row: (row["accuracy"], row["run_name"]))
    best_interactive = max(
        run_overview,
        key=lambda row: (row["belief_trajectory_quality"], row["run_name"]),
    )
    insights = [
        f"`{best_accuracy['run_name']}` had the highest overall static accuracy at {best_accuracy['accuracy']}.",
        (
            f"`{best_interactive['run_name']}` had the strongest belief trajectory quality at "
            f"{best_interactive['belief_trajectory_quality']}."
        ),
    ]

    largest_family_gap = None
    for family, scores in static_family.items():
        values = list(scores.values())
        gap = max(values) - min(values)
        candidate = (gap, family)
        if largest_family_gap is None or candidate > largest_family_gap:
            largest_family_gap = candidate
    if largest_family_gap is not None:
        gap, family = largest_family_gap
        insights.append(f"`{family}` showed the largest static between-run spread at {round(gap, 4)}.")

    if failure_differences:
        top_failure = failure_differences[0]
        insights.append(
            f"The biggest failure-mode divergence was `{top_failure['reason']}` with a cross-run delta of "
            f"{top_failure['delta']}."
        )

    if composition_mismatch:
        insights.append(
            "Run compositions differed across compared runs, so family-level deltas should be interpreted with care."
        )
    else:
        insights.append("All compared runs used matching family composition, making the metric deltas directly comparable.")
    return insights


def compare_evaluation_runs(run_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    """Compare multiple completed run directories and write summary artifacts."""
    run_bundles = [_load_run_bundle(run_dir) for run_dir in run_dirs]
    run_overview = _run_overview_rows(run_bundles)
    static_family = _family_comparison_table(run_bundles, "static_family_average_score")
    interactive_family = _family_comparison_table(run_bundles, "interactive_family_average_score")
    interactive_metrics = _interactive_metric_table(run_bundles)
    failure_differences = _top_failure_differences(run_bundles)
    composition_mismatch = _composition_mismatch(run_bundles)
    insights = _top_insights(run_overview, static_family, failure_differences, composition_mismatch)

    summary = {
        "runs": run_overview,
        "static_family_comparison": static_family,
        "interactive_family_comparison": interactive_family,
        "interactive_metric_comparison": interactive_metrics,
        "top_failure_differences": failure_differences,
        "top_insights": insights,
        "composition_mismatch": composition_mismatch,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "comparison_summary.json", summary)

    run_headers = [
        "run_name",
        "adapter",
        "model",
        "tasks",
        "accuracy",
        "belief_trajectory_quality",
        "episode_cognitive_flexibility_score",
        "failure_count",
        "errors",
    ]
    overview_rows = [[row[header] for header in run_headers] for row in run_overview]

    run_names = [row["run_name"] for row in run_overview]
    static_rows = [[family, *(scores.get(run_name, 0.0) for run_name in run_names)] for family, scores in static_family.items()]
    interactive_rows = [
        [family, *(scores.get(run_name, 0.0) for run_name in run_names)]
        for family, scores in interactive_family.items()
    ]
    interactive_metric_rows = [
        [metric, *(scores.get(run_name, 0.0) for run_name in run_names)]
        for metric, scores in interactive_metrics.items()
    ]
    failure_rows = [
        [row["reason"], row["delta"], *[row["counts"].get(run_name, 0) for run_name in run_names]]
        for row in failure_differences
    ]

    comparison_table = "\n\n".join(
        [
            "# AGUS Run Comparison",
            "## Overview",
            _markdown_table(run_headers, overview_rows),
            "## Static Family Average Score",
            _markdown_table(["family", *run_names], static_rows) if static_rows else "_No static family data._",
            "## Interactive Family Average Score",
            _markdown_table(["family", *run_names], interactive_rows) if interactive_rows else "_No interactive family data._",
            "## Interactive Metrics",
            _markdown_table(["metric", *run_names], interactive_metric_rows)
            if interactive_metric_rows
            else "_No interactive metrics._",
            "## Top Failure-Mode Differences",
            _markdown_table(["reason", "delta", *run_names], failure_rows) if failure_rows else "_No failure deltas._",
        ]
    )
    (output_dir / "comparison_table.md").write_text(comparison_table, encoding="utf-8")

    insights_text = "# Top Insights\n\n" + "\n".join(f"- {insight}" for insight in insights)
    (output_dir / "top_insights.md").write_text(insights_text, encoding="utf-8")
    return summary
