"""Validation bundle builder for hostile-review sanity checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.eval.run_comparison import _extract_run_composition
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


def _load_aggregate(run_dir: Path) -> dict[str, Any]:
    aggregate = load_json(run_dir / "aggregate_summary.json")
    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "aggregate": aggregate,
        "composition": _extract_run_composition({"aggregate": aggregate}),
    }


def _metric_ranking(rows: list[dict[str, Any]], metric_path: tuple[str, ...]) -> list[dict[str, Any]]:
    ranking = []
    for row in rows:
        current: Any = row["aggregate"]
        for key in metric_path:
            current = current.get(key, {})
        ranking.append({"run_name": row["run_name"], "value": _round(float(current or 0.0))})
    ranking.sort(key=lambda item: (-item["value"], item["run_name"]))
    return ranking


def _matching_composition(rows: list[dict[str, Any]]) -> bool:
    compositions = [row["composition"].get("tasks_planned_per_family", {}) for row in rows]
    return all(composition == compositions[0] for composition in compositions[1:]) if compositions else True


def _largest_family_delta(comparison: dict[str, Any], metric_name: str) -> dict[str, Any] | None:
    candidates: list[tuple[float, str]] = []
    for family, payload in comparison.get("by_family", {}).items():
        candidates.append((float(payload.get(metric_name, 0.0)), family))
    if not candidates:
        return None
    delta, family = max(candidates)
    return {"family": family, "delta": round(delta, 4)}


def build_validation_bundle(
    *,
    curation_comparison_path: Path,
    search_conditioned_comparison_path: Path | None,
    shallow_run_dir: Path,
    adaptive_run_dirs: list[Path],
    rank_shift_run_dirs: list[Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Assemble a compact validation bundle aimed at hostile-review objections."""
    curation_comparison = load_json(curation_comparison_path)
    search_conditioned_comparison = (
        load_json(search_conditioned_comparison_path) if search_conditioned_comparison_path else None
    )

    shallow_row = _load_aggregate(shallow_run_dir)
    adaptive_rows = [_load_aggregate(path) for path in adaptive_run_dirs]
    rank_shift_rows = [_load_aggregate(path) for path in rank_shift_run_dirs]

    static_ranking = _metric_ranking(rank_shift_rows, ("static_summary", "accuracy"))
    dynamic_ranking = _metric_ranking(rank_shift_rows, ("interactive_summary", "belief_trajectory_quality"))
    flexibility_ranking = _metric_ranking(
        rank_shift_rows,
        ("interactive_summary", "episode_cognitive_flexibility_score"),
    )

    curation_effect = {
        "manual_refinement": {
            "overall": curation_comparison.get("overall", {}),
            "largest_signal_gain_family": _largest_family_delta(
                curation_comparison,
                "avg_benchmark_signal_delta",
            ),
            "largest_trajectory_gain_family": _largest_family_delta(
                curation_comparison,
                "avg_trajectory_value_delta",
            ),
        },
        "search_conditioned_refinement": None,
    }
    if search_conditioned_comparison is not None:
        curation_effect["search_conditioned_refinement"] = {
            "overall": search_conditioned_comparison.get("overall", {}),
            "largest_signal_gain_family": _largest_family_delta(
                search_conditioned_comparison,
                "avg_benchmark_signal_delta",
            ),
            "largest_trajectory_gain_family": _largest_family_delta(
                search_conditioned_comparison,
                "avg_trajectory_value_delta",
            ),
        }

    shallow_vs_adaptive = {
        "shallow_run": {
            "run_name": shallow_row["run_name"],
            "accuracy": _round(shallow_row["aggregate"]["static_summary"]["accuracy"]),
            "belief_trajectory_quality": _round(
                shallow_row["aggregate"]["interactive_summary"].get("belief_trajectory_quality", 0.0)
            ),
            "episode_cognitive_flexibility_score": _round(
                shallow_row["aggregate"]["interactive_summary"].get("episode_cognitive_flexibility_score", 0.0)
            ),
        },
        "adaptive_runs": [
            {
                "run_name": row["run_name"],
                "accuracy": _round(row["aggregate"]["static_summary"]["accuracy"]),
                "belief_trajectory_quality": _round(
                    row["aggregate"]["interactive_summary"].get("belief_trajectory_quality", 0.0)
                ),
                "episode_cognitive_flexibility_score": _round(
                    row["aggregate"]["interactive_summary"].get("episode_cognitive_flexibility_score", 0.0)
                ),
            }
            for row in adaptive_rows
        ],
        "matching_composition": _matching_composition([shallow_row, *adaptive_rows]),
    }

    rank_shift = {
        "static_accuracy_ranking": static_ranking,
        "belief_trajectory_quality_ranking": dynamic_ranking,
        "episode_cognitive_flexibility_ranking": flexibility_ranking,
        "rank_shift_present": [row["run_name"] for row in static_ranking]
        != [row["run_name"] for row in dynamic_ranking],
        "matching_composition": _matching_composition(rank_shift_rows),
    }

    insights = [
        (
            f"Manual refinement increased kept tasks by {curation_effect['manual_refinement']['overall'].get('kept_delta', 0)} "
            "while improving benchmark signal in the weakest families."
        ),
        (
            "Static and interactive rankings diverged across the compared model runs."
            if rank_shift["rank_shift_present"]
            else "Static and interactive rankings did not diverge in this validation bundle."
        ),
        (
            "Shallow and adaptive runs used matching task composition."
            if shallow_vs_adaptive["matching_composition"]
            else "Shallow and adaptive runs differed in composition, so this sanity check is weaker."
        ),
    ]

    summary = {
        "curation_effect": curation_effect,
        "shallow_vs_adaptive": shallow_vs_adaptive,
        "rank_shift": rank_shift,
        "top_insights": insights,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "validation_summary.json", summary)

    shallow_headers = ["run_name", "accuracy", "belief_trajectory_quality", "episode_cognitive_flexibility_score"]
    shallow_rows = [
        [
            shallow_vs_adaptive["shallow_run"]["run_name"],
            shallow_vs_adaptive["shallow_run"]["accuracy"],
            shallow_vs_adaptive["shallow_run"]["belief_trajectory_quality"],
            shallow_vs_adaptive["shallow_run"]["episode_cognitive_flexibility_score"],
        ],
        *[
            [
                row["run_name"],
                row["accuracy"],
                row["belief_trajectory_quality"],
                row["episode_cognitive_flexibility_score"],
            ]
            for row in shallow_vs_adaptive["adaptive_runs"]
        ],
    ]
    rank_headers = ["metric", *[row["run_name"] for row in rank_shift_rows]]
    rank_rows = [
        ["static_accuracy", *[row["value"] for row in static_ranking]],
        ["belief_trajectory_quality", *[row["value"] for row in dynamic_ranking]],
        ["episode_cognitive_flexibility_score", *[row["value"] for row in flexibility_ranking]],
    ]

    lines = [
        "# AGUS Validation Bundle",
        "",
        "## Curation Effect",
        f"- manual_kept_delta: `{curation_effect['manual_refinement']['overall'].get('kept_delta', 0)}`",
        f"- manual_rejected_delta: `{curation_effect['manual_refinement']['overall'].get('rejected_delta', 0)}`",
    ]
    if curation_effect["manual_refinement"]["largest_signal_gain_family"] is not None:
        signal_gain = curation_effect["manual_refinement"]["largest_signal_gain_family"]
        lines.append(
            f"- largest_manual_signal_gain: `{signal_gain['family']}` ({signal_gain['delta']})"
        )
    if curation_effect["search_conditioned_refinement"] is not None:
        lines.append(
            f"- search_conditioned_kept_delta: "
            f"`{curation_effect['search_conditioned_refinement']['overall'].get('kept_delta', 0)}`"
        )
    lines.extend(
        [
            "",
            "## Shallow Vs Adaptive",
            f"- matching_composition: `{shallow_vs_adaptive['matching_composition']}`",
            _markdown_table(shallow_headers, shallow_rows),
            "",
            "## Rank Shift",
            f"- rank_shift_present: `{rank_shift['rank_shift_present']}`",
            f"- matching_composition: `{rank_shift['matching_composition']}`",
            _markdown_table(rank_headers, rank_rows),
            "",
            "## Top Insights",
            "\n".join(f"- {insight}" for insight in insights),
        ]
    )
    (output_dir / "validation_bundle.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
