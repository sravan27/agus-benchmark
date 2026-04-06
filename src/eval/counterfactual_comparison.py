"""Cross-run comparison utilities for AGUS counterfactual evaluations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io_utils import load_json, save_json

COUNTERFACTUAL_METRICS = (
    "counterfactual_update_fidelity",
    "invariant_preservation_score",
    "branch_belief_coherence",
    "cross_branch_consistency",
    "counterfactual_confidence_calibration",
)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body_lines])


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    return value


def _load_summary(run_dir: Path) -> dict[str, Any]:
    payload = load_json(run_dir / "counterfactual_summary.json")
    adapter = payload.get("adapter", {})
    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "model": adapter.get("model") or adapter.get("name") or run_dir.name,
        "summary": payload,
    }


def compare_counterfactual_runs(run_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    """Compare multiple completed counterfactual evaluation runs."""
    rows = [_load_summary(run_dir) for run_dir in run_dirs]
    overall_metric_comparison = {
        metric: {row["run_name"]: _round(row["summary"]["overall_metrics"].get(metric, 0.0)) for row in rows}
        for metric in COUNTERFACTUAL_METRICS
    }
    families = sorted(
        {
            family
            for row in rows
            for family in row["summary"].get("family_metrics", {})
        }
    )
    family_metric_comparison = {
        family: {
            row["run_name"]: {
                metric: _round(row["summary"].get("family_metrics", {}).get(family, {}).get(metric, 0.0))
                for metric in COUNTERFACTUAL_METRICS
            }
            for row in rows
        }
        for family in families
    }

    rankings = {
        metric: sorted(
            [
                {"run_name": run_name, "value": value}
                for run_name, value in overall_metric_comparison[metric].items()
            ],
            key=lambda row: (-row["value"], row["run_name"]),
        )
        for metric in COUNTERFACTUAL_METRICS
    }

    top_insights = [
        (
            f"`{rankings['counterfactual_update_fidelity'][0]['run_name']}` led counterfactual update fidelity "
            f"at {rankings['counterfactual_update_fidelity'][0]['value']}."
        ),
        (
            f"`{rankings['branch_belief_coherence'][0]['run_name']}` led branch belief coherence "
            f"at {rankings['branch_belief_coherence'][0]['value']}."
        ),
        (
            f"`{rankings['counterfactual_confidence_calibration'][0]['run_name']}` had the strongest "
            f"counterfactual confidence calibration at {rankings['counterfactual_confidence_calibration'][0]['value']}."
        ),
    ]

    summary = {
        "runs": [
            {
                "run_name": row["run_name"],
                "model": row["model"],
                "num_bundles": row["summary"]["num_bundles"],
                "num_branches": row["summary"]["num_branches"],
                "overall_metrics": row["summary"]["overall_metrics"],
            }
            for row in rows
        ],
        "overall_metric_comparison": overall_metric_comparison,
        "family_metric_comparison": family_metric_comparison,
        "rankings": rankings,
        "top_insights": top_insights,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "counterfactual_comparison.json", summary)

    headers = ["run_name", "model", "num_bundles", "num_branches", *COUNTERFACTUAL_METRICS]
    overview_rows = [
        [
            row["run_name"],
            row["model"],
            row["summary"]["num_bundles"],
            row["summary"]["num_branches"],
            *[_round(row["summary"]["overall_metrics"].get(metric, 0.0)) for metric in COUNTERFACTUAL_METRICS],
        ]
        for row in rows
    ]
    lines = [
        "# AGUS Counterfactual Comparison",
        "",
        "## Run Overview",
        _markdown_table(headers, overview_rows),
        "",
        "## Top Insights",
        "\n".join(f"- {insight}" for insight in top_insights),
    ]
    (output_dir / "counterfactual_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
