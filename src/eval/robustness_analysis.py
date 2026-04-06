"""Multi-slice robustness analysis for AGUS core-result checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.eval.replication_analysis import (
    WEAKNESS_PROXY_NAMES,
    _bundle_metric,
    _core_run_summary,
    _execution_health,
    _leader,
    _load_run_bundle,
    _markdown_table,
    _pairwise_ranking,
    _round,
)
from src.utils.io_utils import save_json


def _slice_order_key(slice_name: str) -> tuple[int, str]:
    if slice_name == "original":
        return (0, slice_name)
    return (1, slice_name)


def _slice_bundles(run_dirs: list[Path]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for run_dir in run_dirs:
        bundle = _load_run_bundle(run_dir)
        grouped.setdefault(bundle["slice_name"], {})[bundle["model"]] = bundle
    return grouped


def _shared_models(slice_bundles: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    model_sets = [set(bundles) for bundles in slice_bundles.values() if bundles]
    if not model_sets:
        return []
    return sorted(set.intersection(*model_sets))


def _validate_bundles(slice_bundles: dict[str, dict[str, dict[str, Any]]], shared_models: list[str]) -> None:
    if "original" not in slice_bundles:
        raise ValueError("Robustness comparison requires an `original` slice.")
    if not shared_models:
        raise ValueError("No shared models were found across the supplied slices.")

    invalid_runs: list[dict[str, Any]] = []
    for slice_name, bundles in slice_bundles.items():
        for model in shared_models:
            if model not in bundles:
                invalid_runs.append(
                    {
                        "slice": slice_name,
                        "model": model,
                        "run_name": None,
                        "invalid_reasons": ["missing_model_for_slice"],
                    }
                )
                continue
            bundle = bundles[model]
            health = _execution_health(bundle)
            if not health["valid_for_replication"]:
                invalid_runs.append(
                    {
                        "slice": slice_name,
                        "model": model,
                        "run_name": bundle["run_name"],
                        "invalid_reasons": health["invalid_reasons"],
                    }
                )

    if invalid_runs:
        details = "; ".join(
            f"{row['slice']}:{row['model']} ({row['run_name']}): {', '.join(row['invalid_reasons'])}"
            for row in invalid_runs
        )
        raise ValueError(f"Robustness comparison refused invalid or incomplete runs: {details}")


def _slice_ranking_payload(bundles: dict[str, dict[str, Any]], metric_name: str, metric_path: tuple[str, ...]) -> dict[str, Any]:
    return _pairwise_ranking(
        bundles,
        metric_name=metric_name,
        metric_getter=lambda bundle: _bundle_metric(bundle, metric_path),
        higher_is_better=True,
    )


def _slice_summary(slice_name: str, bundles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "slice_name": slice_name,
        "runs": [_core_run_summary(bundle) for bundle in bundles.values()],
        "rankings": {
            "static_accuracy": _slice_ranking_payload(bundles, "static_accuracy", ("static_summary", "accuracy")),
            "belief_trajectory_quality": _slice_ranking_payload(
                bundles,
                "belief_trajectory_quality",
                ("interactive_summary", "belief_trajectory_quality"),
            ),
            "episode_cognitive_flexibility_score": _slice_ranking_payload(
                bundles,
                "episode_cognitive_flexibility_score",
                ("interactive_summary", "episode_cognitive_flexibility_score"),
            ),
        },
    }


def _slice_weakness_proxy_summary(bundles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    proxy_counts = {
        proxy_name: {
            model: _core_run_summary(bundle)["weakness_proxies"][proxy_name]
            for model, bundle in bundles.items()
        }
        for proxy_name in WEAKNESS_PROXY_NAMES
    }
    return {
        proxy_name: {
            "counts": counts,
            "higher": _leader(counts),
        }
        for proxy_name, counts in proxy_counts.items()
    }


def compare_robustness_runs(run_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    """Compare AGUS results across an original slice and multiple replication slices."""
    raw_slices = _slice_bundles(run_dirs)
    shared_models = _shared_models(raw_slices)
    _validate_bundles(raw_slices, shared_models)

    slice_names = sorted(raw_slices, key=_slice_order_key)
    slice_bundles = {
        slice_name: {model: raw_slices[slice_name][model] for model in shared_models}
        for slice_name in slice_names
    }

    original_summary = _slice_summary("original", slice_bundles["original"])
    original_weakness = _slice_weakness_proxy_summary(slice_bundles["original"])

    per_slice_checks: list[dict[str, Any]] = []
    weakness_replication: dict[str, dict[str, Any]] = {
        proxy_name: {
            "original_higher": original_weakness[proxy_name]["higher"],
            "held": 0,
            "total": 0,
            "per_slice": {},
        }
        for proxy_name in WEAKNESS_PROXY_NAMES
    }

    for slice_name in slice_names:
        if slice_name == "original":
            continue

        slice_summary = _slice_summary(slice_name, slice_bundles[slice_name])
        slice_weakness = _slice_weakness_proxy_summary(slice_bundles[slice_name])
        static_replicated = [
            row["model"] for row in original_summary["rankings"]["static_accuracy"]["ranking"]
        ] == [
            row["model"] for row in slice_summary["rankings"]["static_accuracy"]["ranking"]
        ]
        dynamic_replicated = [
            row["model"] for row in original_summary["rankings"]["belief_trajectory_quality"]["ranking"]
        ] == [
            row["model"] for row in slice_summary["rankings"]["belief_trajectory_quality"]["ranking"]
        ]
        original_divergence = (
            original_summary["rankings"]["static_accuracy"]["leader"]
            != original_summary["rankings"]["belief_trajectory_quality"]["leader"]
        )
        slice_divergence = (
            slice_summary["rankings"]["static_accuracy"]["leader"]
            != slice_summary["rankings"]["belief_trajectory_quality"]["leader"]
        )
        divergence_replicated = (
            original_divergence
            and slice_divergence
            and original_summary["rankings"]["static_accuracy"]["leader"]
            == slice_summary["rankings"]["static_accuracy"]["leader"]
            and original_summary["rankings"]["belief_trajectory_quality"]["leader"]
            == slice_summary["rankings"]["belief_trajectory_quality"]["leader"]
        )

        proxy_checks: dict[str, Any] = {}
        for proxy_name in WEAKNESS_PROXY_NAMES:
            direction_replicated = (
                original_weakness[proxy_name]["higher"] == slice_weakness[proxy_name]["higher"]
            )
            weakness_replication[proxy_name]["total"] += 1
            weakness_replication[proxy_name]["held"] += int(direction_replicated)
            weakness_replication[proxy_name]["per_slice"][slice_name] = {
                "higher": slice_weakness[proxy_name]["higher"],
                "direction_replicated": direction_replicated,
            }
            proxy_checks[proxy_name] = {
                "original_higher": original_weakness[proxy_name]["higher"],
                "slice_higher": slice_weakness[proxy_name]["higher"],
                "direction_replicated": direction_replicated,
            }

        per_slice_checks.append(
            {
                "slice_name": slice_name,
                "static_accuracy_ranking_replicated": static_replicated,
                "belief_trajectory_quality_ranking_replicated": dynamic_replicated,
                "static_vs_dynamic_divergence_replicated": divergence_replicated,
                "core_pattern_replicated": static_replicated and dynamic_replicated and divergence_replicated,
                "weakness_proxy_checks": proxy_checks,
            }
        )

    total_replications = len(per_slice_checks)
    held_counts = {
        "static_accuracy_ranking": {
            "held": sum(int(row["static_accuracy_ranking_replicated"]) for row in per_slice_checks),
            "total": total_replications,
        },
        "belief_trajectory_quality_ranking": {
            "held": sum(int(row["belief_trajectory_quality_ranking_replicated"]) for row in per_slice_checks),
            "total": total_replications,
        },
        "static_vs_dynamic_divergence": {
            "held": sum(int(row["static_vs_dynamic_divergence_replicated"]) for row in per_slice_checks),
            "total": total_replications,
        },
        "core_pattern": {
            "held": sum(int(row["core_pattern_replicated"]) for row in per_slice_checks),
            "total": total_replications,
        },
    }

    slice_summaries = {
        slice_name: _slice_summary(slice_name, bundles)
        for slice_name, bundles in slice_bundles.items()
    }
    slice_weakness_summaries = {
        slice_name: _slice_weakness_proxy_summary(bundles)
        for slice_name, bundles in slice_bundles.items()
    }

    summary = {
        "models_compared": shared_models,
        "slice_names": slice_names,
        "slice_summaries": slice_summaries,
        "slice_weakness_summaries": slice_weakness_summaries,
        "per_slice_checks": per_slice_checks,
        "replication_counts": held_counts,
        "weakness_proxy_replication": weakness_replication,
        "core_pattern_held_on": held_counts["core_pattern"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "robustness_summary.json", summary)

    overview_headers = [
        "slice",
        "run_name",
        "static_accuracy",
        "belief_trajectory_quality",
        "episode_cognitive_flexibility_score",
    ]
    overview_rows: list[list[Any]] = []
    for slice_name in slice_names:
        for row in slice_summaries[slice_name]["runs"]:
            overview_rows.append(
                [
                    slice_name,
                    row["run_name"],
                    row["static_accuracy"],
                    row["belief_trajectory_quality"],
                    row["episode_cognitive_flexibility_score"],
                ]
            )

    check_headers = [
        "slice",
        "static_rank_replicated",
        "btq_rank_replicated",
        "divergence_replicated",
        "core_pattern_replicated",
    ]
    check_rows = [
        [
            row["slice_name"],
            row["static_accuracy_ranking_replicated"],
            row["belief_trajectory_quality_ranking_replicated"],
            row["static_vs_dynamic_divergence_replicated"],
            row["core_pattern_replicated"],
        ]
        for row in per_slice_checks
    ]

    weakness_headers = ["proxy", "original_higher", "held", "total"]
    weakness_rows = [
        [
            proxy_name,
            payload["original_higher"],
            payload["held"],
            payload["total"],
        ]
        for proxy_name, payload in weakness_replication.items()
    ]

    top_insights = [
        (
            f"Core pattern held on {held_counts['core_pattern']['held']} of "
            f"{held_counts['core_pattern']['total']} replication slices."
        ),
        (
            f"Static accuracy ranking held on {held_counts['static_accuracy_ranking']['held']} of "
            f"{held_counts['static_accuracy_ranking']['total']} replication slices."
        ),
        (
            f"Belief trajectory quality ranking held on {held_counts['belief_trajectory_quality_ranking']['held']} of "
            f"{held_counts['belief_trajectory_quality_ranking']['total']} replication slices."
        ),
        (
            f"Static-vs-dynamic divergence held on {held_counts['static_vs_dynamic_divergence']['held']} of "
            f"{held_counts['static_vs_dynamic_divergence']['total']} replication slices."
        ),
    ]
    weakest_proxy = min(
        weakness_replication.items(),
        key=lambda item: (item[1]["held"] / item[1]["total"]) if item[1]["total"] else 1.0,
    )
    weakest_ratio = (
        round(weakest_proxy[1]["held"] / weakest_proxy[1]["total"], 4)
        if weakest_proxy[1]["total"]
        else 1.0
    )
    top_insights.append(
        f"The least stable weakness proxy was `{weakest_proxy[0]}` with directional replication {weakest_ratio}."
    )

    markdown = "\n\n".join(
        [
            "# AGUS Robustness Summary",
            f"- models_compared: `{', '.join(shared_models)}`",
            f"- slices: `{', '.join(slice_names)}`",
            (
                f"- core_pattern_held_on: `{held_counts['core_pattern']['held']}` / "
                f"`{held_counts['core_pattern']['total']}` replication slices"
            ),
            "## Slice Overview",
            _markdown_table(overview_headers, overview_rows) if overview_rows else "_No runs supplied._",
            "## Replication Checks",
            _markdown_table(check_headers, check_rows) if check_rows else "_No replication slices supplied._",
            "## Weakness Proxy Stability",
            _markdown_table(weakness_headers, weakness_rows) if weakness_rows else "_No weakness proxy data._",
            "## Top Insights",
            "\n".join(f"- {insight}" for insight in top_insights),
        ]
    )
    (output_dir / "robustness_summary.md").write_text(markdown + "\n", encoding="utf-8")
    return summary
