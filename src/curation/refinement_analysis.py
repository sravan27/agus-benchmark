"""Analyze curation failures and compare refinement cycles."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.utils.io_utils import load_json

TARGET_FAMILIES = ("attention_distractors", "shift_transfer")


def _family_rows(rows: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["family"] == family]


def _top_reasons(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(reason for row in rows for reason in row["curation"]["reasons"]).most_common())


def _baseline_success_patterns(rows: list[dict[str, Any]], threshold: float = 0.65) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    examples: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        for baseline in row["curation"]["baseline_results"]:
            if not baseline["solved"] and baseline["session_score"] < threshold:
                continue
            solver_name = baseline["solver_name"]
            counts[solver_name] += 1
            if len(examples[solver_name]) < 3:
                examples[solver_name].append(
                    {
                        "task_id": row["task_id"],
                        "session_score": baseline["session_score"],
                        "reasons": row["curation"]["reasons"],
                    }
                )

    return {
        "solver_counts": dict(counts.most_common()),
        "solver_examples": dict(examples),
    }


def _top_failure_modes_for_attention(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reasons = _top_reasons(rows)
    patterns = _baseline_success_patterns(rows)
    failure_modes: list[dict[str, Any]] = []

    if reasons.get("too_templated", 0):
        failure_modes.append(
            {
                "failure_mode": "repeated_distractor_templates",
                "evidence": reasons["too_templated"],
                "recommendation": "Increase distractor profile diversity and vary cue timing.",
            }
        )
    if patterns["solver_counts"].get("pattern_match_solver", 0):
        failure_modes.append(
            {
                "failure_mode": "nearest_example_shortcuts",
                "evidence": patterns["solver_counts"]["pattern_match_solver"],
                "recommendation": "Select adversarial query rows that are similar to examples but require different outputs.",
            }
        )
    if patterns["solver_counts"].get("distractor_vulnerable_solver", 0):
        failure_modes.append(
            {
                "failure_mode": "salience_trap_capture",
                "evidence": patterns["solver_counts"]["distractor_vulnerable_solver"],
                "recommendation": "Use distractors that are salient but inconsistent under later evidence.",
            }
        )
    if reasons.get("weak_revision_signal", 0):
        failure_modes.append(
            {
                "failure_mode": "cue_arrives_too_early",
                "evidence": reasons["weak_revision_signal"],
                "recommendation": "Delay the disambiguating cue and force a temporary commitment under clutter.",
            }
        )
    return failure_modes


def _top_failure_modes_for_shift(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reasons = _top_reasons(rows)
    patterns = _baseline_success_patterns(rows, threshold=0.55)
    failure_modes: list[dict[str, Any]] = []

    if reasons.get("too_templated", 0):
        failure_modes.append(
            {
                "failure_mode": "clean_direct_remaps",
                "evidence": reasons["too_templated"],
                "recommendation": "Compose codebook shifts and vary remap profiles.",
            }
        )
    if patterns["solver_counts"].get("representation_anchor_solver", 0):
        failure_modes.append(
            {
                "failure_mode": "source_answer_anchoring",
                "evidence": patterns["solver_counts"]["representation_anchor_solver"],
                "recommendation": "Use multi-step remaps and queries that punish answer reuse across representations.",
            }
        )
    if patterns["solver_counts"].get("majority_rule_solver", 0):
        failure_modes.append(
            {
                "failure_mode": "local_rule_overgeneralization",
                "evidence": patterns["solver_counts"]["majority_rule_solver"],
                "recommendation": "Bias toward rules that cannot be recovered by identity or reversal anchors.",
            }
        )
    return failure_modes


def summarize_refinement_opportunities(
    report: dict[str, Any],
    rejected_tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a structured summary of generator weaknesses from curation artifacts."""
    families: dict[str, Any] = {}
    for family, stats in report["per_family_retention_rates"].items():
        family_rows = _family_rows(rejected_tasks, family)
        baseline_patterns = _baseline_success_patterns(family_rows)
        families[family] = {
            "retention": stats,
            "rejection_reasons": _top_reasons(family_rows),
            "baseline_success_patterns": baseline_patterns,
        }

    attention_rows = _family_rows(rejected_tasks, "attention_distractors")
    shift_rows = _family_rows(rejected_tasks, "shift_transfer")
    return {
        "total_tasks_processed": report["total_tasks_processed"],
        "family_summaries": families,
        "attention_distractors": {
            "top_failure_modes": _top_failure_modes_for_attention(attention_rows),
            "shallow_solver_success_patterns": _baseline_success_patterns(attention_rows),
        },
        "shift_transfer": {
            "top_failure_modes": _top_failure_modes_for_shift(shift_rows),
            "shallow_solver_success_patterns": _baseline_success_patterns(shift_rows, threshold=0.55),
        },
    }


def summarize_refinement_from_paths(report_path: Path, rejected_tasks_path: Path) -> dict[str, Any]:
    """Load curation artifacts and summarize refinement opportunities."""
    report = load_json(report_path)
    rejected_tasks = load_json(rejected_tasks_path)
    return summarize_refinement_opportunities(report, rejected_tasks)


def build_pre_post_curation_comparison(
    pre_report: dict[str, Any],
    post_report: dict[str, Any],
) -> dict[str, Any]:
    """Compare curation outcomes before and after a refinement cycle."""

    def family_metric_map(report: dict[str, Any], metric_name: str) -> dict[str, list[float]]:
        grouped: defaultdict[str, list[float]] = defaultdict(list)
        for row in report["task_index"]:
            if metric_name in row:
                grouped[row["family"]].append(float(row[metric_name]))
        return grouped

    pre_scores = family_metric_map(pre_report, "benchmark_signal_score")
    post_scores = family_metric_map(post_report, "benchmark_signal_score")
    pre_trajectory = family_metric_map(pre_report, "trajectory_value_score")
    post_trajectory = family_metric_map(post_report, "trajectory_value_score")
    family_names = sorted(set(pre_report["per_family_retention_rates"]) | set(post_report["per_family_retention_rates"]))
    by_family: dict[str, Any] = {}

    for family in family_names:
        pre_retention = pre_report["per_family_retention_rates"].get(family, {})
        post_retention = post_report["per_family_retention_rates"].get(family, {})
        pre_family_scores = pre_scores.get(family, [])
        post_family_scores = post_scores.get(family, [])
        pre_family_trajectory = pre_trajectory.get(family, [])
        post_family_trajectory = post_trajectory.get(family, [])
        pre_avg_signal = (sum(pre_family_scores) / len(pre_family_scores)) if pre_family_scores else 0.0
        post_avg_signal = (sum(post_family_scores) / len(post_family_scores)) if post_family_scores else 0.0
        pre_avg_trajectory = (sum(pre_family_trajectory) / len(pre_family_trajectory)) if pre_family_trajectory else 0.0
        post_avg_trajectory = (sum(post_family_trajectory) / len(post_family_trajectory)) if post_family_trajectory else 0.0
        by_family[family] = {
            "pre": pre_retention,
            "post": post_retention,
            "retention_rate_delta": round(post_retention.get("retention_rate", 0.0) - pre_retention.get("retention_rate", 0.0), 4),
            "avg_benchmark_signal_pre": round(pre_avg_signal, 4),
            "avg_benchmark_signal_post": round(post_avg_signal, 4),
            "avg_benchmark_signal_delta": round(post_avg_signal - pre_avg_signal, 4),
            "avg_trajectory_value_pre": round(pre_avg_trajectory, 4),
            "avg_trajectory_value_post": round(post_avg_trajectory, 4),
            "avg_trajectory_value_delta": round(post_avg_trajectory - pre_avg_trajectory, 4),
        }

    return {
        "overall": {
            "pre_kept": pre_report["kept_count"],
            "post_kept": post_report["kept_count"],
            "kept_delta": post_report["kept_count"] - pre_report["kept_count"],
            "pre_rejected": pre_report["rejected_count"],
            "post_rejected": post_report["rejected_count"],
            "rejected_delta": post_report["rejected_count"] - pre_report["rejected_count"],
        },
        "by_family": by_family,
    }


def load_best_generator_configs(
    best_configs_path: Path,
    *,
    target_families: tuple[str, ...] = TARGET_FAMILIES,
) -> dict[str, dict[str, Any]]:
    """Load winning generator configs from search artifacts."""
    payload = load_json(best_configs_path)
    promoted: dict[str, dict[str, Any]] = {}
    for family in target_families:
        if family not in payload:
            continue
        winner = payload[family]["winner"]
        promoted[family] = {
            "config_id": winner["config_id"],
            "label": winner["label"],
            "config": dict(winner["config"]),
            "search_metrics": dict(winner["search_metrics"]),
        }
    return promoted


def build_search_conditioned_refinement_summary(
    *,
    target_families: tuple[str, ...],
    promoted_configs: dict[str, dict[str, Any]],
    search_summary: dict[str, Any],
    pre_to_search_conditioned: dict[str, Any],
    manual_to_search_conditioned: dict[str, Any],
    manual_baseline_source: str,
    search_source: str,
) -> dict[str, Any]:
    """Summarize search-conditioned refinement outcomes and config promotion."""
    family_outcomes: dict[str, Any] = {}
    improved_families: list[str] = []

    for family in target_families:
        promoted = promoted_configs.get(family)
        search_family = search_summary.get("families", {}).get(family, {})
        pre_metrics = pre_to_search_conditioned["by_family"].get(family, {})
        manual_metrics = manual_to_search_conditioned["by_family"].get(family, {})
        improved_vs_manual = (
            manual_metrics.get("retention_rate_delta", 0.0) >= 0.0
            and manual_metrics.get("avg_benchmark_signal_delta", 0.0) >= 0.0
        )
        if improved_vs_manual:
            improved_families.append(family)

        family_outcomes[family] = {
            "promoted_config": promoted,
            "winner_rationale": search_family.get("winner", {}).get("why_it_won", []),
            "pre_to_search_conditioned": pre_metrics,
            "manual_to_search_conditioned": manual_metrics,
            "improved_vs_manual_refinement": improved_vs_manual,
        }

    return {
        "mode": "search_conditioned",
        "families_targeted": list(target_families),
        "promotion_sources": {
            "manual_refinement_baseline": manual_baseline_source,
            "search_artifacts": search_source,
        },
        "promoted_configs": promoted_configs,
        "pre_to_search_conditioned": pre_to_search_conditioned,
        "manual_to_search_conditioned": manual_to_search_conditioned,
        "family_outcomes": family_outcomes,
        "improved_families_vs_manual": improved_families,
        "search_conditioned_outperformed_manual": len(improved_families) == len(target_families),
    }
