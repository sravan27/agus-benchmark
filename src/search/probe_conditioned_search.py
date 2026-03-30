"""Probe-conditioned generator search for AGUS benchmark development."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.curation.adversarial_curation import curate_tasks
from src.generators.attention_distractors import AttentionDistractorConfig, generate_attention_distractor_tasks
from src.generators.shift_transfer import ShiftTransferConfig, generate_shift_transfer_tasks
from src.utils.io_utils import load_json, save_json

SEARCH_SCORE_WEIGHTS = {
    "average_benchmark_signal": 0.32,
    "kept_rate": 0.24,
    "shortcut_resistance_index": 0.16,
    "family_specific_signal": 0.14,
    "average_trajectory_value": 0.08,
    "survival_rate": 0.06,
}


@dataclass(frozen=True)
class SearchCandidate:
    """One deterministic generator configuration to evaluate."""

    family: str
    config_id: str
    label: str
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _attention_candidates(count: int, seed: int) -> list[SearchCandidate]:
    configs = [
        ("attn_00", "baseline_first", {"anti_template_strength": 1, "distractor_diversity_level": 1, "cue_delay_level": 1, "adversarial_query_mode": "first_candidate"}),
        ("attn_01", "confusable_mid", {"anti_template_strength": 1, "distractor_diversity_level": 2, "cue_delay_level": 1, "adversarial_query_mode": "confusable"}),
        ("attn_02", "delayed_confusable", {"anti_template_strength": 1, "distractor_diversity_level": 2, "cue_delay_level": 2, "adversarial_query_mode": "confusable"}),
        ("attn_03", "diverse_hard", {"anti_template_strength": 2, "distractor_diversity_level": 3, "cue_delay_level": 1, "adversarial_query_mode": "max_confusable"}),
        ("attn_04", "diverse_delayed_hard", {"anti_template_strength": 2, "distractor_diversity_level": 3, "cue_delay_level": 2, "adversarial_query_mode": "max_confusable"}),
        ("attn_05", "contrastive_delayed", {"anti_template_strength": 2, "distractor_diversity_level": 3, "cue_delay_level": 2, "adversarial_query_mode": "contrastive_confusable"}),
    ]
    return [
        SearchCandidate(
            family="attention_distractors",
            config_id=config_id,
            label=label,
            config={"count": count, "seed": seed, **payload},
        )
        for config_id, label, payload in configs
    ]


def _shift_candidates(count: int, seed: int) -> list[SearchCandidate]:
    configs = [
        ("shift_00", "direct_baseline", {"anti_template_strength": 0, "remap_composition_depth": 1, "bridge_representation_mode": "none", "anti_anchor_strength": 0, "latent_rule_mix": "baseline"}),
        ("shift_01", "alias_chain_mid", {"anti_template_strength": 1, "remap_composition_depth": 2, "bridge_representation_mode": "alias_chain", "anti_anchor_strength": 1, "latent_rule_mix": "anchor_resistant"}),
        ("shift_02", "alias_chain_hard", {"anti_template_strength": 2, "remap_composition_depth": 3, "bridge_representation_mode": "alias_chain", "anti_anchor_strength": 2, "latent_rule_mix": "anchor_resistant"}),
        ("shift_03", "attribute_bridge_mid", {"anti_template_strength": 1, "remap_composition_depth": 2, "bridge_representation_mode": "attribute_bridge", "anti_anchor_strength": 1, "latent_rule_mix": "anchor_resistant"}),
        ("shift_04", "attribute_bridge_hard", {"anti_template_strength": 2, "remap_composition_depth": 3, "bridge_representation_mode": "attribute_bridge", "anti_anchor_strength": 2, "latent_rule_mix": "transfer_hard"}),
        ("shift_05", "mixed_bridge_hard", {"anti_template_strength": 2, "remap_composition_depth": 3, "bridge_representation_mode": "mixed", "anti_anchor_strength": 2, "latent_rule_mix": "transfer_hard"}),
    ]
    return [
        SearchCandidate(
            family="shift_transfer",
            config_id=config_id,
            label=label,
            config={"count": count, "seed": seed, **payload},
        )
        for config_id, label, payload in configs
    ]


def build_search_space(
    families: tuple[str, ...] = ("attention_distractors", "shift_transfer"),
    *,
    count_per_config: int = 24,
    seed_overrides: dict[str, int] | None = None,
) -> dict[str, list[SearchCandidate]]:
    """Build compact deterministic search spaces for supported families."""
    seeds = {"attention_distractors": 51, "shift_transfer": 23}
    if seed_overrides:
        seeds.update(seed_overrides)

    space: dict[str, list[SearchCandidate]] = {}
    for family in families:
        if family == "attention_distractors":
            space[family] = _attention_candidates(count_per_config, seeds[family])
        elif family == "shift_transfer":
            space[family] = _shift_candidates(count_per_config, seeds[family])
        else:
            raise ValueError(f"Search is not yet implemented for family: {family}")
    return space


def _generate_for_candidate(candidate: SearchCandidate) -> list[dict[str, Any]]:
    if candidate.family == "attention_distractors":
        return generate_attention_distractor_tasks(AttentionDistractorConfig(**candidate.config))
    if candidate.family == "shift_transfer":
        return generate_shift_transfer_tasks(ShiftTransferConfig(**candidate.config))
    raise ValueError(f"Unsupported family: {candidate.family}")


def _collect_records(curation_results: dict[str, Any]) -> list[dict[str, Any]]:
    return curation_results["curated_tasks"] + curation_results["rejected_tasks"]


def _family_specific_metric_name(family: str) -> str:
    if family == "attention_distractors":
        return "distractor_discrimination_score"
    if family == "shift_transfer":
        return "transfer_depth_score"
    return "benchmark_signal_score"


def _aggregate_search_metrics(family: str, curation_results: dict[str, Any]) -> dict[str, Any]:
    records = _collect_records(curation_results)
    total = len(records)
    kept_count = sum(1 for row in records if row["curation"]["decision"] == "keep")
    rejected_count = sum(1 for row in records if row["curation"]["decision"] == "reject")
    review_count = sum(1 for row in records if row["curation"]["decision"] == "review")

    scores = [row["curation"]["scores"] for row in records]
    family_metric_name = _family_specific_metric_name(family)
    family_specific_values = [float(score[family_metric_name]) for score in scores if score.get(family_metric_name) is not None]
    benchmark_signal = [float(score["benchmark_signal_score"]) for score in scores]
    trajectory_values = [float(score["trajectory_value_score"]) for score in scores]
    shortcut_resistance = [1.0 - float(score["shortcut_vulnerability_score"]) for score in scores]
    baseline_solve_rates = [float(score["baseline_solve_rate"]) for score in scores]

    kept_rate = round(kept_count / total, 4) if total else 0.0
    reject_rate = round(rejected_count / total, 4) if total else 0.0
    review_rate = round(review_count / total, 4) if total else 0.0
    survival_rate = round(1.0 - reject_rate, 4)
    average_benchmark_signal = _mean(benchmark_signal)
    average_trajectory_value = _mean(trajectory_values)
    shortcut_resistance_index = _mean(shortcut_resistance)
    family_specific_signal = _mean(family_specific_values)
    average_baseline_solve_rate = _mean(baseline_solve_rates)

    curation_weighted_score = round(
        (SEARCH_SCORE_WEIGHTS["average_benchmark_signal"] * average_benchmark_signal)
        + (SEARCH_SCORE_WEIGHTS["kept_rate"] * kept_rate)
        + (SEARCH_SCORE_WEIGHTS["shortcut_resistance_index"] * shortcut_resistance_index)
        + (SEARCH_SCORE_WEIGHTS["family_specific_signal"] * family_specific_signal)
        + (SEARCH_SCORE_WEIGHTS["average_trajectory_value"] * average_trajectory_value)
        + (SEARCH_SCORE_WEIGHTS["survival_rate"] * survival_rate),
        4,
    )

    top_reasons = dict(curation_results["curation_report"]["most_common_rejection_reasons"])
    return {
        "kept_rate": kept_rate,
        "review_rate": review_rate,
        "reject_rate": reject_rate,
        "average_benchmark_signal": average_benchmark_signal,
        "average_trajectory_value": average_trajectory_value,
        "shortcut_resistance_index": shortcut_resistance_index,
        "family_specific_signal": family_specific_signal,
        "average_baseline_solve_rate": average_baseline_solve_rate,
        "curation_weighted_score": curation_weighted_score,
        "family_specific_metric": family_metric_name,
        "top_rejection_reasons": top_reasons,
    }


def _rank_key(result: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = result["search_metrics"]
    return (
        float(metrics["curation_weighted_score"]),
        float(metrics["average_benchmark_signal"]),
        float(metrics["kept_rate"]),
        -float(metrics["reject_rate"]),
    )


def rank_family_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank results from strongest to weakest using curation-weighted signal."""
    return sorted(results, key=_rank_key, reverse=True)


def _metric_deltas(
    winner_metrics: dict[str, Any],
    peer_metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tracked = [
        "curation_weighted_score",
        "average_benchmark_signal",
        "kept_rate",
        "shortcut_resistance_index",
        "family_specific_signal",
        "average_trajectory_value",
    ]
    median_like = {name: _mean([float(metrics[name]) for metrics in peer_metrics]) for name in tracked}
    deltas = [
        {"metric": name, "delta_vs_family_mean": round(float(winner_metrics[name]) - float(median_like[name]), 4)}
        for name in tracked
    ]
    return sorted(deltas, key=lambda item: item["delta_vs_family_mean"], reverse=True)


def _selection_summary(family: str, ranked_results: list[dict[str, Any]]) -> dict[str, Any]:
    winner = ranked_results[0]
    fallback = ranked_results[1] if len(ranked_results) > 1 else None
    top_three = ranked_results[:3]
    winner_deltas = _metric_deltas(winner["search_metrics"], [row["search_metrics"] for row in ranked_results])
    tradeoff_notes = []
    if winner["search_metrics"]["reject_rate"] > 0.25:
        tradeoff_notes.append("strong signal with non-trivial rejection rate")
    if winner["search_metrics"]["kept_rate"] < 0.5:
        tradeoff_notes.append("high-quality but lower-yield configuration")
    if not tradeoff_notes:
        tradeoff_notes.append("balanced signal and retention")

    return {
        "family": family,
        "winner": {
            "config_id": winner["config_id"],
            "label": winner["label"],
            "config": winner["config"],
            "search_metrics": winner["search_metrics"],
            "why_it_won": winner_deltas[:3],
            "tradeoffs": tradeoff_notes,
        },
        "fallback": (
            {
                "config_id": fallback["config_id"],
                "label": fallback["label"],
                "config": fallback["config"],
                "search_metrics": fallback["search_metrics"],
            }
            if fallback
            else None
        ),
        "top_3": [
            {
                "config_id": row["config_id"],
                "label": row["label"],
                "search_metrics": row["search_metrics"],
            }
            for row in top_three
        ],
    }


def evaluate_search_space(
    family: str,
    candidates: list[SearchCandidate],
) -> list[dict[str, Any]]:
    """Generate tasks for each candidate and evaluate them with adversarial curation."""
    evaluated: list[dict[str, Any]] = []
    for candidate in candidates:
        tasks = _generate_for_candidate(candidate)
        curation_results = curate_tasks(tasks)
        search_metrics = _aggregate_search_metrics(family, curation_results)
        evaluated.append(
            {
                "family": family,
                "config_id": candidate.config_id,
                "label": candidate.label,
                "config": candidate.config,
                "num_tasks": len(tasks),
                "search_metrics": search_metrics,
                "curation_report": curation_results["curation_report"],
            }
        )
    return rank_family_results(evaluated)


def run_probe_conditioned_search(
    families: tuple[str, ...] = ("attention_distractors", "shift_transfer"),
    *,
    count_per_config: int = 24,
    seed_overrides: dict[str, int] | None = None,
    emit_best_batches: bool = False,
    best_batch_size: int = 12,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run probe-conditioned search across supported families."""
    space = build_search_space(families, count_per_config=count_per_config, seed_overrides=seed_overrides)
    search_results: dict[str, Any] = {}
    best_configs: dict[str, Any] = {}
    search_summary: dict[str, Any] = {"weights": SEARCH_SCORE_WEIGHTS, "families": {}}
    best_batches: dict[str, list[dict[str, Any]]] = {}

    for family in families:
        ranked = evaluate_search_space(family, space[family])
        search_results[family] = ranked
        selection = _selection_summary(family, ranked)
        search_summary["families"][family] = selection
        best_configs[family] = {
            "winner": selection["winner"],
            "fallback": selection["fallback"],
        }
        if emit_best_batches:
            winner_config = dict(selection["winner"]["config"])
            winner_config["count"] = best_batch_size
            winner_candidate = SearchCandidate(
                family=family,
                config_id=f"{selection['winner']['config_id']}_best_batch",
                label=f"{selection['winner']['label']}_best_batch",
                config=winner_config,
            )
            best_batches[family] = _generate_for_candidate(winner_candidate)

    payload = {
        "search_results": search_results,
        "best_generator_configs": best_configs,
        "search_summary": search_summary,
        "best_batches": best_batches,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(output_dir / "search_results.json", search_results)
        save_json(output_dir / "best_generator_configs.json", best_configs)
        save_json(output_dir / "search_summary.json", search_summary)
        if emit_best_batches:
            batch_dir = output_dir / "best_batches"
            batch_dir.mkdir(parents=True, exist_ok=True)
            for family, rows in best_batches.items():
                save_json(batch_dir / f"{family}.json", rows)

    return payload


def load_search_artifacts(output_dir: Path) -> dict[str, Any]:
    """Load saved search artifacts from disk."""
    return {
        "search_results": load_json(output_dir / "search_results.json"),
        "best_generator_configs": load_json(output_dir / "best_generator_configs.json"),
        "search_summary": load_json(output_dir / "search_summary.json"),
    }
