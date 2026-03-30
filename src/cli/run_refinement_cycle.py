"""CLI for running AGUS refinement cycles."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.cli.generate_tasks import generate_all
from src.config import default_config
from src.curation.adversarial_curation import curate_tasks
from src.curation.refinement_analysis import (
    TARGET_FAMILIES,
    build_pre_post_curation_comparison,
    build_search_conditioned_refinement_summary,
    load_best_generator_configs,
    summarize_refinement_opportunities,
)
from src.generators.attention_distractors import AttentionDistractorConfig, generate_attention_distractor_tasks
from src.generators.shift_transfer import ShiftTransferConfig, generate_shift_transfer_tasks
from src.search.probe_conditioned_search import load_search_artifacts, run_probe_conditioned_search
from src.utils.io_utils import load_json, save_json
from src.utils.validation import validate_tasks

ORDERED_FAMILIES = (
    "hidden_rule",
    "shift_transfer",
    "metacog_revision",
    "attention_distractors",
    "social_miniworlds",
)


def _group_by_family(tasks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        grouped[task["family"]].append(task)
    return grouped


def _parse_families(raw: str | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(raw, tuple):
        return raw
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _load_or_generate_tasks(project_root: Path, tasks_path: Path, count_per_family: int) -> list[dict[str, Any]]:
    if tasks_path.exists():
        return load_json(tasks_path)

    datasets = generate_all(project_root, count_per_family=count_per_family)
    combined: list[dict[str, Any]] = []
    for rows in datasets.values():
        combined.extend(rows)
    save_json(tasks_path, combined)
    return combined


def _load_or_run_pre_curation(
    tasks: list[dict[str, Any]],
    report_path: Path,
    rejected_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if report_path.exists() and rejected_path.exists():
        return (load_json(report_path), load_json(rejected_path))

    results = curate_tasks(tasks)
    save_json(report_path, results["curation_report"])
    save_json(rejected_path, results["rejected_tasks"])
    return (results["curation_report"], results["rejected_tasks"])


def _derive_refined_attention_config(count: int, seed: int, summary: dict[str, Any]) -> AttentionDistractorConfig:
    attention_summary = summary["attention_distractors"]
    reasons = summary["family_summaries"].get("attention_distractors", {}).get("rejection_reasons", {})
    patterns = attention_summary["shallow_solver_success_patterns"]["solver_counts"]
    return AttentionDistractorConfig(
        count=count,
        seed=seed,
        anti_template_strength=2 if reasons.get("too_templated", 0) or patterns.get("pattern_match_solver", 0) else 1,
        distractor_diversity_level=3 if reasons.get("shortcut_solvable", 0) or patterns.get("distractor_vulnerable_solver", 0) else 2,
        cue_delay_level=2 if reasons.get("weak_revision_signal", 0) else 1,
        adversarial_query_mode="max_confusable" if patterns.get("pattern_match_solver", 0) else "confusable",
    )


def _derive_refined_shift_config(count: int, seed: int, summary: dict[str, Any]) -> ShiftTransferConfig:
    shift_summary = summary["shift_transfer"]
    reasons = summary["family_summaries"].get("shift_transfer", {}).get("rejection_reasons", {})
    patterns = shift_summary["shallow_solver_success_patterns"]["solver_counts"]
    return ShiftTransferConfig(
        count=count,
        seed=seed,
        anti_template_strength=2 if reasons.get("too_templated", 0) else 1,
        remap_composition_depth=3 if patterns.get("representation_anchor_solver", 0) or reasons.get("too_templated", 0) else 2,
        bridge_representation_mode="attribute_bridge" if reasons.get("too_templated", 0) else "alias_chain",
        anti_anchor_strength=2 if patterns.get("representation_anchor_solver", 0) else 1,
        latent_rule_mix="transfer_hard" if patterns.get("representation_anchor_solver", 0) else "anchor_resistant",
    )


def _build_manual_refined_configs(
    family_counts: Counter[str],
    seed_map: dict[str, int],
    refinement_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "attention_distractors": _derive_refined_attention_config(
            family_counts.get("attention_distractors", 0),
            seed_map["attention_distractors"],
            refinement_summary,
        ),
        "shift_transfer": _derive_refined_shift_config(
            family_counts.get("shift_transfer", 0),
            seed_map["shift_transfer"],
            refinement_summary,
        ),
    }


def _instantiate_promoted_configs(
    promoted_configs: dict[str, dict[str, Any]],
    family_counts: Counter[str],
    seed_map: dict[str, int],
    target_families: tuple[str, ...],
) -> dict[str, Any]:
    configured: dict[str, Any] = {}
    for family in target_families:
        if family not in promoted_configs:
            continue
        payload = dict(promoted_configs[family]["config"])
        payload["count"] = family_counts.get(family, int(payload.get("count", 0)))
        payload["seed"] = int(payload.get("seed", seed_map[family]))
        if family == "attention_distractors":
            configured[family] = AttentionDistractorConfig(**payload)
        elif family == "shift_transfer":
            configured[family] = ShiftTransferConfig(**payload)
    return configured


def _config_payload(configs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {family: config.__dict__ for family, config in configs.items()}


def _apply_refinement_configs(
    tasks: list[dict[str, Any]],
    configs: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped = _group_by_family(tasks)
    refined_grouped = dict(grouped)

    if "attention_distractors" in configs:
        refined_grouped["attention_distractors"] = generate_attention_distractor_tasks(configs["attention_distractors"])
    if "shift_transfer" in configs:
        refined_grouped["shift_transfer"] = generate_shift_transfer_tasks(configs["shift_transfer"])

    refined_tasks: list[dict[str, Any]] = []
    for family in ORDERED_FAMILIES:
        refined_tasks.extend(refined_grouped.get(family, []))
    validate_tasks(refined_tasks)
    return (refined_tasks, curate_tasks(refined_tasks))


def _artifact_prefix_for_mode(mode: str) -> str:
    return "" if mode == "plain" else "search_conditioned_"


def _save_refinement_artifacts(
    output_dir: Path,
    *,
    mode: str,
    refinement_summary: dict[str, Any],
    comparison: dict[str, Any],
    refined_tasks: list[dict[str, Any]],
    post_results: dict[str, Any],
) -> None:
    prefix = _artifact_prefix_for_mode(mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / f"{prefix}refinement_summary.json", refinement_summary)
    save_json(output_dir / f"{prefix}pre_post_curation_comparison.json", comparison)
    save_json(output_dir / f"{prefix}refined_tasks.json", refined_tasks)
    save_json(output_dir / f"{prefix}refined_curated_tasks.json", post_results["curated_tasks"])
    save_json(output_dir / f"{prefix}refined_rejected_tasks.json", post_results["rejected_tasks"])
    save_json(output_dir / f"{prefix}refined_curation_report.json", post_results["curation_report"])


def _existing_manual_refinement_report(output_dir: Path) -> dict[str, Any] | None:
    report_path = output_dir / "refined_curation_report.json"
    if report_path.exists():
        return load_json(report_path)
    return None


def _load_or_run_search_configs(
    *,
    search_dir: Path,
    target_families: tuple[str, ...],
    search_count_per_config: int,
    refresh_search: bool,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    best_configs_path = search_dir / "best_generator_configs.json"
    summary_path = search_dir / "search_summary.json"

    if not refresh_search and best_configs_path.exists() and summary_path.exists():
        search_payload = load_search_artifacts(search_dir)
        source = "loaded_existing"
    else:
        search_payload = run_probe_conditioned_search(
            families=target_families,
            count_per_config=search_count_per_config,
            output_dir=search_dir,
        )
        source = "ran_search"

    promoted = load_best_generator_configs(best_configs_path, target_families=target_families)
    return (promoted, search_payload["search_summary"], source)


def run_refinement_cycle(
    project_root: Path,
    tasks_path: Path,
    curation_report_path: Path,
    rejected_tasks_path: Path,
    output_dir: Path,
    count_per_family: int = 100,
    *,
    mode: str = "plain",
    target_families: tuple[str, ...] = TARGET_FAMILIES,
    search_dir: Path | None = None,
    search_count_per_config: int = 24,
    refresh_search: bool = False,
) -> dict[str, Any]:
    """Run one refinement cycle in plain or search-conditioned mode."""
    cfg = default_config(project_root)
    tasks = _load_or_generate_tasks(project_root, tasks_path, count_per_family=count_per_family)
    pre_report, pre_rejected = _load_or_run_pre_curation(tasks, curation_report_path, rejected_tasks_path)
    refinement_summary = summarize_refinement_opportunities(pre_report, pre_rejected)

    family_counts = Counter(task["family"] for task in tasks)
    seed_map = {spec.name: spec.seed for spec in cfg.family_specs}
    manual_configs = _build_manual_refined_configs(family_counts, seed_map, refinement_summary)

    if mode == "plain":
        refined_tasks, post_results = _apply_refinement_configs(tasks, manual_configs)
        comparison = build_pre_post_curation_comparison(pre_report, post_results["curation_report"])
        _save_refinement_artifacts(
            output_dir,
            mode=mode,
            refinement_summary=refinement_summary,
            comparison=comparison,
            refined_tasks=refined_tasks,
            post_results=post_results,
        )
        return {
            "mode": mode,
            "refinement_summary": refinement_summary,
            "comparison": comparison,
            "refined_configs": _config_payload(manual_configs),
        }

    if mode != "search_conditioned":
        raise ValueError(f"Unsupported refinement mode: {mode}")

    manual_baseline_source = "existing_artifact" if _existing_manual_refinement_report(output_dir) else "computed_inline"
    manual_baseline_report = _existing_manual_refinement_report(output_dir)
    if manual_baseline_report is None:
        _manual_tasks, manual_post_results = _apply_refinement_configs(tasks, manual_configs)
        manual_baseline_report = manual_post_results["curation_report"]

    active_search_dir = search_dir or (project_root / "data" / "generated" / "search")
    promoted_configs, search_summary, search_source = _load_or_run_search_configs(
        search_dir=active_search_dir,
        target_families=target_families,
        search_count_per_config=search_count_per_config,
        refresh_search=refresh_search,
    )
    promoted_config_objects = _instantiate_promoted_configs(promoted_configs, family_counts, seed_map, target_families)
    search_conditioned_tasks, search_conditioned_results = _apply_refinement_configs(tasks, promoted_config_objects)

    pre_to_search = build_pre_post_curation_comparison(pre_report, search_conditioned_results["curation_report"])
    manual_to_search = build_pre_post_curation_comparison(manual_baseline_report, search_conditioned_results["curation_report"])
    search_conditioned_summary = build_search_conditioned_refinement_summary(
        target_families=target_families,
        promoted_configs=promoted_configs,
        search_summary=search_summary,
        pre_to_search_conditioned=pre_to_search,
        manual_to_search_conditioned=manual_to_search,
        manual_baseline_source=manual_baseline_source,
        search_source=search_source,
    )

    _save_refinement_artifacts(
        output_dir,
        mode=mode,
        refinement_summary=search_conditioned_summary,
        comparison=pre_to_search,
        refined_tasks=search_conditioned_tasks,
        post_results=search_conditioned_results,
    )
    save_json(output_dir / "search_conditioned_refinement_summary.json", search_conditioned_summary)
    save_json(output_dir / "search_promoted_configs.json", promoted_configs)

    return {
        "mode": mode,
        "refinement_summary": search_conditioned_summary,
        "comparison": pre_to_search,
        "manual_to_search_conditioned": manual_to_search,
        "promoted_configs": promoted_configs,
        "search_summary": search_summary,
        "refined_configs": _config_payload(promoted_config_objects),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AGUS adversarial refinement cycle.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--tasks", type=Path, default=Path("data/generated/agus_v1_all.json"))
    parser.add_argument("--curation-report", type=Path, default=Path("data/generated/curation/curation_report.json"))
    parser.add_argument("--rejected-tasks", type=Path, default=Path("data/generated/curation/rejected_tasks.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated/refinement"))
    parser.add_argument("--count-per-family", type=int, default=100)
    parser.add_argument("--mode", choices=("plain", "search_conditioned"), default="plain")
    parser.add_argument("--target-families", type=str, default="attention_distractors,shift_transfer")
    parser.add_argument("--search-dir", type=Path, default=Path("data/generated/search"))
    parser.add_argument("--search-count-per-config", type=int, default=24)
    parser.add_argument("--refresh-search", action="store_true")
    args = parser.parse_args()

    target_families = _parse_families(args.target_families)
    results = run_refinement_cycle(
        project_root=args.project_root,
        tasks_path=args.tasks,
        curation_report_path=args.curation_report,
        rejected_tasks_path=args.rejected_tasks,
        output_dir=args.output_dir,
        count_per_family=args.count_per_family,
        mode=args.mode,
        target_families=target_families,
        search_dir=args.search_dir,
        search_count_per_config=args.search_count_per_config,
        refresh_search=args.refresh_search,
    )

    comparison = results["comparison"]
    print(f"mode: {results['mode']}")
    print(f"pre_kept: {comparison['overall']['pre_kept']}")
    print(f"post_kept: {comparison['overall']['post_kept']}")
    print(f"pre_rejected: {comparison['overall']['pre_rejected']}")
    print(f"post_rejected: {comparison['overall']['post_rejected']}")
    for family in target_families:
        family_stats = comparison["by_family"][family]
        print(
            f"{family}: retention {family_stats['pre'].get('retention_rate', 0.0)} -> "
            f"{family_stats['post'].get('retention_rate', 0.0)}"
        )
    if results["mode"] == "search_conditioned":
        print(f"search_promoted_families: {', '.join(results['promoted_configs'])}")
        print(
            "search_conditioned_outperformed_manual: "
            f"{results['refinement_summary']['search_conditioned_outperformed_manual']}"
        )
    print(f"saved refinement artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
