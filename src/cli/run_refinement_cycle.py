"""CLI for running an adversarial refinement cycle over AGUS."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.cli.generate_tasks import generate_all
from src.config import default_config
from src.curation.adversarial_curation import curate_tasks
from src.curation.refinement_analysis import build_pre_post_curation_comparison, summarize_refinement_opportunities
from src.generators.attention_distractors import AttentionDistractorConfig, generate_attention_distractor_tasks
from src.generators.shift_transfer import ShiftTransferConfig, generate_shift_transfer_tasks
from src.utils.io_utils import load_json, save_json
from src.utils.validation import validate_tasks


def _group_by_family(tasks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        grouped[task["family"]].append(task)
    return grouped


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
    )


def run_refinement_cycle(
    project_root: Path,
    tasks_path: Path,
    curation_report_path: Path,
    rejected_tasks_path: Path,
    output_dir: Path,
    count_per_family: int = 100,
) -> dict[str, Any]:
    """Run one generator-refinement cycle using prior curation artifacts."""
    cfg = default_config(project_root)
    tasks = _load_or_generate_tasks(project_root, tasks_path, count_per_family=count_per_family)
    pre_report, pre_rejected = _load_or_run_pre_curation(tasks, curation_report_path, rejected_tasks_path)
    refinement_summary = summarize_refinement_opportunities(pre_report, pre_rejected)

    grouped = _group_by_family(tasks)
    family_counts = Counter(task["family"] for task in tasks)
    seed_map = {spec.name: spec.seed for spec in cfg.family_specs}

    refined_attention_cfg = _derive_refined_attention_config(
        family_counts.get("attention_distractors", count_per_family),
        seed_map["attention_distractors"],
        refinement_summary,
    )
    refined_shift_cfg = _derive_refined_shift_config(
        family_counts.get("shift_transfer", count_per_family),
        seed_map["shift_transfer"],
        refinement_summary,
    )

    refined_grouped = dict(grouped)
    refined_grouped["attention_distractors"] = generate_attention_distractor_tasks(refined_attention_cfg)
    refined_grouped["shift_transfer"] = generate_shift_transfer_tasks(refined_shift_cfg)

    refined_tasks: list[dict[str, Any]] = []
    for family in ("hidden_rule", "shift_transfer", "metacog_revision", "attention_distractors", "social_miniworlds"):
        refined_tasks.extend(refined_grouped.get(family, []))
    validate_tasks(refined_tasks)

    post_results = curate_tasks(refined_tasks)
    comparison = build_pre_post_curation_comparison(pre_report, post_results["curation_report"])

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "refinement_summary.json", refinement_summary)
    save_json(output_dir / "pre_post_curation_comparison.json", comparison)
    save_json(output_dir / "refined_tasks.json", refined_tasks)
    save_json(output_dir / "refined_curated_tasks.json", post_results["curated_tasks"])
    save_json(output_dir / "refined_rejected_tasks.json", post_results["rejected_tasks"])
    save_json(output_dir / "refined_curation_report.json", post_results["curation_report"])

    return {
        "refinement_summary": refinement_summary,
        "comparison": comparison,
        "refined_configs": {
            "attention_distractors": refined_attention_cfg.__dict__,
            "shift_transfer": refined_shift_cfg.__dict__,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AGUS adversarial refinement cycle.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--tasks", type=Path, default=Path("data/generated/agus_v1_all.json"))
    parser.add_argument("--curation-report", type=Path, default=Path("data/generated/curation/curation_report.json"))
    parser.add_argument("--rejected-tasks", type=Path, default=Path("data/generated/curation/rejected_tasks.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated/refinement"))
    parser.add_argument("--count-per-family", type=int, default=100)
    args = parser.parse_args()

    results = run_refinement_cycle(
        project_root=args.project_root,
        tasks_path=args.tasks,
        curation_report_path=args.curation_report,
        rejected_tasks_path=args.rejected_tasks,
        output_dir=args.output_dir,
        count_per_family=args.count_per_family,
    )

    comparison = results["comparison"]
    print(f"pre_kept: {comparison['overall']['pre_kept']}")
    print(f"post_kept: {comparison['overall']['post_kept']}")
    print(f"pre_rejected: {comparison['overall']['pre_rejected']}")
    print(f"post_rejected: {comparison['overall']['post_rejected']}")
    for family in ("attention_distractors", "shift_transfer"):
        family_stats = comparison["by_family"][family]
        print(
            f"{family}: retention {family_stats['pre'].get('retention_rate', 0.0)} -> "
            f"{family_stats['post'].get('retention_rate', 0.0)}"
        )
    print(f"saved refinement artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
