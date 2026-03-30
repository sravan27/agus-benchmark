"""Failure distillation and weakness extraction for AGUS evaluation runs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from src.utils.io_utils import load_json, save_json

WEAKNESS_CATEGORIES = (
    "static_dynamic_gap",
    "overconfident_error",
    "failed_hypothesis_update",
    "distractor_trap",
    "poor_attention_recovery",
    "failed_representation_transfer",
    "failed_metacognitive_revision",
    "social_belief_confusion",
    "trust_revision_failure",
    "deceptive_evidence_failure",
)

CATEGORY_EXPLANATIONS = {
    "static_dynamic_gap": "The model handled the frozen task but broke when the benchmark demanded live revision.",
    "overconfident_error": "The model was wrong while maintaining high confidence, which is especially useful for judge-facing evidence.",
    "failed_hypothesis_update": "The model saw new evidence but failed to revise its working hypothesis correctly.",
    "distractor_trap": "A salient but irrelevant cue appears to have hijacked the answer.",
    "poor_attention_recovery": "The model stayed stuck on a distractor even after a disambiguating cue arrived.",
    "failed_representation_transfer": "The model did not preserve the latent rule under surface remapping.",
    "failed_metacognitive_revision": "The model did not update its answer or rule belief cleanly after correction.",
    "social_belief_confusion": "The model appears to have confused world state with another agent's belief state.",
    "trust_revision_failure": "The model did not revise trust toward the reliable agent when evidence warranted it.",
    "deceptive_evidence_failure": "The model mishandled deceptive or conflicting social evidence.",
}

CATEGORY_BONUS = {
    "static_dynamic_gap": 0.35,
    "overconfident_error": 0.2,
    "failed_hypothesis_update": 0.3,
    "distractor_trap": 0.2,
    "poor_attention_recovery": 0.3,
    "failed_representation_transfer": 0.25,
    "failed_metacognitive_revision": 0.25,
    "social_belief_confusion": 0.25,
    "trust_revision_failure": 0.2,
    "deceptive_evidence_failure": 0.25,
}


def _load_run_bundle(run_dir: Path) -> dict[str, Any]:
    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "aggregate_summary": load_json(run_dir / "aggregate_summary.json"),
        "failure_cases": load_json(run_dir / "failure_cases.json"),
        "task_level_static_results": load_json(run_dir / "task_level_static_results.json"),
        "task_level_interactive_results": load_json(run_dir / "task_level_interactive_results.json"),
    }


def _index_by_task(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["task_id"]: row for row in rows}


def _extract_confidence(case: dict[str, Any]) -> float | None:
    if case["source"] == "interactive":
        response = case.get("response", {})
        if "revised_confidence" in response:
            return float(response["revised_confidence"])
        if response.get("turn_confidences"):
            return float(max(response["turn_confidences"]))
        return None

    prediction = case.get("prediction") or {}
    candidates = []
    for key in ("revised_confidence", "initial_confidence", "confidence"):
        if key in prediction:
            candidates.append(float(prediction[key]))
    return max(candidates) if candidates else None


def assign_failure_categories(
    case: dict[str, Any],
    *,
    static_result: dict[str, Any] | None = None,
    interactive_result: dict[str, Any] | None = None,
) -> list[str]:
    """Assign conservative weakness categories to a failure case."""
    categories: set[str] = set()
    reasons = set(case.get("failure_reasons", []))
    family = case["family"]
    confidence = _extract_confidence(case)

    if confidence is not None and confidence >= 0.75:
        categories.add("overconfident_error")

    if static_result is not None and interactive_result is not None:
        if bool(static_result.get("correct")) and not bool(interactive_result.get("correct")):
            categories.add("static_dynamic_gap")

    if case["source"] == "interactive":
        response = case.get("response", {})
        if (
            "missed_contradiction" in reasons
            or (response.get("evidence_acknowledged") and not response.get("revision_events"))
            or response.get("revised_answer") == response.get("initial_answer")
        ):
            categories.add("failed_hypothesis_update")

    if family == "attention_distractors":
        if "followed_distractor" in reasons:
            categories.add("distractor_trap")
        if "failed_attention_recovery" in reasons:
            categories.add("poor_attention_recovery")

    if family == "shift_transfer":
        if "transfer_failure" in reasons or (
            case["source"] == "interactive" and "final_answer_incorrect" in reasons
        ):
            categories.add("failed_representation_transfer")

    if family == "metacog_revision":
        if {"revision_failure", "no_answer_revision"} & reasons or (
            case["source"] == "interactive" and {"final_answer_incorrect", "missed_contradiction"} & reasons
        ):
            categories.add("failed_metacognitive_revision")

    if family == "social_miniworlds":
        if {"belief_state_miss", "belief_state_inconsistent"} & reasons:
            categories.add("social_belief_confusion")
        if {"trust_inference_miss", "trust_not_revised"} & reasons:
            categories.add("trust_revision_failure")
        if (
            {"missed_contradiction", "trust_not_revised"} & reasons
            or (
                case["source"] == "interactive"
                and not case.get("response", {}).get("contradiction_detected", True)
            )
        ):
            categories.add("deceptive_evidence_failure")

    return [category for category in WEAKNESS_CATEGORIES if category in categories]


def _short_reason(case: dict[str, Any], categories: list[str]) -> str:
    reasons = case.get("failure_reasons", [])
    if reasons:
        return ", ".join(reasons[:2])
    if categories:
        return categories[0].replace("_", " ")
    return "incorrect benchmark behavior"


def _why_interesting(categories: list[str], case: dict[str, Any]) -> str:
    if categories:
        return " ".join(CATEGORY_EXPLANATIONS[category] for category in categories[:2])
    if case["source"] == "interactive":
        return "This case failed during the dynamic evaluation path, which is central to AGUS."
    return "This case shows a benchmark miss without a strong category label."


def rank_failure_case(
    case: dict[str, Any],
    *,
    categories: list[str],
    static_result: dict[str, Any] | None = None,
    interactive_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute a distilled ranking payload for one failure."""
    confidence = _extract_confidence(case)
    severity = float(case.get("severity", 0.0))
    category_bonus = sum(CATEGORY_BONUS.get(category, 0.0) for category in categories)
    confidence_bonus = 0.2 if confidence is not None and confidence >= 0.85 else 0.1 if confidence is not None and confidence >= 0.7 else 0.0
    multi_turn_bonus = 0.0
    if case["source"] == "interactive":
        response = case.get("response", {})
        if len(response.get("turns", [])) >= 3:
            multi_turn_bonus = 0.15
    mismatch_bonus = 0.2 if "static_dynamic_gap" in categories else 0.0
    insight_score = round(severity + category_bonus + confidence_bonus + multi_turn_bonus + mismatch_bonus, 4)

    return {
        "task_id": case["task_id"],
        "family": case["family"],
        "source": case["source"],
        "episode_type": case.get("episode_type"),
        "severity": round(severity, 4),
        "confidence": round(confidence, 4) if confidence is not None else None,
        "categories": categories,
        "short_reason": _short_reason(case, categories),
        "why_interesting": _why_interesting(categories, case),
        "failure_reasons": list(case.get("failure_reasons", [])),
        "interestingness_score": insight_score,
        "static_correct": bool(static_result.get("correct")) if static_result else None,
        "interactive_correct": bool(interactive_result.get("correct")) if interactive_result else None,
    }


def _top_n(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    return rows[:n]


def distill_run_failures(run_dir: Path) -> dict[str, Any]:
    """Distill one completed evaluation run into judge-facing failure evidence."""
    bundle = _load_run_bundle(run_dir)
    static_by_task = _index_by_task(bundle["task_level_static_results"])
    interactive_by_task = _index_by_task(bundle["task_level_interactive_results"])

    distilled_rows = []
    for case in bundle["failure_cases"]:
        static_result = static_by_task.get(case["task_id"])
        interactive_result = interactive_by_task.get(case["task_id"])
        categories = assign_failure_categories(
            case,
            static_result=static_result,
            interactive_result=interactive_result,
        )
        distilled_rows.append(
            rank_failure_case(
                case,
                categories=categories,
                static_result=static_result,
                interactive_result=interactive_result,
            )
        )

    distilled_rows.sort(
        key=lambda row: (
            row["interestingness_score"],
            row["severity"],
            row["task_id"],
        ),
        reverse=True,
    )

    category_counts: dict[str, int] = defaultdict(int)
    family_top: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in distilled_rows:
        for category in row["categories"]:
            category_counts[category] += 1
        family_top[row["family"]].append(row)

    top_overall = _top_n(distilled_rows, 3)
    top_per_family = {family: _top_n(rows, 2) for family, rows in sorted(family_top.items())}
    top_static_dynamic_gap = _top_n(
        [row for row in distilled_rows if "static_dynamic_gap" in row["categories"]],
        3,
    )
    top_overconfident = _top_n(
        [row for row in distilled_rows if "overconfident_error" in row["categories"]],
        3,
    )
    top_multi_turn = _top_n(
        [
            row
            for row in distilled_rows
            if row["source"] == "interactive" and row["episode_type"] is not None
        ],
        3,
    )

    signature_weaknesses = sorted(
        (
            {
                "category": category,
                "count": count,
                "example_task_id": next(
                    (row["task_id"] for row in distilled_rows if category in row["categories"]),
                    None,
                ),
            }
            for category, count in category_counts.items()
        ),
        key=lambda row: (row["count"], row["category"]),
        reverse=True,
    )[:5]

    summary = {
        "run_name": bundle["run_name"],
        "run_dir": bundle["run_dir"],
        "num_failures": len(distilled_rows),
        "category_counts": {category: category_counts.get(category, 0) for category in WEAKNESS_CATEGORIES},
        "top_overall_failures": top_overall,
        "top_failures_per_family": top_per_family,
        "top_static_dynamic_gap_cases": top_static_dynamic_gap,
        "top_overconfident_errors": top_overconfident,
        "top_multi_turn_breakdowns": top_multi_turn,
        "signature_weaknesses": signature_weaknesses,
        "distilled_failures": distilled_rows,
    }

    save_json(run_dir / "distilled_failures.json", summary)

    lines = [
        f"# Signature Weaknesses: {bundle['run_name']}",
        "",
        "## Top Overall Failures",
    ]
    if top_overall:
        for row in top_overall:
            confidence = f", confidence={row['confidence']}" if row["confidence"] is not None else ""
            lines.append(
                f"- `{row['task_id']}` | `{row['family']}` | `{', '.join(row['categories']) or 'unlabeled'}`"
                f" | {row['short_reason']}{confidence}"
            )
            lines.append(f"  Why interesting: {row['why_interesting']}")
    else:
        lines.append("- No failures were found.")

    lines.extend(["", "## Top Weakness Types"])
    if signature_weaknesses:
        for weakness in signature_weaknesses[:3]:
            lines.append(
                f"- `{weakness['category']}` occurred {weakness['count']} times; example `{weakness['example_task_id']}`."
            )

    lines.extend(["", "## Per-Family Highlights"])
    for family, rows in top_per_family.items():
        if not rows:
            continue
        for row in rows[:2]:
            confidence = f", confidence={row['confidence']}" if row["confidence"] is not None else ""
            lines.append(
                f"- `{family}`: `{row['task_id']}` | `{', '.join(row['categories']) or 'unlabeled'}`"
                f" | {row['short_reason']}{confidence}"
            )
            lines.append(f"  Why interesting: {row['why_interesting']}")

    (run_dir / "signature_weaknesses.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def compare_distilled_failures(run_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    """Compare distilled weakness patterns across multiple runs."""
    per_run = [distill_run_failures(run_dir) for run_dir in run_dirs]
    per_run_counts = {
        item["run_name"]: item["category_counts"]
        for item in per_run
    }

    separating = []
    for category in WEAKNESS_CATEGORIES:
        counts = {run_name: category_counts.get(category, 0) for run_name, category_counts in per_run_counts.items()}
        delta = max(counts.values()) - min(counts.values()) if counts else 0
        separating.append({"category": category, "counts": counts, "delta": delta})
    separating.sort(key=lambda row: (row["delta"], row["category"]), reverse=True)

    summary = {
        "runs": [
            {
                "run_name": item["run_name"],
                "num_failures": item["num_failures"],
                "top_signature_weaknesses": item["signature_weaknesses"][:3],
                "category_counts": item["category_counts"],
            }
            for item in per_run
        ],
        "most_separating_weaknesses": separating[:5],
        "top_overall_by_run": {
            item["run_name"]: item["top_overall_failures"]
            for item in per_run
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "weakness_comparison.json", summary)

    lines = ["# Weakness Highlights", ""]
    for item in per_run:
        lines.append(f"## {item['run_name']}")
        for weakness in item["signature_weaknesses"][:3]:
            lines.append(
                f"- `{weakness['category']}`: {weakness['count']} cases, example `{weakness['example_task_id']}`."
            )
        lines.append("")
    lines.append("## Most Separating Weaknesses")
    for row in separating[:5]:
        counts_text = ", ".join(f"`{run_name}`={count}" for run_name, count in row["counts"].items())
        lines.append(f"- `{row['category']}` separated runs most strongly ({counts_text}).")

    (output_dir / "weakness_highlights.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
