"""Instability analysis for AGUS interactive evaluation traces."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from src.utils.io_utils import load_json, load_jsonl, save_json

INSTABILITY_METRIC_DIRECTIONS = {
    "confidence_volatility": "higher_is_worse",
    "unnecessary_revision_rate": "higher_is_worse",
    "contradiction_blindness_rate": "higher_is_worse",
    "brittle_reversal_rate": "higher_is_worse",
    "attention_instability_score": "higher_is_worse",
    "belief_state_drift": "higher_is_worse",
    "trajectory_instability_index": "higher_is_worse",
}


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _event_types(session: dict[str, Any]) -> list[str]:
    events = []
    for turn in session.get("turns", []):
        event = turn["turn_spec"].get("event")
        if event and event.get("event_type"):
            events.append(event["event_type"])
    return events


def _contradiction_expected(session: dict[str, Any]) -> bool:
    if session.get("expected", {}).get("contradiction_expected"):
        return True
    trigger_words = ("conflict", "contradiction", "corrective", "cue", "deceptive")
    return any(any(word in event_type for word in trigger_words) for event_type in _event_types(session))


def _top_attended_signal(turn: dict[str, Any]) -> str | None:
    attended = turn["model_response"].get("attended_signals", [])
    return attended[0] if attended else None


def compute_session_instability(session: dict[str, Any]) -> dict[str, Any]:
    """Compute instability metrics for one interactive session."""
    response = session["response"]
    derived = session["derived"]
    confidences = [float(value) for value in response.get("turn_confidences", [])]
    deltas = [abs(next_value - value) for value, next_value in zip(confidences, confidences[1:])]
    confidence_volatility = _mean(deltas)

    unnecessary_revision = bool(
        derived.get("initial_correct")
        and not derived.get("revised_correct")
        and (derived.get("answer_changed") or derived.get("rule_changed"))
    )
    contradiction_blindness = bool(_contradiction_expected(session) and not response.get("contradiction_detected"))
    brittle_reversal = bool(derived.get("initial_correct") and not derived.get("revised_correct"))

    attention_instability = None
    if session["family"] == "attention_distractors":
        num_turns = max(int(derived.get("num_turns", 0)), 1)
        capture_turns = derived.get("capture_turns", [])
        cue_turn = int(derived.get("cue_turn", num_turns))
        recovery_turn = derived.get("recovery_turn")
        capture_rate = len(capture_turns) / num_turns
        if recovery_turn is None:
            recovery_penalty = 1.0
        else:
            recovery_penalty = _clamp((recovery_turn - cue_turn) / max(num_turns - cue_turn + 1, 1))
        post_cue_turns = session.get("turns", [])[max(cue_turn - 1, 0) :]
        post_signals = [_top_attended_signal(turn) for turn in post_cue_turns if _top_attended_signal(turn) is not None]
        post_switches = sum(signal_a != signal_b for signal_a, signal_b in zip(post_signals, post_signals[1:]))
        post_switch_rate = post_switches / max(len(post_signals) - 1, 1) if len(post_signals) > 1 else 0.0
        attention_instability = round(_mean([capture_rate, recovery_penalty, post_switch_rate]), 4)

    belief_state_drift = None
    if session["family"] == "social_miniworlds":
        trace = [value for value in derived.get("belief_consistency_trace", []) if value is not None]
        if trace:
            drift_steps = [abs(next_value - value) for value, next_value in zip(trace, trace[1:])]
            stability_loss = 1.0 - trace[-1]
            belief_state_drift = round(_mean(drift_steps + [stability_loss]), 4)

    components = [
        confidence_volatility,
        1.0 if unnecessary_revision else 0.0,
        1.0 if contradiction_blindness else 0.0,
        1.0 if brittle_reversal else 0.0,
    ]
    if attention_instability is not None:
        components.append(attention_instability)
    if belief_state_drift is not None:
        components.append(belief_state_drift)
    trajectory_instability_index = round(_mean(components), 4)

    return {
        "task_id": session["task_id"],
        "family": session["family"],
        "episode_type": session["episode_type"],
        "confidence_volatility": confidence_volatility,
        "unnecessary_revision": unnecessary_revision,
        "contradiction_blindness": contradiction_blindness,
        "brittle_reversal": brittle_reversal,
        "attention_instability_score": attention_instability,
        "belief_state_drift": belief_state_drift,
        "trajectory_instability_index": trajectory_instability_index,
        "interesting_reason": _instability_reason(
            session,
            confidence_volatility=confidence_volatility,
            unnecessary_revision=unnecessary_revision,
            contradiction_blindness=contradiction_blindness,
            brittle_reversal=brittle_reversal,
            attention_instability=attention_instability,
            belief_state_drift=belief_state_drift,
        ),
    }


def _instability_reason(
    session: dict[str, Any],
    *,
    confidence_volatility: float,
    unnecessary_revision: bool,
    contradiction_blindness: bool,
    brittle_reversal: bool,
    attention_instability: float | None,
    belief_state_drift: float | None,
) -> str:
    if contradiction_blindness:
        return "The model did not register contradictory evidence despite an interactive shift."
    if brittle_reversal:
        return "The model started correct and then collapsed after new evidence."
    if attention_instability is not None and attention_instability >= 0.6:
        return "Attention stayed unstable after clutter and cueing."
    if belief_state_drift is not None and belief_state_drift >= 0.4:
        return "Belief tracking drifted across the social episode."
    if unnecessary_revision:
        return "The model revised away from a previously correct path."
    if confidence_volatility >= 0.2:
        return "Confidence moved sharply across turns, suggesting brittle internal state."
    return "The trajectory showed moderate instability."


def analyze_run_instability(run_dir: Path) -> dict[str, Any]:
    """Analyze instability for one completed run and write artifacts."""
    sessions_path = run_dir / "interactive_sessions.jsonl"
    aggregate_summary = load_json(run_dir / "aggregate_summary.json")
    sessions = load_jsonl(sessions_path) if sessions_path.exists() else []

    session_rows = [compute_session_instability(session) for session in sessions]
    session_rows.sort(
        key=lambda row: (row["trajectory_instability_index"], row["task_id"]),
        reverse=True,
    )

    family_scores: dict[str, list[float]] = defaultdict(list)
    attention_scores: list[float] = []
    belief_drifts: list[float] = []
    for row in session_rows:
        family_scores[row["family"]].append(row["trajectory_instability_index"])
        if row["attention_instability_score"] is not None:
            attention_scores.append(float(row["attention_instability_score"]))
        if row["belief_state_drift"] is not None:
            belief_drifts.append(float(row["belief_state_drift"]))

    summary = {
        "run_name": run_dir.name,
        "adapter": aggregate_summary["adapter"],
        "num_sessions": len(session_rows),
        "metric_directions": INSTABILITY_METRIC_DIRECTIONS,
        "confidence_volatility": _mean([row["confidence_volatility"] for row in session_rows]),
        "unnecessary_revision_rate": _mean([1.0 if row["unnecessary_revision"] else 0.0 for row in session_rows]),
        "contradiction_blindness_rate": _mean([1.0 if row["contradiction_blindness"] else 0.0 for row in session_rows]),
        "brittle_reversal_rate": _mean([1.0 if row["brittle_reversal"] else 0.0 for row in session_rows]),
        "attention_instability_score": _mean(attention_scores),
        "belief_state_drift": _mean(belief_drifts),
        "trajectory_instability_index": _mean([row["trajectory_instability_index"] for row in session_rows]),
        "family_instability": {
            family: _mean(scores)
            for family, scores in sorted(family_scores.items())
        },
        "top_instability_cases": session_rows[:5],
    }

    save_json(run_dir / "instability_summary.json", summary)

    lines = [
        f"# Instability Highlights: {run_dir.name}",
        "",
        f"- trajectory_instability_index: {summary['trajectory_instability_index']}",
        f"- contradiction_blindness_rate: {summary['contradiction_blindness_rate']}",
        f"- brittle_reversal_rate: {summary['brittle_reversal_rate']}",
        f"- confidence_volatility: {summary['confidence_volatility']}",
        "",
        "## Most Unstable Families",
    ]
    for family, score in sorted(summary["family_instability"].items(), key=lambda item: item[1], reverse=True)[:3]:
        lines.append(f"- `{family}`: {score}")

    lines.extend(["", "## Top Instability Cases"])
    for row in summary["top_instability_cases"][:5]:
        lines.append(
            f"- `{row['task_id']}` | `{row['family']}` | instability={row['trajectory_instability_index']}"
        )
        lines.append(f"  Why interesting: {row['interesting_reason']}")

    (run_dir / "instability_highlights.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _blindness_qualifier(rate: float) -> str:
    if rate >= 0.5:
        return "still high"
    if rate >= 0.25:
        return "moderate"
    return "relatively low"


def compare_run_instability(run_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    """Compare instability metrics across multiple runs."""
    per_run = [analyze_run_instability(run_dir) for run_dir in run_dirs]
    family_names = sorted({family for item in per_run for family in item["family_instability"]})

    family_comparison = {
        family: {item["run_name"]: item["family_instability"].get(family, 0.0) for item in per_run}
        for family in family_names
    }

    most_brittle = max(per_run, key=lambda item: (item["trajectory_instability_index"], item["run_name"])) if per_run else None
    least_contradiction_blind = min(
        per_run,
        key=lambda item: (item["contradiction_blindness_rate"], item["run_name"]),
    ) if per_run else None

    summary = {
        "runs": per_run,
        "metric_directions": INSTABILITY_METRIC_DIRECTIONS,
        "family_instability_comparison": family_comparison,
        "most_brittle_model": most_brittle["run_name"] if most_brittle else None,
        "least_contradiction_blind_model": (
            least_contradiction_blind["run_name"] if least_contradiction_blind else None
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "instability_comparison.json", summary)

    lines = ["# Instability Insights", ""]
    if most_brittle:
        lines.append(
            f"- Most brittle overall: `{most_brittle['run_name']}` with trajectory_instability_index "
            f"{most_brittle['trajectory_instability_index']}."
        )
    if least_contradiction_blind:
        lines.append(
            f"- Least contradiction-blind: `{least_contradiction_blind['run_name']}` with "
            f"contradiction_blindness_rate {least_contradiction_blind['contradiction_blindness_rate']}"
            f" (lower is better, but this is {_blindness_qualifier(least_contradiction_blind['contradiction_blindness_rate'])})."
        )
    lines.append("")
    lines.append("## Families That Trigger The Most Instability")
    for family, scores in sorted(
        family_comparison.items(),
        key=lambda item: max(item[1].values()) if item[1] else 0.0,
        reverse=True,
    )[:5]:
        counts_text = ", ".join(f"`{run_name}`={score}" for run_name, score in scores.items())
        lines.append(f"- `{family}`: {counts_text}")

    (output_dir / "instability_insights.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
