"""Adversarial curation for AGUS tasks and episodes."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Callable

from src.eval.interactive_runner import EpisodeSpec, EpisodeTurnSpec, build_interaction_spec, run_interactive_session
from src.scoring.metrics import (
    attention_recovery_score,
    belief_state_consistency,
    contradiction_sensitivity,
    cue_utilization_score,
    deception_sensitivity,
    distractor_capture_rate,
    episode_cognitive_flexibility_score,
    exact_match,
    hypothesis_update_score,
    multi_turn_adaptation_score,
    online_adaptation_gain,
    trust_revision_score,
)

BaselineResponder = Callable[[EpisodeSpec, EpisodeTurnSpec, list[dict[str, Any]]], dict[str, Any]]


@dataclass(frozen=True)
class CurationPolicy:
    """Deterministic threshold policy for keep/review/reject decisions."""

    keep_min_signal: float = 0.66
    keep_max_baseline_solve_rate: float = 0.34
    review_min_signal: float = 0.48
    reject_baseline_solve_rate: float = 0.6
    reject_shortcut_vulnerability: float = 0.58
    reject_template_novelty: float = 0.22
    reject_revision_discrimination: float = 0.35
    reject_trajectory_value: float = 0.42
    reject_family_specific: float = 0.4


@dataclass(frozen=True)
class BaselineResult:
    """Summary of one baseline probe on one task."""

    solver_name: str
    session_score: float
    solved: bool
    revised_correct: bool
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result."""
        return asdict(self)


def _extract_location_from_text(content: str) -> str | None:
    """Recover a location phrase from templated social statements."""
    cleaned = content.strip().strip('"').strip("'").rstrip(".")
    markers = (" in the ", " to the ")
    for marker in markers:
        if marker in cleaned:
            return cleaned.rsplit(marker, 1)[-1].strip()
    return None


def _normalize_seq_like(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _sequence_overlap(a: Any, b: Any) -> float:
    seq_a = _normalize_seq_like(a)
    seq_b = _normalize_seq_like(b)
    if not seq_a or not seq_b:
        return 0.0
    matches = sum(1 for left, right in zip(seq_a, seq_b) if left == right)
    return matches / max(len(seq_a), len(seq_b))


def _last_visible_examples(turn: EpisodeTurnSpec) -> list[dict[str, Any]]:
    examples = turn.prompt.get("examples", [])
    if turn.event is not None:
        for key in ("example", "controlled_example", "truthful_statement", "deceptive_statement", "rumor_statement"):
            item = turn.event.get(key)
            if isinstance(item, dict) and item:
                examples = examples + [item]
    return examples


def _last_output_example(turn: EpisodeTurnSpec) -> dict[str, Any] | None:
    for example in reversed(_last_visible_examples(turn)):
        if "output" in example:
            return example
    return None


def _infer_numeric_rule_from_examples(examples: list[dict[str, Any]], field: str = "input") -> tuple[str | None, int | None]:
    """
    Infer a very small subset of easy rules.

    This intentionally models shallow induction rather than full rule recovery.
    """
    valid = [example for example in examples if field in example and "output" in example]
    if not valid:
        return (None, None)

    candidate_checks: list[tuple[str, Callable[[list[int], int], list[int]], range]] = [
        ("add_const", lambda seq, k: [value + k for value in seq], range(-3, 4)),
        ("rotate_left", lambda seq, k: seq[k:] + seq[:k], range(0, 4)),
        ("reverse", lambda seq, _k: list(reversed(seq)), range(1)),
    ]

    for name, fn, params in candidate_checks:
        for param in params:
            ok = True
            for example in valid:
                source = example[field]
                target = example["output"]
                if not isinstance(source, list) or not isinstance(target, list):
                    ok = False
                    break
                predicted = fn(list(source), param)
                if name == "add_const":
                    if predicted != target:
                        ok = False
                        break
                else:
                    if predicted != target:
                        ok = False
                        break
            if ok:
                return (name, int(param))
    return (None, None)


def _apply_inferred_numeric_rule(rule_name: str | None, param: int | None, seq: Any) -> Any:
    if rule_name is None or not isinstance(seq, list):
        return seq
    if rule_name == "add_const":
        return [value + int(param) for value in seq]
    if rule_name == "rotate_left":
        k = int(param) % len(seq)
        return seq[k:] + seq[:k]
    if rule_name == "reverse":
        return list(reversed(seq))
    return seq


def _pattern_match_payload(spec: EpisodeSpec, turn: EpisodeTurnSpec, _prior: list[dict[str, Any]]) -> dict[str, Any]:
    probe = turn.prompt.get("probe_input") or turn.prompt.get("query_record") or turn.prompt.get("question")
    visible = _last_visible_examples(turn)
    best = None
    best_score = -1.0
    for example in visible:
        if "input" in example and "output" in example:
            score = _sequence_overlap(example["input"], probe)
            if score > best_score:
                best = example
                best_score = score
        elif "signal_sequence" in example and "output" in example and isinstance(probe, dict):
            score = _sequence_overlap(example["signal_sequence"], probe.get("signal_sequence"))
            if score > best_score:
                best = example
                best_score = score

    if spec.family == "social_miniworlds":
        event = turn.event or {}
        chosen_location = spec.expected.get("initial_location")
        if event.get("deceptive_statement"):
            chosen_location = _extract_location_from_text(event["deceptive_statement"]["content"]) or chosen_location
        elif event.get("truthful_statement"):
            chosen_location = _extract_location_from_text(event["truthful_statement"]["content"]) or chosen_location
        elif event.get("rumor_statement"):
            chosen_location = _extract_location_from_text(event["rumor_statement"]["content"]) or chosen_location
        answer = {"actual_location": chosen_location}
        if "belief_of_false_belief_agent" in (turn.accepted_answers[0] if turn.accepted_answers else {}):
            answer["belief_of_false_belief_agent"] = chosen_location
        if "trusted_agent" in (turn.accepted_answers[0] if turn.accepted_answers else {}):
            answer["trusted_agent"] = spec.expected["deceptive_agent"]
        if "likely_action" in (turn.accepted_answers[0] if turn.accepted_answers else {}):
            answer["likely_action"] = "unclear"
        return {
            "answer": answer,
            "confidence": 0.72,
            "rule_explanation": "surface_pattern_match",
            "evidence_acknowledged": turn.event is not None,
            "contradiction_detected": False,
            "trust_scores_by_agent": {agent: 0.5 for agent in spec.initial_context["agents"]},
            "metadata": {"rule_tag": "surface_pattern_match"},
        }

    if best and "output" in best:
        predicted = best["output"]
    elif _prior:
        predicted = _prior[-1].get("answer", probe)
    else:
        predicted = probe
    attended = ["signal_sequence"] if spec.family == "attention_distractors" and isinstance(best, dict) and "signal_sequence" in best else []
    return {
        "answer": predicted,
        "confidence": 0.68,
        "rule_explanation": "surface_pattern_match",
        "evidence_acknowledged": turn.event is not None,
        "contradiction_detected": False,
        "attended_signals": attended,
        "ignored_signals": ["distractor_sequence"] if attended else [],
        "metadata": {"rule_tag": "surface_pattern_match"},
    }


def _majority_rule_payload(spec: EpisodeSpec, turn: EpisodeTurnSpec, prior: list[dict[str, Any]]) -> dict[str, Any]:
    visible = _last_visible_examples(turn)

    if spec.family in {"hidden_rule", "metacog_revision"}:
        rule_name, param = _infer_numeric_rule_from_examples(visible, field="input")
        probe = turn.prompt["probe_input"]
        predicted = _apply_inferred_numeric_rule(rule_name, param, probe)
        if rule_name is None:
            predicted = prior[-1]["answer"] if prior else probe
        return {
            "answer": predicted,
            "confidence": 0.7 if rule_name else 0.45,
            "rule_explanation": rule_name or "local_majority_pattern",
            "evidence_acknowledged": turn.event is not None,
            "contradiction_detected": False,
            "metadata": {"rule_tag": rule_name or "local_majority_pattern"},
        }

    if spec.family == "shift_transfer":
        probe = turn.prompt["probe_input"]
        first_example = visible[0] if visible else None
        if first_example and "input" in first_example and "output" in first_example:
            input_tokens = first_example["input"]
            output_tokens = first_example["output"]
            if output_tokens == list(reversed(input_tokens)):
                predicted = list(reversed(probe))
                rule_name = "reverse"
            else:
                predicted = probe
                rule_name = "identity_anchor"
        else:
            predicted = probe
            rule_name = "identity_anchor"
        return {
            "answer": predicted,
            "confidence": 0.62,
            "rule_explanation": rule_name,
            "evidence_acknowledged": turn.event is not None,
            "contradiction_detected": False,
            "metadata": {"rule_tag": rule_name},
        }

    if spec.family == "attention_distractors":
        query = turn.prompt["query_record"]
        notes = query.get("irrelevant_notes", [])
        use_distractor = len(notes) >= 2
        examples_with_signal = [
            {"input": example["signal_sequence"], "output": example["output"]}
            for example in visible
            if "signal_sequence" in example and "output" in example
        ]
        examples_with_distractor = [
            {"input": example["distractor_sequence"], "output": example["output"]}
            for example in visible
            if "distractor_sequence" in example and "output" in example
        ]
        chosen_field = "distractor_sequence" if use_distractor else "signal_sequence"
        rule_name, param = _infer_numeric_rule_from_examples(
            examples_with_distractor if use_distractor else examples_with_signal,
            field="input",
        )
        probe = query[chosen_field]
        predicted = _apply_inferred_numeric_rule(rule_name, param, probe)
        return {
            "answer": predicted,
            "confidence": 0.73 if use_distractor else 0.58,
            "rule_explanation": rule_name or "majority_local_pattern",
            "evidence_acknowledged": turn.event is not None,
            "contradiction_detected": use_distractor and turn.event is not None,
            "attended_signals": [chosen_field],
            "ignored_signals": ["signal_sequence" if chosen_field == "distractor_sequence" else "distractor_sequence"],
            "metadata": {"rule_tag": rule_name or "majority_local_pattern"},
        }

    if spec.family == "social_miniworlds":
        accepted = turn.accepted_answers[0] if turn.accepted_answers else {}
        answer = {"actual_location": spec.expected["initial_location"]}
        if "belief_of_false_belief_agent" in accepted:
            answer["belief_of_false_belief_agent"] = spec.expected["initial_location"]
        if "trusted_agent" in accepted:
            answer["trusted_agent"] = spec.expected["deceptive_agent"]
        if "likely_action" in accepted:
            answer["likely_action"] = "mislead_search"
        return {
            "answer": answer,
            "confidence": 0.6,
            "rule_explanation": "majority_statement",
            "evidence_acknowledged": turn.event is not None,
            "contradiction_detected": False,
            "trust_scores_by_agent": {
                agent: 0.55 if agent == spec.expected["deceptive_agent"] else 0.45
                for agent in spec.initial_context["agents"]
            },
            "metadata": {"rule_tag": "majority_statement"},
        }

    return _pattern_match_payload(spec, turn, prior)


def _distractor_payload(spec: EpisodeSpec, turn: EpisodeTurnSpec, prior: list[dict[str, Any]]) -> dict[str, Any]:
    if spec.family != "attention_distractors":
        return _pattern_match_payload(spec, turn, prior)
    query = turn.prompt["query_record"]
    if turn.event and "decoy_hypothesis_output" in turn.event:
        predicted = turn.event["decoy_hypothesis_output"]
    else:
        predicted = query["distractor_sequence"]
    return {
        "answer": predicted,
        "confidence": 0.82,
        "rule_explanation": "salience_overweighting",
        "evidence_acknowledged": turn.event is not None,
        "contradiction_detected": False,
        "attended_signals": ["distractor_sequence"],
        "ignored_signals": ["signal_sequence"],
        "revision_events": ["captured_by_distractor"],
        "metadata": {"rule_tag": "salience_overweighting"},
    }


def _trust_naive_payload(spec: EpisodeSpec, turn: EpisodeTurnSpec, prior: list[dict[str, Any]]) -> dict[str, Any]:
    if spec.family != "social_miniworlds":
        return _pattern_match_payload(spec, turn, prior)
    accepted = turn.accepted_answers[0] if turn.accepted_answers else {}
    answer = {"actual_location": spec.expected["initial_location"]}
    if "belief_of_false_belief_agent" in accepted:
        answer["belief_of_false_belief_agent"] = spec.expected["initial_location"]
    if "trusted_agent" in accepted:
        answer["trusted_agent"] = sorted(spec.initial_context["agents"])[0]
    if "likely_action" in accepted:
        answer["likely_action"] = "unclear"
    uniform = {agent: round(1.0 / len(spec.initial_context["agents"]), 3) for agent in spec.initial_context["agents"]}
    return {
        "answer": answer,
        "confidence": 0.58,
        "rule_explanation": "uniform_trust",
        "evidence_acknowledged": turn.event is not None,
        "contradiction_detected": False,
        "trust_scores_by_agent": uniform,
        "inferred_agent_beliefs": {
            agent: spec.expected["initial_location"] for agent in spec.initial_context["agents"]
        },
        "metadata": {"rule_tag": "uniform_trust"},
    }


def _representation_anchor_payload(spec: EpisodeSpec, turn: EpisodeTurnSpec, prior: list[dict[str, Any]]) -> dict[str, Any]:
    if spec.family != "shift_transfer":
        return _pattern_match_payload(spec, turn, prior)
    if not prior:
        return _majority_rule_payload(spec, turn, prior)
    return {
        "answer": prior[0]["answer"],
        "confidence": 0.76,
        "rule_explanation": "source_representation_anchor",
        "evidence_acknowledged": turn.event is not None,
        "contradiction_detected": False,
        "revision_events": [],
        "metadata": {"rule_tag": "source_representation_anchor"},
    }


def _make_no_revision_responder(initial_strategy: BaselineResponder) -> BaselineResponder:
    def respond(spec: EpisodeSpec, turn: EpisodeTurnSpec, prior: list[dict[str, Any]]) -> dict[str, Any]:
        if not prior:
            payload = initial_strategy(spec, turn, prior)
            payload["metadata"] = dict(payload.get("metadata", {}))
            payload["metadata"]["rule_tag"] = payload["metadata"].get("rule_tag", "no_revision")
            return payload
        first = prior[0]
        return {
            "answer": first["answer"],
            "confidence": float(first["confidence"]),
            "rule_explanation": first["rule_explanation"],
            "evidence_acknowledged": False,
            "contradiction_detected": False,
            "attended_signals": list(first.get("attended_signals", [])),
            "ignored_signals": list(first.get("ignored_signals", [])),
            "trust_scores_by_agent": dict(first.get("trust_scores_by_agent", {})),
            "inferred_agent_beliefs": dict(first.get("inferred_agent_beliefs", {})),
            "revision_events": [],
            "metadata": dict(first.get("metadata", {})),
        }

    return respond


def _baseline_registry() -> dict[str, BaselineResponder]:
    return {
        "pattern_match_solver": _pattern_match_payload,
        "majority_rule_solver": _majority_rule_payload,
        "distractor_vulnerable_solver": _distractor_payload,
        "no_revision_solver": _make_no_revision_responder(_majority_rule_payload),
        "trust_naive_solver": _trust_naive_payload,
        "representation_anchor_solver": _representation_anchor_payload,
    }


def run_baseline_suite(task: dict[str, Any], baselines: dict[str, BaselineResponder] | None = None) -> list[BaselineResult]:
    """Evaluate one task against the lightweight baseline probe suite."""
    active_baselines = baselines or _baseline_registry()
    baseline_results: list[BaselineResult] = []
    for name, responder in active_baselines.items():
        session = run_interactive_session(task, responder)
        session_score, metrics = _baseline_session_score(task, session)
        solved = session_score >= _solve_threshold_for_family(task["family"])
        baseline_results.append(
            BaselineResult(
                solver_name=name,
                session_score=session_score,
                solved=solved,
                revised_correct=bool(session["derived"]["revised_correct"]),
                metrics=metrics,
            )
        )
    return baseline_results


def _session_metric_summary(session: dict[str, Any]) -> dict[str, float]:
    return {
        "hypothesis_update_score": hypothesis_update_score([session]),
        "contradiction_sensitivity": contradiction_sensitivity([session]),
        "online_adaptation_gain": online_adaptation_gain([session]),
        "attention_recovery_score": attention_recovery_score([session]),
        "distractor_capture_rate": distractor_capture_rate([session]),
        "cue_utilization_score": cue_utilization_score([session]),
        "trust_revision_score": trust_revision_score([session]),
        "belief_state_consistency": belief_state_consistency([session]),
        "deception_sensitivity": deception_sensitivity([session]),
        "multi_turn_adaptation_score": multi_turn_adaptation_score([session]),
        "episode_cognitive_flexibility_score": episode_cognitive_flexibility_score([session]),
    }


def _baseline_session_score(task: dict[str, Any], session: dict[str, Any]) -> tuple[float, dict[str, float]]:
    metrics = _session_metric_summary(session)
    revised_correct = 1.0 if session["derived"]["revised_correct"] else 0.0
    family = task["family"]

    if family in {"hidden_rule", "shift_transfer", "metacog_revision"}:
        components = [
            revised_correct,
            metrics["hypothesis_update_score"],
            metrics["contradiction_sensitivity"] if session["expected"]["contradiction_expected"] else 1.0,
            metrics["online_adaptation_gain"],
        ]
    elif family == "attention_distractors":
        components = [
            revised_correct,
            metrics["attention_recovery_score"],
            metrics["cue_utilization_score"],
            1.0 - metrics["distractor_capture_rate"],
            metrics["multi_turn_adaptation_score"],
        ]
    else:
        components = [
            revised_correct,
            metrics["trust_revision_score"],
            metrics["belief_state_consistency"],
            metrics["deception_sensitivity"],
            metrics["multi_turn_adaptation_score"],
        ]

    return (round(sum(components) / len(components), 4), metrics)


def _solve_threshold_for_family(family: str) -> float:
    if family in {"attention_distractors", "social_miniworlds"}:
        return 0.68
    return 0.74


def _template_signature(task: dict[str, Any]) -> tuple[Any, ...]:
    family = task["family"]
    if family == "hidden_rule":
        return (
            family,
            task["difficulty"],
            task["metadata"]["internal_rules"]["induction"]["name"],
            task["metadata"]["internal_rules"]["shifted"]["name"],
        )
    if family == "shift_transfer":
        return (
            family,
            task["difficulty"],
            task["metadata"]["internal_rule"]["name"],
            task["metadata"].get("remap_profile", "direct_remap"),
            int(task["metadata"].get("remap_composition_depth", 1)),
            task["metadata"].get("bridge_representation_mode", "none"),
            task["metadata"].get("latent_rule_mix", "baseline"),
            int(task["metadata"].get("anti_anchor_strength", task["metadata"].get("anti_template_strength", 0))),
            tuple(task["metadata"]["source_vocab"][:2]),
            tuple(task["metadata"]["transfer_vocab"][:2]),
        )
    if family == "metacog_revision":
        return (
            family,
            task["difficulty"],
            task["metadata"]["actual_rule"]["name"],
            tuple(rule["name"] for rule in task["metadata"]["candidate_rules"]),
        )
    if family == "attention_distractors":
        return (
            family,
            task["difficulty"],
            task["metadata"]["internal_rule"]["name"],
            task["distractor_level"],
            int(task["metadata"].get("cue_delay_level", 1)),
            task["metadata"].get("adversarial_query_mode", "first_candidate"),
            tuple(sorted(set(task["metadata"].get("distractor_profiles", [])))[:3]),
        )
    return (
        family,
        task["difficulty"],
        task["metadata"]["witness"],
        task["metadata"]["deceiver"],
        task["metadata"]["rumor_agent"],
    )


def _template_novelty_map(tasks: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[tuple[str, tuple[Any, ...]]]] = defaultdict(list)
    for task in tasks:
        signature = _template_signature(task)
        grouped[task["family"]].append((task["task_id"], signature))

    novelty: dict[str, float] = {}
    for family, task_signatures in grouped.items():
        signatures = [signature for _, signature in task_signatures]
        counts = Counter(signatures)
        max_count = max(counts.values())
        min_count = min(counts.values())
        for task_id, signature in task_signatures:
            count = counts[signature]
            if len(counts) == 1:
                novelty[task_id] = 1.0 if count == 1 else 0.0
            elif max_count == min_count:
                novelty[task_id] = 1.0
            else:
                novelty[task_id] = round(1.0 - ((count - min_count) / (max_count - min_count)), 4)
    return novelty


def _trajectory_value(spec: EpisodeSpec) -> float:
    base = min(1.0, (len(spec.turns) - 1) / 4.0)
    if spec.expected["should_update"]:
        base += 0.2
    if spec.expected["contradiction_expected"]:
        base += 0.15
    if spec.family in {"attention_distractors", "social_miniworlds"}:
        base += 0.2
    if spec.expected.get("cue_turn", 0) >= 3:
        base += 0.1
    return round(min(base, 1.0), 4)


def _decision_for_scores(scores: dict[str, float], policy: CurationPolicy) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if scores["baseline_solve_rate"] >= policy.reject_baseline_solve_rate:
        reasons.append("too_easy")
    if scores["shortcut_vulnerability_score"] >= policy.reject_shortcut_vulnerability:
        reasons.append("shortcut_solvable")
    if scores["template_novelty_score"] < policy.reject_template_novelty:
        reasons.append("too_templated")
    if scores["revision_discrimination_score"] < policy.reject_revision_discrimination:
        reasons.append("weak_revision_signal")
    if scores["trajectory_value_score"] < policy.reject_trajectory_value:
        reasons.append("weak_adaptive_signal")

    family_specific_metrics = [
        scores.get("distractor_discrimination_score"),
        scores.get("social_reasoning_discrimination_score"),
        scores.get("transfer_depth_score"),
    ]
    concrete_family_scores = [metric for metric in family_specific_metrics if metric is not None]
    if concrete_family_scores and max(concrete_family_scores) < policy.reject_family_specific:
        reasons.append("insufficiently_discriminative")

    if reasons:
        return ("reject", reasons)
    if scores["benchmark_signal_score"] >= policy.keep_min_signal and scores["baseline_solve_rate"] <= policy.keep_max_baseline_solve_rate:
        return ("keep", [])
    if scores["benchmark_signal_score"] < policy.review_min_signal:
        return ("reject", ["low_benchmark_signal"])
    return ("review", ["borderline_signal"])


def _aggregate_task_scores(task: dict[str, Any], spec: EpisodeSpec, baseline_results: list[BaselineResult], template_novelty: float) -> dict[str, float]:
    results_by_name = {result.solver_name: result for result in baseline_results}
    baseline_solve_rate = round(sum(1 for result in baseline_results if result.solved) / len(baseline_results), 4)
    shortcut_vulnerability = round(sum(result.session_score for result in baseline_results) / len(baseline_results), 4)
    revision_discrimination = round(1.0 - results_by_name["no_revision_solver"].session_score, 4)
    distractor_discrimination = None
    social_discrimination = None
    transfer_depth = None

    if task["family"] == "attention_distractors":
        distractor_discrimination = round(1.0 - results_by_name["distractor_vulnerable_solver"].session_score, 4)
    if task["family"] == "social_miniworlds":
        social_discrimination = round(1.0 - results_by_name["trust_naive_solver"].session_score, 4)
    if task["family"] == "shift_transfer":
        transfer_depth = round(1.0 - results_by_name["representation_anchor_solver"].session_score, 4)

    trajectory_value = _trajectory_value(spec)
    components = [
        1.0 - baseline_solve_rate,
        1.0 - shortcut_vulnerability,
        revision_discrimination,
        trajectory_value,
        template_novelty,
    ]
    for maybe_metric in (distractor_discrimination, social_discrimination, transfer_depth):
        if maybe_metric is not None:
            components.append(maybe_metric)
    benchmark_signal = round(sum(components) / len(components), 4)

    return {
        "baseline_solve_rate": baseline_solve_rate,
        "shortcut_vulnerability_score": shortcut_vulnerability,
        "revision_discrimination_score": revision_discrimination,
        "distractor_discrimination_score": distractor_discrimination,
        "social_reasoning_discrimination_score": social_discrimination,
        "transfer_depth_score": transfer_depth,
        "trajectory_value_score": trajectory_value,
        "template_novelty_score": template_novelty,
        "benchmark_signal_score": benchmark_signal,
    }


def score_task_for_curation(
    task: dict[str, Any],
    policy: CurationPolicy | None = None,
    template_novelty: float = 1.0,
    baselines: dict[str, BaselineResponder] | None = None,
) -> dict[str, Any]:
    """Score one task and attach a deterministic keep/review/reject decision."""
    active_policy = policy or CurationPolicy()
    spec = build_interaction_spec(task)
    baseline_results = run_baseline_suite(task, baselines=baselines)
    score_payload = _aggregate_task_scores(task, spec, baseline_results, template_novelty)
    decision, reasons = _decision_for_scores(score_payload, active_policy)
    return {
        "episode_type": spec.episode_type,
        "scores": score_payload,
        "baseline_results": [result.to_dict() for result in baseline_results],
        "decision": decision,
        "reasons": reasons,
    }


def curate_tasks(tasks: list[dict[str, Any]], policy: CurationPolicy | None = None) -> dict[str, Any]:
    """Run adversarial curation over AGUS tasks."""
    active_policy = policy or CurationPolicy()
    template_novelty = _template_novelty_map(tasks)

    curated_tasks: list[dict[str, Any]] = []
    rejected_tasks: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    rejection_reason_counter: Counter[str] = Counter()
    family_counters: dict[str, Counter[str]] = defaultdict(Counter)

    for task in tasks:
        curation = score_task_for_curation(
            task,
            policy=active_policy,
            template_novelty=template_novelty.get(task["task_id"], 1.0),
        )
        decision = curation["decision"]
        reasons = curation["reasons"]
        family_counters[task["family"]][decision] += 1
        rejection_reason_counter.update(reasons)

        record = {
            **task,
            "curation": curation,
        }

        report_rows.append(
            {
                "task_id": task["task_id"],
                "family": task["family"],
                "decision": decision,
                "benchmark_signal_score": curation["scores"]["benchmark_signal_score"],
                "trajectory_value_score": curation["scores"]["trajectory_value_score"],
                "reasons": reasons,
            }
        )

        if decision == "reject":
            rejected_tasks.append(record)
        else:
            curated_tasks.append(record)

    per_family_retention = {}
    top_high_signal_by_family: dict[str, list[dict[str, Any]]] = {}
    for family in sorted({task["family"] for task in tasks}):
        family_tasks = [task for task in curated_tasks + rejected_tasks if task["family"] == family]
        total = len(family_tasks)
        kept = family_counters[family]["keep"]
        reviewed = family_counters[family]["review"]
        rejected = family_counters[family]["reject"]
        per_family_retention[family] = {
            "total": total,
            "kept": kept,
            "reviewed": reviewed,
            "rejected": rejected,
            "retention_rate": round(kept / total, 4) if total else 0.0,
        }
        ranked = sorted(
            family_tasks,
            key=lambda row: row["curation"]["scores"]["benchmark_signal_score"],
            reverse=True,
        )[:5]
        top_high_signal_by_family[family] = [
            {
                "task_id": row["task_id"],
                "decision": row["curation"]["decision"],
                "benchmark_signal_score": row["curation"]["scores"]["benchmark_signal_score"],
            }
            for row in ranked
        ]

    report = {
        "total_tasks_processed": len(tasks),
        "kept_count": sum(1 for row in curated_tasks if row["curation"]["decision"] == "keep"),
        "review_count": sum(1 for row in curated_tasks if row["curation"]["decision"] == "review"),
        "rejected_count": len(rejected_tasks),
        "most_common_rejection_reasons": dict(rejection_reason_counter.most_common()),
        "per_family_retention_rates": per_family_retention,
        "top_high_signal_tasks_by_family": top_high_signal_by_family,
        "task_index": report_rows,
    }

    return {
        "curated_tasks": curated_tasks,
        "rejected_tasks": rejected_tasks,
        "curation_report": report,
    }
