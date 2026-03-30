"""Model-agnostic benchmark metrics for AGUS."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def exact_match(predicted: Any, target: Any) -> bool:
    """Exact match with recursive list normalization."""
    return _normalize(predicted) == _normalize(target)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _extract_rule_text(session: dict[str, Any], prefix: str) -> str:
    response = session["response"]
    metadata = response.get("metadata", {})
    return str(metadata.get(f"{prefix}_rule_tag") or response.get(f"{prefix}_rule_explanation", "")).strip().lower()


def _rule_matches(observed: str, canonical: Any) -> bool:
    if canonical is None:
        return False
    if isinstance(canonical, list):
        return any(_rule_matches(observed, candidate) for candidate in canonical)
    canonical_text = str(canonical).strip().lower()
    return observed == canonical_text or canonical_text in observed or observed in canonical_text


def _turn_records(session: dict[str, Any]) -> list[dict[str, Any]]:
    return session.get("turns", [])


def _turn_confidences(session: dict[str, Any]) -> list[float]:
    return [float(record["model_response"]["confidence"]) for record in _turn_records(session)]


def _turn_expected_confidences(session: dict[str, Any]) -> list[float]:
    return [float(record["turn_spec"]["expected_confidence"]) for record in _turn_records(session)]


def _turn_correctness(session: dict[str, Any]) -> list[bool]:
    if "turn_correctness" in session.get("derived", {}):
        return [bool(value) for value in session["derived"]["turn_correctness"]]
    return [bool(record["derived"]["answer_correct"]) for record in _turn_records(session)]


def _turn_quality(record: dict[str, Any]) -> float:
    pieces = [1.0 if record["derived"]["answer_correct"] else 0.0]
    if record["derived"].get("signal_focus_correct") is not None:
        pieces.append(1.0 if record["derived"]["signal_focus_correct"] else 0.0)
    if record["derived"].get("trust_top_correct") is not None:
        pieces.append(1.0 if record["derived"]["trust_top_correct"] else 0.0)
    belief_consistency = record["derived"].get("belief_consistency")
    if belief_consistency is not None:
        pieces.append(float(belief_consistency))
    return sum(pieces) / len(pieces)


def _top_agent(scores: dict[str, float]) -> str | None:
    if not scores:
        return None
    return max(scores.items(), key=lambda item: (item[1], item[0]))[0]


def score_accuracy(tasks: list[dict], predictions: dict[str, dict]) -> dict[str, Any]:
    """Aggregate exact-match accuracy over all scorable answer slots."""
    total = 0
    correct = 0
    family_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})

    for task in tasks:
        task_id = task["task_id"]
        family = task["family"]
        pred = predictions.get(task_id, {})
        answer = task["answer"]

        if family == "hidden_rule":
            for idx, target in enumerate(answer["induction_targets"]):
                ok = idx < len(pred.get("induction_predictions", [])) and exact_match(
                    pred["induction_predictions"][idx], target
                )
                total += 1
                correct += int(ok)
                family_counts[family]["total"] += 1
                family_counts[family]["correct"] += int(ok)

            for idx, target in enumerate(answer["shift_targets"]):
                ok = idx < len(pred.get("shift_predictions", [])) and exact_match(
                    pred["shift_predictions"][idx], target
                )
                total += 1
                correct += int(ok)
                family_counts[family]["total"] += 1
                family_counts[family]["correct"] += int(ok)

        elif family == "shift_transfer":
            source_ok = exact_match(pred.get("source_prediction"), answer["source_target"])
            transfer_ok = exact_match(pred.get("transfer_prediction"), answer["transfer_target"])
            for ok in (source_ok, transfer_ok):
                total += 1
                correct += int(ok)
                family_counts[family]["total"] += 1
                family_counts[family]["correct"] += int(ok)

        elif family == "metacog_revision":
            acceptable = answer["acceptable_initial_targets"]
            initial_ok = any(exact_match(pred.get("initial_answer"), candidate) for candidate in acceptable)
            revised_ok = exact_match(pred.get("revised_answer"), answer["revised_target"])
            for ok in (initial_ok, revised_ok):
                total += 1
                correct += int(ok)
                family_counts[family]["total"] += 1
                family_counts[family]["correct"] += int(ok)

        elif family == "attention_distractors":
            ok = exact_match(pred.get("prediction"), answer["target"])
            total += 1
            correct += int(ok)
            family_counts[family]["total"] += 1
            family_counts[family]["correct"] += int(ok)

        elif family == "social_miniworlds":
            checks = (
                exact_match(pred.get("actual_location_prediction"), answer["actual_location"]),
                exact_match(pred.get("belief_prediction"), answer["belief_of_false_belief_agent"]),
                exact_match(pred.get("trust_prediction"), answer["most_reliable_agent"]),
            )
            for ok in checks:
                total += 1
                correct += int(ok)
                family_counts[family]["total"] += 1
                family_counts[family]["correct"] += int(ok)

    family_accuracy = {
        family: round(counts["correct"] / counts["total"], 4) if counts["total"] else 0.0
        for family, counts in family_counts.items()
    }
    return {
        "accuracy": round(correct / total, 4) if total else 0.0,
        "family_accuracy": family_accuracy,
        "num_scored_items": total,
    }


def score_adaptation_speed(tasks: list[dict], predictions: dict[str, dict]) -> float:
    """Measure how quickly models stabilize after a hidden-rule shift."""
    scores: list[float] = []
    for task in tasks:
        if task["family"] != "hidden_rule":
            continue
        pred = predictions.get(task["task_id"], {})
        correctness = []
        for idx, target in enumerate(task["answer"]["shift_targets"]):
            ok = idx < len(pred.get("shift_predictions", [])) and exact_match(
                pred["shift_predictions"][idx], target
            )
            correctness.append(ok)
        if not correctness:
            scores.append(0.0)
            continue
        score = 0.0
        for start in range(len(correctness)):
            if all(correctness[start:]):
                score = (len(correctness) - start) / len(correctness)
                break
        scores.append(score)
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def score_transfer(tasks: list[dict], predictions: dict[str, dict]) -> float:
    """Measure structure-preserving transfer across representations."""
    total = 0
    correct = 0
    for task in tasks:
        if task["family"] != "shift_transfer":
            continue
        total += 1
        pred = predictions.get(task["task_id"], {})
        if exact_match(pred.get("transfer_prediction"), task["answer"]["transfer_target"]):
            correct += 1
    return round(correct / total, 4) if total else 0.0


def score_calibration(tasks: list[dict], predictions: dict[str, dict]) -> float:
    """Compare confidence against correctness or expected ambiguity."""
    scores: list[float] = []
    for task in tasks:
        if task["family"] != "metacog_revision":
            continue
        pred = predictions.get(task["task_id"], {})
        initial_conf = float(pred.get("initial_confidence", 0.0))
        revised_conf = float(pred.get("revised_confidence", 0.0))

        initial_target = float(task["metadata"].get("expected_initial_certainty", 0.5))
        revised_correct = 1.0 if exact_match(pred.get("revised_answer"), task["answer"]["revised_target"]) else 0.0

        initial_score = 1.0 - (initial_conf - initial_target) ** 2
        revised_score = 1.0 - (revised_conf - revised_correct) ** 2
        scores.append((initial_score + revised_score) / 2.0)
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def score_revision_quality(tasks: list[dict], predictions: dict[str, dict]) -> float:
    """Measure whether answers and confidence are revised appropriately."""
    scores: list[float] = []
    for task in tasks:
        if task["family"] != "metacog_revision":
            continue
        pred = predictions.get(task["task_id"], {})
        initial_answer = pred.get("initial_answer")
        revised_answer = pred.get("revised_answer")
        answer_changed = initial_answer != revised_answer
        should_revise = bool(task["answer"].get("should_revise", False))
        revised_correct = exact_match(revised_answer, task["answer"]["revised_target"])
        initial_conf = float(pred.get("initial_confidence", 0.0))
        revised_conf = float(pred.get("revised_confidence", 0.0))
        confidence_updated = revised_conf > initial_conf
        rule_guess_changed = (
            pred.get("initial_rule_guess") is not None
            and pred.get("revised_rule_guess") is not None
            and pred.get("initial_rule_guess") != pred.get("revised_rule_guess")
        )
        belief_updated = answer_changed or confidence_updated or rule_guess_changed

        components = [
            1.0 if belief_updated == should_revise else 0.0,
            1.0 if revised_correct else 0.0,
            1.0 if confidence_updated or rule_guess_changed else 0.0,
        ]
        scores.append(sum(components) / len(components))
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def score_distractor_robustness(tasks: list[dict], predictions: dict[str, dict]) -> dict[str, Any]:
    """Measure how performance changes as distractor load increases."""
    grouped: dict[int, list[float]] = defaultdict(list)
    for task in tasks:
        if task["family"] != "attention_distractors":
            continue
        pred = predictions.get(task["task_id"], {})
        correct = 1.0 if exact_match(pred.get("prediction"), task["answer"]["target"]) else 0.0
        grouped[int(task["distractor_level"])].append(correct)

    if not grouped:
        return {"distractor_robustness": 0.0, "attention_accuracy_by_level": {}}

    accuracy_by_level = {
        level: round(sum(values) / len(values), 4) if values else 0.0
        for level, values in sorted(grouped.items())
    }
    levels = sorted(accuracy_by_level)
    low_acc = accuracy_by_level[levels[0]]
    high_acc = accuracy_by_level[levels[-1]]
    degradation = max(0.0, low_acc - high_acc)
    robustness = max(0.0, min(1.0, 1.0 - degradation))

    return {
        "distractor_robustness": round(robustness, 4),
        "attention_accuracy_by_level": accuracy_by_level,
    }


def hypothesis_update_score(sessions: list[dict[str, Any]]) -> float:
    """Reward evidence-sensitive belief updates after interactive feedback."""
    scores: list[float] = []
    for session in sessions:
        expected = session["expected"]
        response = session["response"]
        derived = session["derived"]
        update_mode = expected["update_mode"]
        initial_rule = _extract_rule_text(session, "initial")
        revised_rule = _extract_rule_text(session, "revised")

        if update_mode == "change_rule":
            update_ok = derived["rule_changed"] or derived["answer_changed"]
        elif update_mode == "preserve_rule":
            update_ok = response["evidence_acknowledged"] and not derived["rule_changed"]
        else:
            update_ok = derived["belief_updated"]

        if update_mode == "preserve_rule":
            rule_alignment = _rule_matches(initial_rule, expected["canonical_initial_rule"]) and _rule_matches(
                revised_rule, expected["canonical_revised_rule"]
            )
        else:
            rule_alignment = _rule_matches(revised_rule, expected["canonical_revised_rule"])

        components = [
            1.0 if update_ok else 0.0,
            1.0 if rule_alignment else 0.0,
            1.0 if derived["revised_correct"] else 0.0,
        ]
        scores.append(sum(components) / len(components))
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def contradiction_sensitivity(sessions: list[dict[str, Any]]) -> float:
    """Measure whether the solver notices and reacts to conflicting evidence."""
    scores: list[float] = []
    for session in sessions:
        if not session["expected"]["contradiction_expected"]:
            continue
        response = session["response"]
        derived = session["derived"]
        components = [
            1.0 if response["contradiction_detected"] else 0.0,
            1.0 if response["evidence_acknowledged"] else 0.0,
            1.0 if (derived["belief_updated"] or derived["revised_correct"]) else 0.0,
        ]
        scores.append(sum(components) / len(components))
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def confidence_recalibration_score(sessions: list[dict[str, Any]]) -> float:
    """Measure whether confidence moves in the expected direction over the episode."""
    scores: list[float] = []
    for session in sessions:
        observed = _turn_confidences(session)
        expected = _turn_expected_confidences(session)
        if not observed or not expected:
            response = session["response"]
            target_initial = float(session["expected"]["expected_initial_confidence"])
            target_revised = float(session["expected"]["expected_revised_confidence"])
            observed = [float(response["initial_confidence"]), float(response["revised_confidence"])]
            expected = [target_initial, target_revised]

        closeness_terms = [1.0 - (obs - exp) ** 2 for obs, exp in zip(observed, expected)]
        closeness = sum(closeness_terms) / len(closeness_terms)

        direction_hits = []
        for obs_prev, obs_next, exp_prev, exp_next in zip(observed, observed[1:], expected, expected[1:]):
            target_delta = exp_next - exp_prev
            observed_delta = obs_next - obs_prev
            if abs(target_delta) < 1e-9:
                direction_hits.append(abs(observed_delta) <= 0.05)
            else:
                direction_hits.append(target_delta * observed_delta > 0)
        direction_score = sum(direction_hits) / len(direction_hits) if direction_hits else 1.0
        scores.append((_clamp(closeness) + direction_score) / 2.0)
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def online_adaptation_gain(sessions: list[dict[str, Any]]) -> float:
    """Measure improvement from the initial turn to the revised turn."""
    scores: list[float] = []
    for session in sessions:
        correctness = _turn_correctness(session)
        initial_correct = correctness[0]
        revised_correct = correctness[-1]
        if revised_correct and not initial_correct:
            scores.append(1.0)
        elif revised_correct and initial_correct:
            scores.append(0.5)
        else:
            scores.append(0.0)
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def belief_trajectory_quality(sessions: list[dict[str, Any]]) -> float:
    """Composite score for interactive revision behavior over time."""
    components = [
        hypothesis_update_score(sessions),
        contradiction_sensitivity(sessions),
        confidence_recalibration_score(sessions),
        online_adaptation_gain(sessions),
    ]
    usable = [component for component in components if component is not None]
    return round(sum(usable) / len(usable), 4) if usable else 0.0


def attention_recovery_score(sessions: list[dict[str, Any]]) -> float:
    """Measure whether attention returns to the relevant signal after a cue appears."""
    scores: list[float] = []
    for session in sessions:
        if session["family"] != "attention_distractors":
            continue
        cue_turn = int(session["expected"]["cue_turn"])
        post_cue = _turn_records(session)[cue_turn - 1 :]
        if not post_cue:
            continue
        post_quality = sum(_turn_quality(record) for record in post_cue) / len(post_cue)
        recovered = 1.0 if session["derived"].get("recovery_turn") is not None else 0.0
        scores.append((post_quality + recovered) / 2.0)
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def distractor_capture_rate(sessions: list[dict[str, Any]]) -> float:
    """Measure how often early attention is captured by distractors before the cue."""
    rates: list[float] = []
    for session in sessions:
        if session["family"] != "attention_distractors":
            continue
        cue_turn = int(session["expected"]["cue_turn"])
        pre_turns = max(cue_turn - 1, 1)
        capture_turns = [
            turn_index
            for turn_index in session["derived"].get("capture_turns", [])
            if turn_index < cue_turn
        ]
        rates.append(_clamp(len(capture_turns) / pre_turns))
    return round(sum(rates) / len(rates), 4) if rates else 0.0


def cue_utilization_score(sessions: list[dict[str, Any]]) -> float:
    """Measure whether the disambiguating cue improves post-cue reasoning."""
    scores: list[float] = []
    for session in sessions:
        if session["family"] != "attention_distractors":
            continue
        cue_turn = int(session["expected"]["cue_turn"])
        turns = _turn_records(session)
        pre_cue = turns[: cue_turn - 1]
        post_cue = turns[cue_turn - 1 :]
        if not post_cue:
            continue
        pre_quality = sum(_turn_quality(record) for record in pre_cue) / len(pre_cue) if pre_cue else 0.0
        post_quality = sum(_turn_quality(record) for record in post_cue) / len(post_cue)
        cue_ack = 1.0 if post_cue[0]["model_response"]["evidence_acknowledged"] else 0.0
        improvement = _clamp((post_quality - pre_quality + 1.0) / 2.0)
        scores.append((improvement + post_quality + cue_ack) / 3.0)
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def trust_revision_score(sessions: list[dict[str, Any]]) -> float:
    """Measure whether trust shifts toward the reliable agent over the episode."""
    scores: list[float] = []
    for session in sessions:
        if session["family"] != "social_miniworlds":
            continue
        reliable = session["expected"]["reliable_agent"]
        deceptive = session["expected"]["deceptive_agent"]
        trust_turns = [record["model_response"]["trust_scores_by_agent"] for record in _turn_records(session) if record["model_response"]["trust_scores_by_agent"]]
        if not trust_turns:
            continue
        initial = trust_turns[0]
        final = trust_turns[-1]
        initial_margin = initial.get(reliable, 0.0) - initial.get(deceptive, 0.0)
        final_margin = final.get(reliable, 0.0) - final.get(deceptive, 0.0)
        components = [
            1.0 if _top_agent(final) == reliable else 0.0,
            1.0 if final_margin > 0 else 0.0,
            1.0 if final_margin > initial_margin else 0.0,
        ]
        scores.append(sum(components) / len(components))
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def belief_state_consistency(sessions: list[dict[str, Any]]) -> float:
    """Measure whether agent belief traces remain internally consistent across turns."""
    scores: list[float] = []
    for session in sessions:
        if session["family"] != "social_miniworlds":
            continue
        trace = [record["derived"]["belief_consistency"] for record in _turn_records(session) if record["derived"]["belief_consistency"] is not None]
        if trace:
            scores.append(sum(trace) / len(trace))
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def deception_sensitivity(sessions: list[dict[str, Any]]) -> float:
    """Measure whether deceptive or conflicting testimony is detected and handled."""
    scores: list[float] = []
    for session in sessions:
        if session["family"] != "social_miniworlds":
            continue
        deceptive = session["expected"]["deceptive_agent"]
        reliable = session["expected"]["reliable_agent"]
        response = session["response"]
        trust_final = session["response"].get("trust_scores_by_agent", {})
        components = [
            1.0 if response["contradiction_detected"] else 0.0,
            1.0 if trust_final.get(reliable, 0.0) > trust_final.get(deceptive, 0.0) else 0.0,
            1.0 if "trust_revised" in response.get("revision_events", []) or "intent_revised" in response.get("revision_events", []) else 0.0,
        ]
        scores.append(sum(components) / len(components))
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def multi_turn_adaptation_score(sessions: list[dict[str, Any]]) -> float:
    """Measure adaptation quality in multi-turn episodes."""
    scores: list[float] = []
    for session in sessions:
        if session["derived"].get("num_turns", 0) <= 2:
            continue
        correctness = _turn_correctness(session)
        initial_quality = float(correctness[0])
        final_quality = float(correctness[-1])
        gain = _clamp(final_quality - initial_quality + 0.5)

        if session["family"] == "attention_distractors":
            recovery = 1.0 if session["derived"].get("recovery_turn") is not None else 0.0
            scores.append((gain + final_quality + recovery) / 3.0)
            continue

        if session["family"] == "social_miniworlds":
            belief_score = session["derived"].get("belief_consistency_trace", [])
            belief_tail = belief_score[-1] if belief_score and belief_score[-1] is not None else 0.0
            trust_trace = session["derived"].get("trust_trace", [])
            trust_final = 1.0 if trust_trace and trust_trace[-1] == session["expected"]["reliable_agent"] else 0.0
            scores.append((gain + float(belief_tail) + trust_final) / 3.0)
            continue

        scores.append((gain + final_quality) / 2.0)
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def episode_cognitive_flexibility_score(sessions: list[dict[str, Any]]) -> float:
    """Aggregate dynamic flexibility across the interactive suite."""
    metrics = [
        hypothesis_update_score(sessions),
        contradiction_sensitivity(sessions),
        confidence_recalibration_score(sessions),
        online_adaptation_gain(sessions),
        multi_turn_adaptation_score(sessions),
    ]
    if any(session["family"] == "attention_distractors" for session in sessions):
        metrics.extend(
            [
                attention_recovery_score(sessions),
                cue_utilization_score(sessions),
                1.0 - distractor_capture_rate(sessions),
            ]
        )
    if any(session["family"] == "social_miniworlds" for session in sessions):
        metrics.extend(
            [
                trust_revision_score(sessions),
                belief_state_consistency(sessions),
                deception_sensitivity(sessions),
            ]
        )
    return round(sum(metrics) / len(metrics), 4) if metrics else 0.0


def _get_nested(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if isinstance(value, dict):
            value = value.get(part)
            continue
        if isinstance(value, list):
            try:
                index = int(part)
            except ValueError:
                return None
            if index < 0 or index >= len(value):
                return None
            value = value[index]
            continue
        return None
    return value


def counterfactual_update_fidelity(bundle_rows: list[dict[str, Any]]) -> float:
    """Measure whether revision behavior matches the branch-specific counterfactual."""
    scores: list[float] = []
    for bundle in bundle_rows:
        for branch in bundle.get("branches", []):
            branch_kind = branch.get("expected_revision_kind", "belief")
            session = branch["session"]
            response = session["response"]
            derived = session["derived"]
            contradiction_expected = bool(session["expected"].get("contradiction_expected"))

            if branch_kind == "none":
                components = [
                    1.0 if not derived["answer_changed"] else 0.0,
                    1.0 if not derived["rule_changed"] else 0.0,
                    1.0 if derived["revised_correct"] else 0.0,
                ]
            elif branch_kind == "rule":
                components = [
                    1.0 if derived["rule_changed"] or derived["answer_changed"] else 0.0,
                    1.0 if response["contradiction_detected"] == contradiction_expected else 0.0,
                    1.0 if derived["revised_correct"] else 0.0,
                ]
            elif branch_kind == "representation":
                components = [
                    1.0 if not derived["rule_changed"] else 0.0,
                    1.0 if derived["answer_changed"] else 0.0,
                    1.0 if derived["revised_correct"] else 0.0,
                ]
            else:
                components = [
                    1.0 if derived["belief_updated"] else 0.0,
                    1.0 if response["contradiction_detected"] == contradiction_expected else 0.0,
                    1.0 if derived["revised_correct"] else 0.0,
                ]
            scores.append(sum(components) / len(components))
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def invariant_preservation_score(bundle_rows: list[dict[str, Any]]) -> float:
    """Measure whether shared structure is preserved across nearby branches."""
    scores: list[float] = []
    for bundle in bundle_rows:
        checks = bundle.get("shared_checks", [])
        for branch in bundle.get("branches", []):
            session = branch["session"]
            for check in checks:
                observed = _get_nested(session, check["path"])
                scores.append(1.0 if exact_match(observed, check["expected"]) else 0.0)
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def branch_belief_coherence(bundle_rows: list[dict[str, Any]]) -> float:
    """Measure internal coherence inside each branch trajectory."""
    scores: list[float] = []
    for bundle in bundle_rows:
        for branch in bundle.get("branches", []):
            session = branch["session"]
            if bundle["family"] == "social_miniworlds":
                belief_trace = session["derived"].get("belief_consistency_trace", [])
                belief_tail = belief_trace[-1] if belief_trace and belief_trace[-1] is not None else 0.0
                trust_trace = session["derived"].get("trust_trace", [])
                trust_ok = (
                    1.0
                    if trust_trace and trust_trace[-1] == session["expected"]["reliable_agent"]
                    else 0.0
                )
                scores.append((float(belief_tail) + trust_ok + float(session["derived"]["revised_correct"])) / 3.0)
                continue

            initial_rule = _get_nested(session, "response.metadata.initial_rule_tag")
            revised_rule = _get_nested(session, "response.metadata.revised_rule_tag")
            expected_initial = session["expected"].get("canonical_initial_rule")
            expected_revised = session["expected"].get("canonical_revised_rule")
            components = [
                1.0 if _rule_matches(str(initial_rule or ""), expected_initial) else 0.0,
                1.0 if _rule_matches(str(revised_rule or ""), expected_revised) else 0.0,
                1.0 if session["derived"]["revised_correct"] else 0.0,
            ]
            scores.append(sum(components) / len(components))
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def cross_branch_consistency(bundle_rows: list[dict[str, Any]]) -> float:
    """Measure whether outputs change when branch outcomes should change, and stay aligned otherwise."""
    scores: list[float] = []
    for bundle in bundle_rows:
        branches = bundle.get("branches", [])
        for idx, left in enumerate(branches):
            for right in branches[idx + 1 :]:
                left_expected = left["expected_final_target"]
                right_expected = right["expected_final_target"]
                left_observed = _get_nested(left["session"], bundle["variant_path"])
                right_observed = _get_nested(right["session"], bundle["variant_path"])
                expected_equal = exact_match(left_expected, right_expected)
                observed_equal = exact_match(left_observed, right_observed)
                scores.append(1.0 if expected_equal == observed_equal else 0.0)
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def counterfactual_confidence_calibration(bundle_rows: list[dict[str, Any]]) -> float:
    """Compare confidence movement against branch-specific expected confidence profiles."""
    scores: list[float] = []
    for bundle in bundle_rows:
        for branch in bundle.get("branches", []):
            observed = branch["session"]["response"].get("turn_confidences", [])
            expected = branch.get("expected_turn_confidences", [])
            if not observed or not expected:
                continue
            pairwise = []
            for observed_value, expected_value in zip(observed, expected):
                pairwise.append(1.0 - (float(observed_value) - float(expected_value)) ** 2)
            scores.append(sum(pairwise) / len(pairwise))
    return round(sum(scores) / len(scores), 4) if scores else 0.0
