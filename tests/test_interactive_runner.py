from src.eval.interactive_runner import build_interaction_spec, run_interactive_session
from src.generators.attention_distractors import AttentionDistractorConfig, generate_attention_distractor_tasks
from src.generators.hidden_rule import HiddenRuleConfig, generate_hidden_rule_tasks
from src.generators.metacog_revision import MetacogRevisionConfig, generate_metacog_revision_tasks
from src.generators.social_miniworlds import SocialMiniworldConfig, generate_social_miniworld_tasks


def _scripted_responder(spec, turn, _prior_turns):
    rule = turn.expected_rule
    if isinstance(rule, list):
        rule = rule[0]
    payload = {
        "answer": turn.accepted_answers[0],
        "confidence": turn.expected_confidence,
        "rule_explanation": str(rule or "structured_reasoning"),
        "evidence_acknowledged": turn.event is not None,
        "contradiction_detected": bool(turn.event and turn.event["event_type"] in {
            "rule_shift_contradiction",
            "corrective_evidence",
            "conflicting_testimony",
            "conflicting_reports",
            "conflicting_statement",
            "noise_escalation",
            "disambiguating_cue",
            "incentive_reveal",
            "intent_reveal",
        }),
        "attended_signals": list(turn.expected_attended_signals),
        "ignored_signals": list(turn.expected_ignored_signals),
        "trust_scores_by_agent": {turn.expected_trust_top: 1.0} if turn.expected_trust_top else {},
        "inferred_agent_beliefs": dict(turn.expected_beliefs),
        "revision_events": ["scripted_revision"] if turn.event is not None else [],
        "metadata": {"rule_tag": str(rule or "structured_reasoning")},
    }
    if spec.family == "social_miniworlds" and turn.expected_trust_top:
        payload["revision_events"].append("trust_revised")
    return {
        **payload,
    }


def test_hidden_rule_interaction_path_has_shift_evidence():
    task = generate_hidden_rule_tasks(HiddenRuleConfig(count=1, seed=11))[0]
    spec = build_interaction_spec(task)
    session = run_interactive_session(task, _scripted_responder)

    assert spec.family == "hidden_rule"
    assert spec.turns[1].event["event_type"] == "rule_shift_contradiction"
    assert spec.expected["accepted_initial_targets"][0] != spec.expected["accepted_revised_targets"][0]
    assert session["derived"]["initial_correct"] is True
    assert session["derived"]["revised_correct"] is True


def test_metacog_interaction_path_uses_corrective_evidence():
    task = generate_metacog_revision_tasks(MetacogRevisionConfig(count=1, seed=37))[0]
    spec = build_interaction_spec(task)
    session = run_interactive_session(task, _scripted_responder)

    assert spec.family == "metacog_revision"
    assert spec.turns[1].event["event_type"] == "corrective_evidence"
    assert len(spec.expected["accepted_initial_targets"]) >= 2
    assert session["response"]["contradiction_detected"] is True
    assert session["derived"]["revised_correct"] is True


def test_attention_episode_has_multi_turn_recovery_path():
    task = generate_attention_distractor_tasks(AttentionDistractorConfig(count=1, seed=51))[0]
    spec = build_interaction_spec(task)
    session = run_interactive_session(task, _scripted_responder)

    assert spec.family == "attention_distractors"
    assert len(spec.turns) == 4
    assert spec.turns[2].event["event_type"] == "disambiguating_cue"
    assert session["derived"]["recovery_turn"] == 3
    assert session["derived"]["revised_correct"] is True


def test_social_episode_has_multi_turn_belief_and_trust_trace():
    task = generate_social_miniworld_tasks(SocialMiniworldConfig(count=2, seed=67))[1]
    spec = build_interaction_spec(task)
    session = run_interactive_session(task, _scripted_responder)

    assert spec.family == "social_miniworlds"
    assert len(spec.turns) == 4
    assert spec.episode_type == "trust_revision"
    assert session["response"]["trust_scores_by_agent"][spec.expected["reliable_agent"]] == 1.0
    assert session["derived"]["belief_consistency_trace"][-1] == 1.0
