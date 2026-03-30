from src.scoring.metrics import (
    attention_recovery_score,
    belief_state_consistency,
    contradiction_sensitivity,
    distractor_capture_rate,
    hypothesis_update_score,
    trust_revision_score,
)


def test_hypothesis_update_score_rewards_meaningful_revision():
    sessions = [
        {
            "expected": {
                "update_mode": "change_rule",
                "canonical_initial_rule": "add_const",
                "canonical_revised_rule": "reverse_add",
                "contradiction_expected": True,
            },
            "response": {
                "initial_rule_explanation": "add_const",
                "revised_rule_explanation": "reverse_add",
                "evidence_acknowledged": True,
                "contradiction_detected": True,
                "metadata": {
                    "initial_rule_tag": "add_const",
                    "revised_rule_tag": "reverse_add",
                },
            },
            "derived": {
                "rule_changed": True,
                "answer_changed": True,
                "revised_correct": True,
                "belief_updated": True,
            },
        }
    ]

    assert hypothesis_update_score(sessions) == 1.0


def test_contradiction_sensitivity_requires_detection_and_revision():
    sessions = [
        {
            "expected": {"contradiction_expected": True},
            "response": {
                "contradiction_detected": True,
                "evidence_acknowledged": True,
            },
            "derived": {
                "belief_updated": True,
                "revised_correct": True,
            },
        }
    ]

    assert contradiction_sensitivity(sessions) == 1.0


def test_attention_recovery_and_capture_metrics():
    sessions = [
        {
            "family": "attention_distractors",
            "expected": {"cue_turn": 3},
            "derived": {"recovery_turn": 3, "capture_turns": [1, 2]},
            "turns": [
                {
                    "model_response": {"evidence_acknowledged": False},
                    "derived": {"answer_correct": False, "signal_focus_correct": False, "trust_top_correct": None, "belief_consistency": None},
                },
                {
                    "model_response": {"evidence_acknowledged": True},
                    "derived": {"answer_correct": False, "signal_focus_correct": False, "trust_top_correct": None, "belief_consistency": None},
                },
                {
                    "model_response": {"evidence_acknowledged": True},
                    "derived": {"answer_correct": True, "signal_focus_correct": True, "trust_top_correct": None, "belief_consistency": None},
                },
                {
                    "model_response": {"evidence_acknowledged": True},
                    "derived": {"answer_correct": True, "signal_focus_correct": True, "trust_top_correct": None, "belief_consistency": None},
                },
            ],
        }
    ]

    assert attention_recovery_score(sessions) == 1.0
    assert distractor_capture_rate(sessions) == 1.0


def test_social_revision_metrics_reward_consistent_belief_tracking():
    sessions = [
        {
            "family": "social_miniworlds",
            "expected": {
                "reliable_agent": "Ava",
                "deceptive_agent": "Ben",
            },
            "response": {
                "contradiction_detected": True,
                "trust_scores_by_agent": {"Ava": 0.9, "Ben": 0.1},
                "revision_events": ["trust_revised"],
            },
            "turns": [
                {
                    "model_response": {"trust_scores_by_agent": {"Ava": 0.4, "Ben": 0.6}},
                    "derived": {"belief_consistency": 1.0},
                },
                {
                    "model_response": {"trust_scores_by_agent": {"Ava": 0.7, "Ben": 0.2}},
                    "derived": {"belief_consistency": 1.0},
                },
                {
                    "model_response": {"trust_scores_by_agent": {"Ava": 0.9, "Ben": 0.1}},
                    "derived": {"belief_consistency": 1.0},
                },
            ],
        }
    ]

    assert trust_revision_score(sessions) == 1.0
    assert belief_state_consistency(sessions) == 1.0
