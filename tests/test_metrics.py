from src.generators.attention_distractors import AttentionDistractorConfig, generate_attention_distractor_tasks
from src.generators.hidden_rule import HiddenRuleConfig, generate_hidden_rule_tasks
from src.generators.metacog_revision import MetacogRevisionConfig, generate_metacog_revision_tasks
from src.generators.social_miniworlds import SocialMiniworldConfig, generate_social_miniworld_tasks
from src.generators.shift_transfer import ShiftTransferConfig, generate_shift_transfer_tasks
from src.scoring.evaluator import evaluate_predictions


def _oracle_predictions(tasks):
    rows = []
    for task in tasks:
        family = task["family"]
        if family == "hidden_rule":
            rows.append(
                {
                    "task_id": task["task_id"],
                    "induction_predictions": task["answer"]["induction_targets"],
                    "shift_predictions": task["answer"]["shift_targets"],
                }
            )
        elif family == "shift_transfer":
            rows.append(
                {
                    "task_id": task["task_id"],
                    "source_prediction": task["answer"]["source_target"],
                    "transfer_prediction": task["answer"]["transfer_target"],
                }
            )
        elif family == "metacog_revision":
            rows.append(
                {
                    "task_id": task["task_id"],
                    "initial_answer": task["answer"]["acceptable_initial_targets"][0],
                    "initial_confidence": task["metadata"]["expected_initial_certainty"],
                    "revised_answer": task["answer"]["revised_target"],
                    "revised_confidence": 1.0,
                }
            )
        elif family == "attention_distractors":
            rows.append(
                {
                    "task_id": task["task_id"],
                    "prediction": task["answer"]["target"],
                }
            )
        elif family == "social_miniworlds":
            rows.append(
                {
                    "task_id": task["task_id"],
                    "actual_location_prediction": task["answer"]["actual_location"],
                    "belief_prediction": task["answer"]["belief_of_false_belief_agent"],
                    "trust_prediction": task["answer"]["most_reliable_agent"],
                }
            )
    return rows


def test_metrics_reach_oracle_scores():
    tasks = []
    tasks.extend(generate_hidden_rule_tasks(HiddenRuleConfig(count=2, seed=11)))
    tasks.extend(generate_shift_transfer_tasks(ShiftTransferConfig(count=2, seed=23)))
    tasks.extend(generate_metacog_revision_tasks(MetacogRevisionConfig(count=2, seed=37)))
    tasks.extend(generate_attention_distractor_tasks(AttentionDistractorConfig(count=4, seed=51)))
    tasks.extend(generate_social_miniworld_tasks(SocialMiniworldConfig(count=2, seed=67)))

    scores = evaluate_predictions(tasks, _oracle_predictions(tasks))
    assert scores["accuracy"] == 1.0
    assert scores["adaptation_speed"] == 1.0
    assert scores["transfer_score"] == 1.0
    assert scores["calibration_score"] == 1.0
    assert scores["revision_quality"] == 1.0
    assert scores["distractor_robustness"] == 1.0

