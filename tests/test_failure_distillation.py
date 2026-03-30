from pathlib import Path

from src.eval.adapters import build_adapter
from src.eval.failure_distillation import (
    assign_failure_categories,
    compare_distilled_failures,
    distill_run_failures,
    rank_failure_case,
)
from src.eval.model_runner import run_model_evaluation
from src.generators.attention_distractors import AttentionDistractorConfig, generate_attention_distractor_tasks
from src.generators.hidden_rule import HiddenRuleConfig, generate_hidden_rule_tasks
from src.generators.metacog_revision import MetacogRevisionConfig, generate_metacog_revision_tasks
from src.generators.shift_transfer import ShiftTransferConfig, generate_shift_transfer_tasks
from src.generators.social_miniworlds import SocialMiniworldConfig, generate_social_miniworld_tasks
from src.utils.io_utils import load_json, save_json


def _sample_tasks():
    return (
        generate_hidden_rule_tasks(HiddenRuleConfig(count=1, seed=81))
        + generate_shift_transfer_tasks(ShiftTransferConfig(count=1, seed=83))
        + generate_metacog_revision_tasks(MetacogRevisionConfig(count=1, seed=89))
        + generate_attention_distractor_tasks(AttentionDistractorConfig(count=1, seed=97))
        + generate_social_miniworld_tasks(SocialMiniworldConfig(count=1, seed=101))
    )


def test_category_assignment_catches_attention_and_dynamic_gap():
    case = {
        "source": "interactive",
        "task_id": "attention_distractors_0001",
        "family": "attention_distractors",
        "severity": 1.0,
        "failure_reasons": ["final_answer_incorrect", "failed_attention_recovery"],
        "response": {
            "revised_confidence": 0.82,
            "evidence_acknowledged": True,
            "revision_events": [],
            "initial_answer": [1, 2],
            "revised_answer": [1, 2],
        },
    }
    static_result = {"correct": True}
    interactive_result = {"correct": False}

    categories = assign_failure_categories(
        case,
        static_result=static_result,
        interactive_result=interactive_result,
    )

    assert "static_dynamic_gap" in categories
    assert "overconfident_error" in categories
    assert "failed_hypothesis_update" in categories
    assert "poor_attention_recovery" in categories


def test_ranking_logic_rewards_insightful_failures():
    low_case = {"task_id": "a", "family": "hidden_rule", "source": "static", "severity": 0.5, "failure_reasons": []}
    high_case = {
        "task_id": "b",
        "family": "metacog_revision",
        "source": "interactive",
        "severity": 1.0,
        "failure_reasons": ["missed_contradiction"],
        "response": {
            "revised_confidence": 0.9,
            "turns": [{}, {}, {}],
            "initial_answer": [1],
            "revised_answer": [1],
        },
    }

    low_rank = rank_failure_case(low_case, categories=[])
    high_rank = rank_failure_case(
        high_case,
        categories=["overconfident_error", "failed_hypothesis_update"],
        static_result={"correct": True},
        interactive_result={"correct": False},
    )

    assert high_rank["interestingness_score"] > low_rank["interestingness_score"]


def test_single_run_artifact_generation(tmp_path: Path):
    tasks_path = tmp_path / "tasks.json"
    save_json(tasks_path, _sample_tasks())
    run_dir = tmp_path / "shallow_run"
    run_model_evaluation(
        tasks_path=tasks_path,
        adapter=build_adapter("mock-shallow", seed=11),
        run_dir=run_dir,
        include_interactive=True,
        resume=True,
    )

    summary = distill_run_failures(run_dir)
    assert (run_dir / "distilled_failures.json").exists()
    assert (run_dir / "signature_weaknesses.md").exists()
    assert summary["num_failures"] > 0
    assert len(summary["top_overall_failures"]) <= 3

    saved = load_json(run_dir / "distilled_failures.json")
    assert "signature_weaknesses" in saved


def test_multi_run_weakness_comparison_artifact_generation(tmp_path: Path):
    tasks_path = tmp_path / "tasks.json"
    save_json(tasks_path, _sample_tasks())
    noisy_run = tmp_path / "noisy_run"
    shallow_run = tmp_path / "shallow_run"

    run_model_evaluation(
        tasks_path=tasks_path,
        adapter=build_adapter("mock-noisy", seed=11),
        run_dir=noisy_run,
        include_interactive=True,
        resume=True,
    )
    run_model_evaluation(
        tasks_path=tasks_path,
        adapter=build_adapter("mock-shallow", seed=11),
        run_dir=shallow_run,
        include_interactive=True,
        resume=True,
    )

    output_dir = tmp_path / "comparisons" / "noisy_vs_shallow"
    summary = compare_distilled_failures([noisy_run, shallow_run], output_dir)

    assert (output_dir / "weakness_comparison.json").exists()
    assert (output_dir / "weakness_highlights.md").exists()
    assert len(summary["most_separating_weaknesses"]) > 0
