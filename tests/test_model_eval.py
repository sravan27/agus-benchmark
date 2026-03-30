from pathlib import Path

from src.eval.adapters import MockAdapter, OllamaAdapter
from src.eval.interactive_runner import build_interaction_spec
from src.eval.model_runner import run_model_evaluation, select_evaluation_tasks
from src.generators.attention_distractors import AttentionDistractorConfig, generate_attention_distractor_tasks
from src.generators.hidden_rule import HiddenRuleConfig, generate_hidden_rule_tasks
from src.generators.metacog_revision import MetacogRevisionConfig, generate_metacog_revision_tasks
from src.generators.shift_transfer import ShiftTransferConfig, generate_shift_transfer_tasks
from src.generators.social_miniworlds import SocialMiniworldConfig, generate_social_miniworld_tasks
from src.utils.io_utils import load_json, load_jsonl, save_json


def _sample_tasks():
    return (
        generate_hidden_rule_tasks(HiddenRuleConfig(count=1, seed=11))
        + generate_shift_transfer_tasks(ShiftTransferConfig(count=1, seed=13))
        + generate_metacog_revision_tasks(MetacogRevisionConfig(count=1, seed=17))
        + generate_attention_distractor_tasks(AttentionDistractorConfig(count=1, seed=19))
        + generate_social_miniworld_tasks(SocialMiniworldConfig(count=1, seed=23))
    )


def _sample_tasks_with_multiple_per_family():
    return (
        generate_hidden_rule_tasks(HiddenRuleConfig(count=3, seed=31))
        + generate_shift_transfer_tasks(ShiftTransferConfig(count=3, seed=37))
        + generate_metacog_revision_tasks(MetacogRevisionConfig(count=3, seed=41))
        + generate_attention_distractor_tasks(AttentionDistractorConfig(count=3, seed=43))
        + generate_social_miniworld_tasks(SocialMiniworldConfig(count=3, seed=47))
    )


def test_mock_adapter_is_deterministic_for_static_and_interactive():
    task = _sample_tasks()[3]
    spec = build_interaction_spec(task)
    adapter_a = MockAdapter(profile="noisy", seed=101)
    adapter_b = MockAdapter(profile="noisy", seed=101)

    assert adapter_a.predict_task(task) == adapter_b.predict_task(task)
    assert adapter_a.respond_turn(spec, spec.turns[0], []) == adapter_b.respond_turn(spec, spec.turns[0], [])


def test_ollama_adapter_scaffold_normalizes_payloads():
    task = _sample_tasks()[0]
    spec = build_interaction_spec(task)

    class StubOllama(OllamaAdapter):
        def _request_json(self, prompt: str):
            if "Episode family" in prompt:
                return {
                    "answer": spec.turns[0].accepted_answers[0],
                    "confidence": 0.61,
                    "rule_explanation": "induction_rule",
                    "metadata": {"rule_tag": "induction_rule"},
                }
            return {
                "induction_predictions": task["answer"]["induction_targets"],
                "shift_predictions": task["answer"]["shift_targets"],
            }

    adapter = StubOllama(model="stub-model")
    prediction = adapter.predict_task(task)
    turn_payload = adapter.respond_turn(spec, spec.turns[0], [])

    assert prediction["task_id"] == task["task_id"]
    assert prediction["induction_predictions"] == task["answer"]["induction_targets"]
    assert turn_payload["answer"] == spec.turns[0].accepted_answers[0]
    assert turn_payload["metadata"]["rule_tag"] == "induction_rule"


def test_balanced_sampling_spreads_small_subset_across_families():
    tasks = _sample_tasks_with_multiple_per_family()
    selected, meta = select_evaluation_tasks(tasks, max_tasks=5, balanced=True)

    assert len(selected) == 5
    assert meta["tasks_planned_per_family"] == {
        "attention_distractors": 1,
        "hidden_rule": 1,
        "metacog_revision": 1,
        "shift_transfer": 1,
        "social_miniworlds": 1,
    }


def test_per_family_cap_limits_selection_deterministically():
    tasks = _sample_tasks_with_multiple_per_family()
    selected, meta = select_evaluation_tasks(tasks, balanced=True, per_family_max=2)

    assert len(selected) == 10
    assert meta["tasks_planned_per_family"] == {
        "attention_distractors": 2,
        "hidden_rule": 2,
        "metacog_revision": 2,
        "shift_transfer": 2,
        "social_miniworlds": 2,
    }


def test_run_model_evaluation_writes_artifacts_and_family_composition(tmp_path: Path):
    tasks_path = tmp_path / "tasks.json"
    save_json(tasks_path, _sample_tasks_with_multiple_per_family())
    run_dir = tmp_path / "eval_run"

    summary = run_model_evaluation(
        tasks_path=tasks_path,
        adapter=MockAdapter(profile="shallow", seed=7),
        run_dir=run_dir,
        include_interactive=True,
        resume=True,
        balanced=True,
        max_tasks=5,
    )

    assert (run_dir / "config.json").exists()
    assert (run_dir / "predictions.jsonl").exists()
    assert (run_dir / "interactive_sessions.jsonl").exists()
    assert (run_dir / "aggregate_summary.json").exists()
    assert (run_dir / "failure_cases.json").exists()
    assert summary["num_tasks_requested"] == 5
    assert summary["failure_count"] > 0
    assert len(summary["static_summary"]["family_accuracy"]) == 5
    assert summary["run_composition"]["tasks_planned_per_family"]["hidden_rule"] == 1


def test_progress_accounting_reports_true_completed_work(tmp_path: Path):
    tasks_path = tmp_path / "tasks.json"
    save_json(tasks_path, _sample_tasks())
    run_dir = tmp_path / "progress_run"

    run_model_evaluation(
        tasks_path=tasks_path,
        adapter=MockAdapter(profile="oracle", seed=9),
        run_dir=run_dir,
        include_interactive=True,
        resume=True,
    )

    progress = load_json(run_dir / "progress.json")
    assert progress["total_tasks_planned"] == 5
    assert progress["tasks_completed"] == 5
    assert progress["interactive_sessions_planned"] == 5
    assert progress["interactive_sessions_completed"] == 5
    assert progress["work_items_completed"] == progress["work_items_planned"]
    assert progress["percentage_complete"] == 100.0


def test_resumed_run_progress_uses_real_completed_state(tmp_path: Path):
    tasks = _sample_tasks()[:3]
    tasks_path = tmp_path / "tasks.json"
    save_json(tasks_path, tasks)
    run_dir = tmp_path / "resume_run"
    adapter = MockAdapter(profile="oracle", seed=9)

    run_model_evaluation(
        tasks_path=tasks_path,
        adapter=adapter,
        run_dir=run_dir,
        include_interactive=True,
        resume=True,
        max_tasks=2,
    )
    partial_progress = load_json(run_dir / "progress.json")
    assert partial_progress["tasks_completed"] == 2
    assert partial_progress["interactive_sessions_completed"] == 2

    run_model_evaluation(
        tasks_path=tasks_path,
        adapter=adapter,
        run_dir=run_dir,
        include_interactive=True,
        resume=True,
    )

    progress = load_json(run_dir / "progress.json")
    assert len(load_jsonl(run_dir / "predictions.jsonl")) == len(tasks)
    assert len(load_jsonl(run_dir / "interactive_sessions.jsonl")) == len(tasks)
    assert progress["tasks_completed"] == len(tasks)
    assert progress["interactive_sessions_completed"] == len(tasks)
    assert progress["run_complete"] is True


def test_failure_extraction_surfaces_dynamic_errors(tmp_path: Path):
    tasks_path = tmp_path / "tasks.json"
    save_json(tasks_path, _sample_tasks())
    run_dir = tmp_path / "failure_run"

    run_model_evaluation(
        tasks_path=tasks_path,
        adapter=MockAdapter(profile="shallow", seed=3),
        run_dir=run_dir,
        include_interactive=True,
        resume=True,
    )

    failures = load_json(run_dir / "failure_cases.json")
    reasons = {reason for row in failures for reason in row["failure_reasons"]}
    assert "followed_distractor" in reasons or "failed_attention_recovery" in reasons
