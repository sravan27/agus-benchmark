from pathlib import Path
import pytest

from src.eval.adapters import MockAdapter
from src.eval.model_runner import run_model_evaluation, select_evaluation_tasks
from src.eval.replication_analysis import compare_replication_runs
from src.generators.attention_distractors import AttentionDistractorConfig, generate_attention_distractor_tasks
from src.generators.hidden_rule import HiddenRuleConfig, generate_hidden_rule_tasks
from src.generators.metacog_revision import MetacogRevisionConfig, generate_metacog_revision_tasks
from src.generators.shift_transfer import ShiftTransferConfig, generate_shift_transfer_tasks
from src.generators.social_miniworlds import SocialMiniworldConfig, generate_social_miniworld_tasks
from src.utils.io_utils import load_json, save_json


def _sample_tasks_many():
    return (
        generate_hidden_rule_tasks(HiddenRuleConfig(count=6, seed=101))
        + generate_shift_transfer_tasks(ShiftTransferConfig(count=6, seed=103))
        + generate_metacog_revision_tasks(MetacogRevisionConfig(count=6, seed=107))
        + generate_attention_distractor_tasks(AttentionDistractorConfig(count=6, seed=109))
        + generate_social_miniworld_tasks(SocialMiniworldConfig(count=6, seed=113))
    )


def test_replication_slice_is_deterministic_and_distinct():
    tasks = _sample_tasks_many()
    original_selected, original_meta = select_evaluation_tasks(
        tasks,
        balanced=True,
        max_tasks=10,
        balanced_slice="original",
    )
    replication_selected, replication_meta = select_evaluation_tasks(
        tasks,
        balanced=True,
        max_tasks=10,
        balanced_slice="replication",
    )
    replication_selected_repeat, _ = select_evaluation_tasks(
        tasks,
        balanced=True,
        max_tasks=10,
        balanced_slice="replication",
    )

    assert original_meta["balanced_slice_name"] == "original"
    assert replication_meta["balanced_slice_name"] == "replication"
    assert original_meta["tasks_planned_per_family"] == replication_meta["tasks_planned_per_family"]
    assert {task["task_id"] for task in original_selected}.isdisjoint(
        {task["task_id"] for task in replication_selected}
    )
    assert [task["task_id"] for task in replication_selected] == [
        task["task_id"] for task in replication_selected_repeat
    ]


def test_additional_replication_slice_is_deterministic_and_distinct():
    tasks = _sample_tasks_many()
    replication_selected, replication_meta = select_evaluation_tasks(
        tasks,
        balanced=True,
        max_tasks=10,
        balanced_slice="replication",
    )
    replication_2_selected, replication_2_meta = select_evaluation_tasks(
        tasks,
        balanced=True,
        max_tasks=10,
        balanced_slice="replication_2",
    )
    replication_2_selected_repeat, _ = select_evaluation_tasks(
        tasks,
        balanced=True,
        max_tasks=10,
        balanced_slice="replication_2",
    )

    assert replication_meta["tasks_planned_per_family"] == replication_2_meta["tasks_planned_per_family"]
    assert replication_2_meta["balanced_slice_name"] == "replication_2"
    assert {task["task_id"] for task in replication_selected}.isdisjoint(
        {task["task_id"] for task in replication_2_selected}
    )
    assert [task["task_id"] for task in replication_2_selected] == [
        task["task_id"] for task in replication_2_selected_repeat
    ]


def test_run_model_evaluation_records_slice_metadata(tmp_path: Path):
    tasks_path = tmp_path / "tasks.json"
    save_json(tasks_path, _sample_tasks_many())
    run_dir = tmp_path / "replication_run"

    summary = run_model_evaluation(
        tasks_path=tasks_path,
        adapter=MockAdapter(profile="shallow", seed=5),
        run_dir=run_dir,
        include_interactive=True,
        resume=True,
        balanced=True,
        max_tasks=10,
        balanced_slice="replication",
    )

    assert summary["run_composition"]["balanced_slice_name"] == "replication"
    assert summary["static_summary"]["run_composition"]["balanced_slice_name"] == "replication"
    assert summary["interactive_summary"]["run_composition"]["balanced_slice_name"] == "replication"
    progress = load_json(run_dir / "progress.json")
    assert progress["balanced_slice_name"] == "replication"


def test_replication_summary_artifact_generation(tmp_path: Path):
    tasks_path = tmp_path / "tasks.json"
    save_json(tasks_path, _sample_tasks_many())

    original_oracle = tmp_path / "oracle_original"
    original_shallow = tmp_path / "shallow_original"
    replication_oracle = tmp_path / "oracle_replication"
    replication_shallow = tmp_path / "shallow_replication"

    for run_dir, adapter, balanced_slice in (
        (original_oracle, MockAdapter(profile="oracle", seed=11), "original"),
        (original_shallow, MockAdapter(profile="shallow", seed=11), "original"),
        (replication_oracle, MockAdapter(profile="oracle", seed=11), "replication"),
        (replication_shallow, MockAdapter(profile="shallow", seed=11), "replication"),
    ):
        run_model_evaluation(
            tasks_path=tasks_path,
            adapter=adapter,
            run_dir=run_dir,
            include_interactive=True,
            resume=True,
            balanced=True,
            max_tasks=10,
            balanced_slice=balanced_slice,
        )

    output_dir = tmp_path / "comparisons" / "oracle_vs_shallow_replication"
    summary = compare_replication_runs(
        original_run_dirs=[original_oracle, original_shallow],
        replication_run_dirs=[replication_oracle, replication_shallow],
        output_dir=output_dir,
    )

    assert (output_dir / "replication_summary.json").exists()
    assert (output_dir / "replication_summary.md").exists()
    assert summary["original_slice"]["slice_name"] == "original"
    assert summary["replication_slice"]["slice_name"] == "replication"
    assert "static_accuracy_ranking" in summary["ranking_checks"]
    assert "overconfident_error_proxy" in summary["weakness_proxy_checks"]


def test_replication_summary_rejects_all_error_runs(tmp_path: Path):
    def _write_invalid_run(run_dir: Path, *, model: str, slice_name: str):
        run_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            run_dir / "aggregate_summary.json",
            {
                "adapter": {"model": model, "name": f"ollama:{model}"},
                "num_tasks_requested": 10,
                "num_static_predictions": 0,
                "num_interactive_sessions": 0,
                "num_errors": 20,
                "static_summary": {"accuracy": 0.0},
                "interactive_summary": {"belief_trajectory_quality": 0.0},
                "failure_count": 10,
                "run_composition": {
                    "balanced_slice_name": slice_name,
                    "interactive_sessions_planned_per_family": {"hidden_rule": 10},
                },
            },
        )
        save_json(run_dir / "failure_cases.json", [])
        (run_dir / "interactive_sessions.jsonl").write_text("", encoding="utf-8")

    original_run = tmp_path / "original_invalid"
    replication_run = tmp_path / "replication_invalid"
    _write_invalid_run(original_run, model="llama3.1:8b", slice_name="original")
    _write_invalid_run(replication_run, model="llama3.1:8b", slice_name="replication")

    with pytest.raises(ValueError, match="refused invalid runs"):
        compare_replication_runs(
            original_run_dirs=[original_run],
            replication_run_dirs=[replication_run],
            output_dir=tmp_path / "comparisons" / "invalid_replication",
        )
