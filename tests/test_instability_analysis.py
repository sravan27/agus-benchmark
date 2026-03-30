from pathlib import Path

from src.eval.adapters import build_adapter
from src.eval.instability_analysis import (
    analyze_run_instability,
    compare_run_instability,
    compute_session_instability,
)
from src.eval.model_runner import run_model_evaluation
from src.generators.attention_distractors import AttentionDistractorConfig, generate_attention_distractor_tasks
from src.generators.hidden_rule import HiddenRuleConfig, generate_hidden_rule_tasks
from src.generators.metacog_revision import MetacogRevisionConfig, generate_metacog_revision_tasks
from src.generators.shift_transfer import ShiftTransferConfig, generate_shift_transfer_tasks
from src.generators.social_miniworlds import SocialMiniworldConfig, generate_social_miniworld_tasks
from src.utils.io_utils import load_json, load_jsonl, save_json


def _sample_tasks():
    return (
        generate_hidden_rule_tasks(HiddenRuleConfig(count=1, seed=111))
        + generate_shift_transfer_tasks(ShiftTransferConfig(count=1, seed=113))
        + generate_metacog_revision_tasks(MetacogRevisionConfig(count=1, seed=127))
        + generate_attention_distractor_tasks(AttentionDistractorConfig(count=1, seed=131))
        + generate_social_miniworld_tasks(SocialMiniworldConfig(count=1, seed=137))
    )


def test_instability_metric_computation_uses_existing_trace_fields(tmp_path: Path):
    tasks_path = tmp_path / "tasks.json"
    save_json(tasks_path, _sample_tasks())
    run_dir = tmp_path / "shallow_run"
    run_model_evaluation(
        tasks_path=tasks_path,
        adapter=build_adapter("mock-shallow", seed=19),
        run_dir=run_dir,
        include_interactive=True,
        resume=True,
    )
    session = load_jsonl(run_dir / "interactive_sessions.jsonl")[0]
    metrics = compute_session_instability(session)

    assert "trajectory_instability_index" in metrics
    assert metrics["confidence_volatility"] >= 0.0


def test_per_run_instability_artifact_generation(tmp_path: Path):
    tasks_path = tmp_path / "tasks.json"
    save_json(tasks_path, _sample_tasks())
    run_dir = tmp_path / "noisy_run"
    run_model_evaluation(
        tasks_path=tasks_path,
        adapter=build_adapter("mock-noisy", seed=19),
        run_dir=run_dir,
        include_interactive=True,
        resume=True,
    )

    summary = analyze_run_instability(run_dir)
    assert (run_dir / "instability_summary.json").exists()
    assert (run_dir / "instability_highlights.md").exists()
    assert "trajectory_instability_index" in summary


def test_cross_run_instability_comparison(tmp_path: Path):
    tasks_path = tmp_path / "tasks.json"
    save_json(tasks_path, _sample_tasks())
    noisy_run = tmp_path / "noisy_run"
    shallow_run = tmp_path / "shallow_run"
    run_model_evaluation(
        tasks_path=tasks_path,
        adapter=build_adapter("mock-noisy", seed=19),
        run_dir=noisy_run,
        include_interactive=True,
        resume=True,
    )
    run_model_evaluation(
        tasks_path=tasks_path,
        adapter=build_adapter("mock-shallow", seed=19),
        run_dir=shallow_run,
        include_interactive=True,
        resume=True,
    )

    output_dir = tmp_path / "comparisons" / "instability"
    summary = compare_run_instability([noisy_run, shallow_run], output_dir)

    assert (output_dir / "instability_comparison.json").exists()
    assert (output_dir / "instability_insights.md").exists()
    assert summary["most_brittle_model"] in {"noisy_run", "shallow_run"}
