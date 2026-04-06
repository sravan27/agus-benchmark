from pathlib import Path

from src.eval.adapters import MockAdapter
from src.eval.model_runner import run_model_evaluation
from src.eval.validation_bundle import build_validation_bundle
from src.generators.attention_distractors import AttentionDistractorConfig, generate_attention_distractor_tasks
from src.generators.hidden_rule import HiddenRuleConfig, generate_hidden_rule_tasks
from src.generators.metacog_revision import MetacogRevisionConfig, generate_metacog_revision_tasks
from src.generators.shift_transfer import ShiftTransferConfig, generate_shift_transfer_tasks
from src.generators.social_miniworlds import SocialMiniworldConfig, generate_social_miniworld_tasks
from src.utils.io_utils import save_json


def _sample_tasks_many():
    return (
        generate_hidden_rule_tasks(HiddenRuleConfig(count=4, seed=101))
        + generate_shift_transfer_tasks(ShiftTransferConfig(count=4, seed=103))
        + generate_metacog_revision_tasks(MetacogRevisionConfig(count=4, seed=107))
        + generate_attention_distractor_tasks(AttentionDistractorConfig(count=4, seed=109))
        + generate_social_miniworld_tasks(SocialMiniworldConfig(count=4, seed=113))
    )


def test_validation_bundle_artifact_generation(tmp_path: Path):
    tasks_path = tmp_path / "tasks.json"
    save_json(tasks_path, _sample_tasks_many())

    shallow_run = tmp_path / "mock_shallow_balanced_interactive"
    oracle_run = tmp_path / "mock_oracle_balanced_interactive"
    noisy_run = tmp_path / "mock_noisy_balanced_interactive"
    for run_dir, profile in (
        (shallow_run, "shallow"),
        (oracle_run, "oracle"),
        (noisy_run, "noisy"),
    ):
        run_model_evaluation(
            tasks_path=tasks_path,
            adapter=MockAdapter(profile=profile, seed=17),
            run_dir=run_dir,
            include_interactive=True,
            resume=True,
            balanced=True,
            max_tasks=10,
        )

    curation_comparison = {
        "overall": {"pre_kept": 10, "post_kept": 14, "kept_delta": 4, "pre_rejected": 5, "post_rejected": 2, "rejected_delta": -3},
        "by_family": {
            "attention_distractors": {"avg_benchmark_signal_delta": 0.21, "avg_trajectory_value_delta": 0.4},
            "shift_transfer": {"avg_benchmark_signal_delta": 0.08, "avg_trajectory_value_delta": 0.2},
        },
    }
    search_conditioned_comparison = {
        "overall": {"pre_kept": 10, "post_kept": 15, "kept_delta": 5, "pre_rejected": 5, "post_rejected": 1, "rejected_delta": -4},
        "by_family": {
            "attention_distractors": {"avg_benchmark_signal_delta": 0.25, "avg_trajectory_value_delta": 0.5},
            "shift_transfer": {"avg_benchmark_signal_delta": 0.09, "avg_trajectory_value_delta": 0.25},
        },
    }
    curation_path = tmp_path / "curation_comparison.json"
    search_path = tmp_path / "search_conditioned_comparison.json"
    save_json(curation_path, curation_comparison)
    save_json(search_path, search_conditioned_comparison)

    output_dir = tmp_path / "comparisons" / "validation_bundle"
    summary = build_validation_bundle(
        curation_comparison_path=curation_path,
        search_conditioned_comparison_path=search_path,
        shallow_run_dir=shallow_run,
        adaptive_run_dirs=[oracle_run, noisy_run],
        rank_shift_run_dirs=[shallow_run, oracle_run, noisy_run],
        output_dir=output_dir,
    )

    assert (output_dir / "validation_summary.json").exists()
    assert (output_dir / "validation_bundle.md").exists()
    assert summary["curation_effect"]["manual_refinement"]["overall"]["kept_delta"] == 4
    assert "static_accuracy_ranking" in summary["rank_shift"]
    assert len(summary["shallow_vs_adaptive"]["adaptive_runs"]) == 2
