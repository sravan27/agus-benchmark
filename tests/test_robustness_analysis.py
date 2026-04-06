from pathlib import Path

from src.eval.adapters import MockAdapter
from src.eval.model_runner import run_model_evaluation
from src.eval.robustness_analysis import compare_robustness_runs
from src.generators.attention_distractors import AttentionDistractorConfig, generate_attention_distractor_tasks
from src.generators.hidden_rule import HiddenRuleConfig, generate_hidden_rule_tasks
from src.generators.metacog_revision import MetacogRevisionConfig, generate_metacog_revision_tasks
from src.generators.shift_transfer import ShiftTransferConfig, generate_shift_transfer_tasks
from src.generators.social_miniworlds import SocialMiniworldConfig, generate_social_miniworld_tasks
from src.utils.io_utils import save_json


def _sample_tasks_many():
    return (
        generate_hidden_rule_tasks(HiddenRuleConfig(count=8, seed=101))
        + generate_shift_transfer_tasks(ShiftTransferConfig(count=8, seed=103))
        + generate_metacog_revision_tasks(MetacogRevisionConfig(count=8, seed=107))
        + generate_attention_distractor_tasks(AttentionDistractorConfig(count=8, seed=109))
        + generate_social_miniworld_tasks(SocialMiniworldConfig(count=8, seed=113))
    )


def test_multi_slice_robustness_artifact_generation(tmp_path: Path):
    tasks_path = tmp_path / "tasks.json"
    save_json(tasks_path, _sample_tasks_many())

    run_dirs = []
    for profile in ("oracle", "shallow"):
        for slice_name in ("original", "replication", "replication_2"):
            run_dir = tmp_path / f"{profile}_{slice_name}"
            run_model_evaluation(
                tasks_path=tasks_path,
                adapter=MockAdapter(profile=profile, seed=11),
                run_dir=run_dir,
                include_interactive=True,
                resume=True,
                balanced=True,
                max_tasks=10,
                balanced_slice=slice_name,
            )
            run_dirs.append(run_dir)

    output_dir = tmp_path / "comparisons" / "robustness_multi_slice"
    summary = compare_robustness_runs(run_dirs, output_dir)

    assert (output_dir / "robustness_summary.json").exists()
    assert (output_dir / "robustness_summary.md").exists()
    assert summary["slice_names"] == ["original", "replication", "replication_2"]
    assert summary["replication_counts"]["core_pattern"]["total"] == 2
    assert summary["replication_counts"]["static_accuracy_ranking"]["held"] >= 1
    assert "overconfident_error_proxy" in summary["weakness_proxy_replication"]
