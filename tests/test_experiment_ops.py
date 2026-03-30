from pathlib import Path

from src.eval.adapters import OllamaAdapter, build_adapter
from src.eval.model_runner import run_model_evaluation
from src.eval.run_comparison import compare_evaluation_runs
from src.eval.run_profiles import resolve_run_profile
from src.generators.attention_distractors import AttentionDistractorConfig, generate_attention_distractor_tasks
from src.generators.hidden_rule import HiddenRuleConfig, generate_hidden_rule_tasks
from src.generators.metacog_revision import MetacogRevisionConfig, generate_metacog_revision_tasks
from src.generators.shift_transfer import ShiftTransferConfig, generate_shift_transfer_tasks
from src.generators.social_miniworlds import SocialMiniworldConfig, generate_social_miniworld_tasks
from src.utils.io_utils import load_json, save_json


def _sample_tasks():
    return (
        generate_hidden_rule_tasks(HiddenRuleConfig(count=1, seed=61))
        + generate_shift_transfer_tasks(ShiftTransferConfig(count=1, seed=67))
        + generate_metacog_revision_tasks(MetacogRevisionConfig(count=1, seed=71))
        + generate_attention_distractor_tasks(AttentionDistractorConfig(count=1, seed=73))
        + generate_social_miniworld_tasks(SocialMiniworldConfig(count=1, seed=79))
    )


def test_keep_alive_propagates_into_ollama_payload():
    adapter = OllamaAdapter(model="llama3.1:8b", keep_alive="45m")
    payload = adapter._build_request_payload("demo prompt")

    assert payload["keep_alive"] == "45m"

    built = build_adapter("ollama", model="llama3.1:8b", keep_alive="20m")
    assert built.describe()["keep_alive"] == "20m"


def test_run_profile_expansion_applies_defaults_and_overrides():
    profile = resolve_run_profile("balanced25")
    assert profile["max_tasks"] == 25
    assert profile["balanced"] is True
    assert profile["per_family_max"] == 5

    overridden = resolve_run_profile("balanced25", max_tasks=12, per_family_max=3)
    assert overridden["max_tasks"] == 12
    assert overridden["per_family_max"] == 3
    assert overridden["balanced"] is True


def test_comparison_artifact_generation(tmp_path: Path):
    tasks_path = tmp_path / "tasks.json"
    save_json(tasks_path, _sample_tasks())

    oracle_run = tmp_path / "oracle_run"
    shallow_run = tmp_path / "shallow_run"
    run_model_evaluation(
        tasks_path=tasks_path,
        adapter=build_adapter("mock-oracle", seed=5),
        run_dir=oracle_run,
        include_interactive=True,
        resume=True,
    )
    run_model_evaluation(
        tasks_path=tasks_path,
        adapter=build_adapter("mock-shallow", seed=5),
        run_dir=shallow_run,
        include_interactive=True,
        resume=True,
    )

    output_dir = tmp_path / "comparisons" / "oracle_vs_shallow"
    summary = compare_evaluation_runs([oracle_run, shallow_run], output_dir)

    assert (output_dir / "comparison_summary.json").exists()
    assert (output_dir / "comparison_table.md").exists()
    assert (output_dir / "top_insights.md").exists()
    assert [row["run_name"] for row in summary["runs"]] == ["oracle_run", "shallow_run"]

    markdown = (output_dir / "comparison_table.md").read_text(encoding="utf-8")
    assert "oracle_run" in markdown
    assert "shallow_run" in markdown

    saved_summary = load_json(output_dir / "comparison_summary.json")
    assert "top_failure_differences" in saved_summary
