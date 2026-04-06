from pathlib import Path

from src.eval.adapters import build_adapter
from src.eval.counterfactual_branching import (
    evaluate_counterfactual_bundles,
    generate_counterfactual_bundles,
    write_counterfactual_artifacts,
)
from src.eval.counterfactual_comparison import compare_counterfactual_runs
from src.generators.hidden_rule import HiddenRuleConfig, generate_hidden_rule_tasks
from src.generators.shift_transfer import ShiftTransferConfig, generate_shift_transfer_tasks
from src.generators.social_miniworlds import SocialMiniworldConfig, generate_social_miniworld_tasks


def _sample_counterfactual_tasks():
    return (
        generate_hidden_rule_tasks(HiddenRuleConfig(count=1, seed=13))
        + generate_shift_transfer_tasks(ShiftTransferConfig(count=1, seed=17))
        + generate_social_miniworld_tasks(SocialMiniworldConfig(count=1, seed=23))
    )


def test_counterfactual_comparison_artifact_generation(tmp_path: Path):
    tasks = _sample_counterfactual_tasks()
    bundles = generate_counterfactual_bundles(tasks, max_per_family=1)

    run_dirs = []
    for profile in ("oracle", "shallow"):
        adapter = build_adapter(f"mock-{profile}", seed=5)
        summary = evaluate_counterfactual_bundles(
            bundles,
            adapter.respond_turn,
            run_name=f"{profile}_counterfactual",
            adapter_description=adapter.describe(),
        )
        run_dir = tmp_path / f"{profile}_counterfactual"
        write_counterfactual_artifacts(run_dir, summary)
        run_dirs.append(run_dir)

    output_dir = tmp_path / "comparisons" / "counterfactual_check"
    comparison = compare_counterfactual_runs(run_dirs, output_dir)

    assert (output_dir / "counterfactual_comparison.json").exists()
    assert (output_dir / "counterfactual_comparison.md").exists()
    assert len(comparison["runs"]) == 2
    assert "counterfactual_update_fidelity" in comparison["overall_metric_comparison"]
