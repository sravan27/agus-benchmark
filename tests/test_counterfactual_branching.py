from pathlib import Path

from src.eval.adapters import build_adapter
from src.eval.counterfactual_branching import (
    CounterfactualProgressTracker,
    build_counterfactual_bundle,
    evaluate_counterfactual_bundles,
    generate_counterfactual_bundles,
    save_counterfactual_bundles,
    write_counterfactual_artifacts,
)
from src.generators.hidden_rule import HiddenRuleConfig, generate_hidden_rule_tasks
from src.generators.shift_transfer import ShiftTransferConfig, generate_shift_transfer_tasks
from src.generators.social_miniworlds import SocialMiniworldConfig, generate_social_miniworld_tasks
from src.utils.io_utils import load_json


def test_hidden_rule_branching_creates_minimal_counterfactual_continuations():
    task = generate_hidden_rule_tasks(HiddenRuleConfig(count=1, seed=13))[0]
    bundle = build_counterfactual_bundle(task)

    assert bundle.family == "hidden_rule"
    assert len(bundle.branches) == 2
    assert bundle.branches[0].label == "confirming_evidence"
    assert bundle.branches[1].label == "contradicting_evidence"
    assert bundle.branches[0].spec.expected["accepted_initial_targets"] == bundle.branches[1].spec.expected["accepted_initial_targets"]
    assert (
        bundle.branches[0].spec.expected["accepted_revised_targets"]
        != bundle.branches[1].spec.expected["accepted_revised_targets"]
    )


def test_shift_transfer_branches_preserve_rule_but_change_surface_targets():
    task = generate_shift_transfer_tasks(ShiftTransferConfig(count=1, seed=17))[0]
    bundle = build_counterfactual_bundle(task)

    assert bundle.family == "shift_transfer"
    assert len(bundle.branches) == 2
    revised_rules = {branch.spec.expected["canonical_revised_rule"] for branch in bundle.branches}
    assert len(revised_rules) == 1
    assert bundle.branches[0].spec.expected["accepted_initial_targets"] == bundle.branches[1].spec.expected["accepted_initial_targets"]
    assert (
        bundle.branches[0].spec.expected["accepted_revised_targets"]
        != bundle.branches[1].spec.expected["accepted_revised_targets"]
    )


def test_social_branching_isolates_private_information_variable():
    task = generate_social_miniworld_tasks(SocialMiniworldConfig(count=1, seed=23))[0]
    bundle = build_counterfactual_bundle(task)
    private_branch, public_branch = bundle.branches

    private_final = private_branch.spec.expected["accepted_revised_targets"][0]
    public_final = public_branch.spec.expected["accepted_revised_targets"][0]

    assert bundle.family == "social_miniworlds"
    assert private_final["actual_location"] == public_final["actual_location"]
    assert private_final["trusted_agent"] == public_final["trusted_agent"]
    assert private_final["belief_of_false_belief_agent"] != public_final["belief_of_false_belief_agent"]


def test_counterfactual_metrics_score_oracle_branches_cleanly(tmp_path: Path):
    tasks = (
        generate_hidden_rule_tasks(HiddenRuleConfig(count=1, seed=29))
        + generate_shift_transfer_tasks(ShiftTransferConfig(count=1, seed=31))
        + generate_social_miniworld_tasks(SocialMiniworldConfig(count=1, seed=37))
    )
    bundles = generate_counterfactual_bundles(tasks, max_per_family=1)
    adapter = build_adapter("mock-oracle", seed=5)

    summary = evaluate_counterfactual_bundles(
        bundles,
        adapter.respond_turn,
        run_name="oracle_counterfactual_test",
        adapter_description=adapter.describe(),
    )

    assert summary["num_bundles"] == 3
    assert summary["num_branches"] == 6
    for metric in summary["overall_metrics"].values():
        assert metric == 1.0

    branches_path = tmp_path / "branches.json"
    save_counterfactual_bundles(branches_path, bundles)
    assert branches_path.exists()
    assert len(load_json(branches_path)) == 3


def test_counterfactual_artifact_generation(tmp_path: Path):
    tasks = generate_hidden_rule_tasks(HiddenRuleConfig(count=1, seed=41))
    bundles = generate_counterfactual_bundles(tasks, max_per_family=1)
    adapter = build_adapter("mock-oracle", seed=7)
    summary = evaluate_counterfactual_bundles(
        bundles,
        adapter.respond_turn,
        run_name="counterfactual_artifact_test",
        adapter_description=adapter.describe(),
    )

    write_counterfactual_artifacts(tmp_path, summary)

    assert (tmp_path / "counterfactual_summary.json").exists()
    assert (tmp_path / "counterfactual_highlights.md").exists()
    highlights = (tmp_path / "counterfactual_highlights.md").read_text(encoding="utf-8")
    assert "Counterfactual Highlights" in highlights
    assert "counterfactual_update_fidelity" in highlights


def test_counterfactual_progress_accounting_and_artifact_generation(tmp_path: Path):
    tasks = generate_hidden_rule_tasks(HiddenRuleConfig(count=1, seed=53))
    bundles = generate_counterfactual_bundles(tasks, max_per_family=1)
    adapter = build_adapter("mock-oracle", seed=11)
    tracker = CounterfactualProgressTracker(
        run_name="counterfactual_progress_test",
        run_dir=tmp_path,
        adapter_description=adapter.describe(),
        families=["hidden_rule"],
        total_bundles=1,
        total_branches=2,
    )
    snapshots: list[tuple[int, int, int, bool]] = []
    tracker.update(bundles_completed=0, branches_completed=0, num_errors=0, run_complete=False)

    def progress_callback(bundles_completed: int, branches_completed: int, num_errors: int, run_complete: bool) -> None:
        snapshots.append((bundles_completed, branches_completed, num_errors, run_complete))
        tracker.update(
            bundles_completed=bundles_completed,
            branches_completed=branches_completed,
            num_errors=num_errors,
            run_complete=run_complete,
        )

    evaluate_counterfactual_bundles(
        bundles,
        adapter.respond_turn,
        run_name="counterfactual_progress_test",
        adapter_description=adapter.describe(),
        progress_callback=progress_callback,
    )

    final_payload = tracker.update(bundles_completed=1, branches_completed=2, num_errors=0, run_complete=True)
    progress_payload = load_json(tmp_path / "counterfactual_progress.json")

    assert snapshots == [(0, 1, 0, False), (0, 2, 0, False), (1, 2, 0, False)]
    assert progress_payload["bundles_completed"] == 1
    assert progress_payload["branches_completed"] == 2
    assert progress_payload["num_errors"] == 0
    assert progress_payload["run_complete"] is True
    assert progress_payload["avg_seconds_per_bundle"] >= 0.0
    assert final_payload["eta_seconds"] == 0.0
