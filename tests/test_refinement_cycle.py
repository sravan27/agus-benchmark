from copy import deepcopy

from src.cli.run_refinement_cycle import run_refinement_cycle
from src.curation.adversarial_curation import curate_tasks, score_task_for_curation
from src.curation.refinement_analysis import summarize_refinement_opportunities
from src.generators.attention_distractors import AttentionDistractorConfig, generate_attention_distractor_tasks
from src.generators.hidden_rule import HiddenRuleConfig, generate_hidden_rule_tasks
from src.generators.metacog_revision import MetacogRevisionConfig, generate_metacog_revision_tasks
from src.generators.shift_transfer import ShiftTransferConfig, generate_shift_transfer_tasks
from src.generators.social_miniworlds import SocialMiniworldConfig, generate_social_miniworld_tasks
from src.utils.io_utils import save_json


def _duplicated_legacy_attention_tasks(count: int = 6) -> list[dict]:
    base = generate_attention_distractor_tasks(
        AttentionDistractorConfig(count=1, seed=51, anti_template_strength=0, distractor_diversity_level=0, cue_delay_level=1)
    )[0]
    rows = []
    for idx in range(count):
        task = deepcopy(base)
        task["task_id"] = f"attention_distractors_{idx:04d}"
        rows.append(task)
    return rows


def _shallow_shift_tasks(count: int = 4) -> list[dict]:
    base = generate_shift_transfer_tasks(
        ShiftTransferConfig(count=1, seed=23, anti_template_strength=0, remap_composition_depth=1)
    )[0]
    rows = []
    for idx in range(count):
        task = deepcopy(base)
        task["task_id"] = f"shift_transfer_{idx:04d}"
        for example in task["examples"]:
            example["output"] = list(example["input"])
        task["answer"]["source_target"] = list(task["query"]["source_query"]["input"])
        task["metadata"]["transfer_vocab"] = list(task["metadata"]["source_vocab"])
        task["context"]["transfer_representation"] = task["context"]["source_representation"]
        task["query"]["transfer_query"]["input"] = list(task["query"]["source_query"]["input"])
        task["answer"]["transfer_target"] = list(task["answer"]["source_target"])
        task["metadata"]["internal_rule"]["name"] = "identity_anchor"
        rows.append(task)
    return rows


def test_refinement_analysis_reports_family_specific_failure_modes():
    tasks = _duplicated_legacy_attention_tasks(count=4) + _shallow_shift_tasks(count=3)
    curated = curate_tasks(tasks)
    summary = summarize_refinement_opportunities(curated["curation_report"], curated["rejected_tasks"])

    assert "family_summaries" in summary
    assert "attention_distractors" in summary
    assert "shift_transfer" in summary
    assert summary["attention_distractors"]["top_failure_modes"]
    assert summary["shift_transfer"]["top_failure_modes"]


def test_hardened_attention_task_outperforms_previously_rejected_template():
    legacy_tasks = _duplicated_legacy_attention_tasks(count=6)
    legacy_results = curate_tasks(legacy_tasks)
    rejected_scores = [row["curation"]["scores"]["benchmark_signal_score"] for row in legacy_results["rejected_tasks"]]
    refined_task = generate_attention_distractor_tasks(
        AttentionDistractorConfig(count=1, seed=51, anti_template_strength=2, distractor_diversity_level=3, cue_delay_level=2)
    )[0]
    refined_score = score_task_for_curation(refined_task)

    assert legacy_results["rejected_tasks"]
    assert refined_score["decision"] == "keep"
    assert refined_score["scores"]["benchmark_signal_score"] > max(rejected_scores)


def test_hardened_shift_transfer_increases_transfer_depth():
    shallow_task = _shallow_shift_tasks(count=1)[0]
    shallow_score = score_task_for_curation(shallow_task)
    refined_task = generate_shift_transfer_tasks(
        ShiftTransferConfig(count=1, seed=23, anti_template_strength=2, remap_composition_depth=3)
    )[0]
    refined_score = score_task_for_curation(refined_task)

    assert shallow_score["decision"] == "reject"
    assert refined_score["decision"] == "keep"
    assert refined_score["scores"]["transfer_depth_score"] > shallow_score["scores"]["transfer_depth_score"]
    assert refined_score["scores"]["benchmark_signal_score"] > shallow_score["scores"]["benchmark_signal_score"]


def test_refinement_cycle_writes_comparison_artifacts(tmp_path):
    tasks = []
    tasks.extend(generate_hidden_rule_tasks(HiddenRuleConfig(count=2, seed=11)))
    tasks.extend(_shallow_shift_tasks(count=4))
    tasks.extend(generate_metacog_revision_tasks(MetacogRevisionConfig(count=2, seed=37)))
    tasks.extend(_duplicated_legacy_attention_tasks(count=4))
    tasks.extend(generate_social_miniworld_tasks(SocialMiniworldConfig(count=2, seed=67)))

    tasks_path = tmp_path / "pre_tasks.json"
    report_path = tmp_path / "pre_curation_report.json"
    rejected_path = tmp_path / "pre_rejected_tasks.json"
    output_dir = tmp_path / "refinement"

    save_json(tasks_path, tasks)
    pre_results = curate_tasks(tasks)
    save_json(report_path, pre_results["curation_report"])
    save_json(rejected_path, pre_results["rejected_tasks"])

    results = run_refinement_cycle(
        project_root=tmp_path,
        tasks_path=tasks_path,
        curation_report_path=report_path,
        rejected_tasks_path=rejected_path,
        output_dir=output_dir,
        count_per_family=4,
    )

    assert (output_dir / "refinement_summary.json").exists()
    assert (output_dir / "pre_post_curation_comparison.json").exists()
    assert (output_dir / "refined_curation_report.json").exists()
    comparison = results["comparison"]
    assert comparison["by_family"]["attention_distractors"]["retention_rate_delta"] > 0
    assert comparison["by_family"]["shift_transfer"]["avg_benchmark_signal_delta"] > 0
