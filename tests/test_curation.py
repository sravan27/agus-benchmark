from copy import deepcopy

from src.curation.adversarial_curation import (
    CurationPolicy,
    _baseline_registry,
    _decision_for_scores,
    curate_tasks,
    score_task_for_curation,
)
from src.eval.interactive_runner import run_interactive_session
from src.generators.attention_distractors import AttentionDistractorConfig, generate_attention_distractor_tasks
from src.generators.hidden_rule import HiddenRuleConfig, generate_hidden_rule_tasks
from src.generators.metacog_revision import MetacogRevisionConfig, generate_metacog_revision_tasks
from src.generators.shift_transfer import ShiftTransferConfig, generate_shift_transfer_tasks
from src.generators.social_miniworlds import SocialMiniworldConfig, generate_social_miniworld_tasks


def _make_shallow_shift_task() -> dict:
    task = deepcopy(generate_shift_transfer_tasks(ShiftTransferConfig(count=1, seed=23))[0])
    task["task_id"] = "shift_transfer_weak_0001"
    for example in task["examples"]:
        example["output"] = list(example["input"])
    task["answer"]["source_target"] = list(task["query"]["source_query"]["input"])
    task["metadata"]["transfer_vocab"] = list(task["metadata"]["source_vocab"])
    task["context"]["transfer_representation"] = task["context"]["source_representation"]
    task["query"]["transfer_query"]["input"] = list(task["query"]["source_query"]["input"])
    task["answer"]["transfer_target"] = list(task["answer"]["source_target"])
    task["metadata"]["internal_rule"]["name"] = "identity_anchor"
    return task


def _make_shallow_metacog_task() -> dict:
    task = deepcopy(generate_metacog_revision_tasks(MetacogRevisionConfig(count=1, seed=37))[0])
    task["task_id"] = "metacog_revision_weak_0000"
    chosen = list(task["answer"]["acceptable_initial_targets"][0])
    task["answer"]["revised_target"] = chosen
    task["metadata"]["actual_rule"] = dict(task["metadata"]["candidate_rules"][0])
    task["query"]["revision_prompt"]["corrective_example"]["output"] = chosen
    for example in task["examples"]:
        if example["phase"] == "corrective_evidence":
            example["output"] = chosen
    return task


def _make_shallow_attention_task() -> dict:
    task = deepcopy(generate_attention_distractor_tasks(AttentionDistractorConfig(count=1, seed=51))[0])
    task["task_id"] = "attention_distractors_weak_0000"
    for example in task["examples"]:
        example["distractor_sequence"] = list(example["signal_sequence"])
    task["query"]["record"]["distractor_sequence"] = list(task["query"]["record"]["signal_sequence"])
    return task


def _make_duplicate_attention_tasks(count: int = 4) -> list[dict]:
    task = generate_attention_distractor_tasks(AttentionDistractorConfig(count=1, seed=51))[0]
    rows = []
    for idx in range(count):
        row = deepcopy(task)
        row["task_id"] = f"attention_distractors_dup_{idx:04d}"
        rows.append(row)
    return rows


def test_representation_anchor_solver_exposes_transfer_shortcut():
    task = generate_shift_transfer_tasks(ShiftTransferConfig(count=1, seed=23))[0]
    responder = _baseline_registry()["representation_anchor_solver"]
    session = run_interactive_session(task, responder)

    assert session["response"]["revised_answer"] == session["response"]["initial_answer"]
    assert session["derived"]["revised_correct"] is False


def test_distractor_vulnerable_solver_gets_captured_before_cue():
    task = generate_attention_distractor_tasks(AttentionDistractorConfig(count=1, seed=51))[0]
    responder = _baseline_registry()["distractor_vulnerable_solver"]
    session = run_interactive_session(task, responder)

    assert session["turns"][0]["model_response"]["attended_signals"] == ["distractor_sequence"]
    assert 1 in session["derived"]["capture_turns"]
    assert session["derived"]["revised_correct"] is False


def test_trust_naive_solver_keeps_uniform_social_trust():
    task = generate_social_miniworld_tasks(SocialMiniworldConfig(count=1, seed=67))[0]
    responder = _baseline_registry()["trust_naive_solver"]
    session = run_interactive_session(task, responder)

    trust_scores = session["response"]["trust_scores_by_agent"]
    assert len(set(trust_scores.values())) == 1
    assert session["derived"]["revised_correct"] is False


def test_curation_decision_logic_keeps_strong_tasks_and_rejects_easy_ones():
    policy = CurationPolicy()

    keep_decision, keep_reasons = _decision_for_scores(
        {
            "baseline_solve_rate": 0.1,
            "shortcut_vulnerability_score": 0.2,
            "revision_discrimination_score": 0.8,
            "distractor_discrimination_score": None,
            "social_reasoning_discrimination_score": None,
            "transfer_depth_score": 0.7,
            "trajectory_value_score": 0.75,
            "template_novelty_score": 0.9,
            "benchmark_signal_score": 0.8,
        },
        policy,
    )
    reject_decision, reject_reasons = _decision_for_scores(
        {
            "baseline_solve_rate": 0.75,
            "shortcut_vulnerability_score": 0.7,
            "revision_discrimination_score": 0.1,
            "distractor_discrimination_score": 0.1,
            "social_reasoning_discrimination_score": None,
            "transfer_depth_score": None,
            "trajectory_value_score": 0.2,
            "template_novelty_score": 0.1,
            "benchmark_signal_score": 0.3,
        },
        policy,
    )

    assert keep_decision == "keep"
    assert keep_reasons == []
    assert reject_decision == "reject"
    assert "too_easy" in reject_reasons
    assert "shortcut_solvable" in reject_reasons


def test_curation_report_summarizes_retention_and_rejections():
    tasks = []
    tasks.extend(generate_hidden_rule_tasks(HiddenRuleConfig(count=1, seed=11)))
    tasks.extend(generate_shift_transfer_tasks(ShiftTransferConfig(count=1, seed=23)))
    tasks.extend(generate_metacog_revision_tasks(MetacogRevisionConfig(count=1, seed=37)))
    tasks.extend(generate_attention_distractor_tasks(AttentionDistractorConfig(count=1, seed=51)))
    tasks.extend(generate_social_miniworld_tasks(SocialMiniworldConfig(count=1, seed=67)))
    tasks.append(_make_shallow_shift_task())

    results = curate_tasks(tasks)
    report = results["curation_report"]

    assert report["total_tasks_processed"] == 6
    assert report["kept_count"] + report["review_count"] + report["rejected_count"] == 6
    assert set(report["per_family_retention_rates"]) == {
        "hidden_rule",
        "shift_transfer",
        "metacog_revision",
        "attention_distractors",
        "social_miniworlds",
    }
    assert report["rejected_count"] >= 1
    assert report["most_common_rejection_reasons"]


def test_curation_rejects_deliberately_shallow_tasks():
    shallow_tasks = [_make_shallow_shift_task(), _make_shallow_metacog_task()]

    for task in shallow_tasks:
        scored = score_task_for_curation(task)
        assert scored["decision"] == "reject"
        assert scored["reasons"]

    duplicate_attention = curate_tasks(_make_duplicate_attention_tasks())
    assert duplicate_attention["curation_report"]["rejected_count"] >= 1
