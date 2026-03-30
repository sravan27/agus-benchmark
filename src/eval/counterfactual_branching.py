"""Counterfactual branching episodes for AGUS v2."""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import time
from typing import Any, Callable

from src.eval.interactive_runner import EpisodeSpec, EpisodeTurnSpec, build_interaction_spec, run_episode_spec
from src.generators.common import RuleSpec
from src.scoring.metrics import (
    branch_belief_coherence,
    counterfactual_confidence_calibration,
    counterfactual_update_fidelity,
    cross_branch_consistency,
    invariant_preservation_score,
)
from src.utils.io_utils import save_json

SUPPORTED_COUNTERFACTUAL_FAMILIES = {
    "hidden_rule",
    "shift_transfer",
    "social_miniworlds",
}


@dataclass(frozen=True)
class CounterfactualBranch:
    """One alternate continuation from a shared base episode."""

    branch_id: str
    label: str
    changed_factor: str
    expected_revision_kind: str
    spec: EpisodeSpec
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["spec"] = self.spec.to_dict()
        return payload


@dataclass(frozen=True)
class CounterfactualBundle:
    """A small family of nearby alternate futures for one base task."""

    bundle_id: str
    task_id: str
    family: str
    shared_structure: dict[str, Any]
    shared_checks: list[dict[str, Any]]
    variant_path: str
    branches: list[CounterfactualBranch]

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "task_id": self.task_id,
            "family": self.family,
            "shared_structure": self.shared_structure,
            "shared_checks": self.shared_checks,
            "variant_path": self.variant_path,
            "branches": [branch.to_dict() for branch in self.branches],
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CounterfactualProgressTracker:
    """Persist real progress for counterfactual branch evaluation."""

    def __init__(
        self,
        *,
        run_name: str,
        run_dir: Path,
        adapter_description: dict[str, Any] | None,
        families: list[str],
        total_bundles: int,
        total_branches: int,
    ) -> None:
        self.run_name = run_name
        self.run_dir = run_dir
        self.adapter_description = adapter_description or {}
        self.families = families
        self.total_bundles = total_bundles
        self.total_branches = total_branches
        self.started_at = _now_iso()
        self.started_monotonic = time.perf_counter()
        self.payload: dict[str, Any] = {}

    @property
    def path(self) -> Path:
        return self.run_dir / "counterfactual_progress.json"

    def update(
        self,
        *,
        bundles_completed: int,
        branches_completed: int,
        num_errors: int = 0,
        run_complete: bool = False,
    ) -> dict[str, Any]:
        elapsed_seconds = round(time.perf_counter() - self.started_monotonic, 4)
        avg_seconds_per_bundle = (
            round(elapsed_seconds / bundles_completed, 4) if bundles_completed else 0.0
        )
        if run_complete:
            eta_seconds: float | None = 0.0
        elif bundles_completed:
            eta_seconds = round(avg_seconds_per_bundle * max(self.total_bundles - bundles_completed, 0), 4)
        else:
            eta_seconds = None

        self.payload = {
            "run_name": self.run_name,
            "model": self.adapter_description.get("model"),
            "adapter": self.adapter_description,
            "total_bundles": self.total_bundles,
            "bundles_completed": bundles_completed,
            "total_branches": self.total_branches,
            "branches_completed": branches_completed,
            "families": self.families,
            "started_at": self.started_at,
            "last_updated_at": _now_iso(),
            "elapsed_seconds": elapsed_seconds,
            "avg_seconds_per_bundle": avg_seconds_per_bundle,
            "eta_seconds": eta_seconds,
            "num_errors": num_errors,
            "run_complete": run_complete,
        }
        save_json(self.path, self.payload)
        return dict(self.payload)


def _rule_from_payload(payload: dict[str, Any]) -> RuleSpec:
    return RuleSpec(name=payload["name"], params=payload["params"])


def _remap_tokens(tokens: list[str], source_vocab: list[str], target_vocab: list[str]) -> list[str]:
    index_map = {token: idx for idx, token in enumerate(source_vocab)}
    return [target_vocab[index_map[token]] for token in tokens]


def _expected_turn_confidences(spec: EpisodeSpec) -> list[float]:
    return [float(turn.expected_confidence) for turn in spec.turns]


def _find_statement(task: dict[str, Any], agent: str) -> dict[str, Any] | None:
    for example in task["examples"]:
        if example.get("phase") == "statement" and example.get("agent") == agent:
            return example
    return None


def _rotated_vocab(tokens: list[str], steps: int = 2) -> list[str]:
    if not tokens:
        return tokens
    offset = steps % len(tokens)
    if offset == 0:
        offset = 1
    return tokens[offset:] + tokens[:offset]


def _build_hidden_rule_bundle(task: dict[str, Any]) -> CounterfactualBundle:
    base_spec = build_interaction_spec(task)
    induction_rule = _rule_from_payload(task["metadata"]["internal_rules"]["induction"])
    domain_size = len(task["context"]["symbol_space"])
    contradiction_turn = base_spec.turns[1]
    confirming_example = dict(contradiction_turn.event["example"])
    confirming_example["output"] = induction_rule.apply(confirming_example["input"], domain_size)

    confirming_turn = replace(
        contradiction_turn,
        turn_id="turn_2_rule_confirmation",
        event={
            "event_type": "rule_confirmation",
            "message": "A new observation confirms the original rule instead of contradicting it.",
            "example": confirming_example,
        },
        accepted_answers=[base_spec.expected["accepted_initial_targets"][0]],
        expected_confidence=0.78,
        expected_rule=base_spec.expected["canonical_initial_rule"],
    )
    confirming_spec = replace(
        base_spec,
        episode_type="counterfactual_rule_confirmation",
        turns=[base_spec.turns[0], confirming_turn],
        expected={
            **base_spec.expected,
            "accepted_revised_targets": [base_spec.expected["accepted_initial_targets"][0]],
            "contradiction_expected": False,
            "canonical_revised_rule": base_spec.expected["canonical_initial_rule"],
            "expected_revised_confidence": 0.78,
        },
    )
    shift_spec = replace(base_spec, episode_type="counterfactual_rule_shift")

    shared_initial_target = base_spec.expected["accepted_initial_targets"][0]
    bundle_id = f"{task['task_id']}__counterfactual"
    return CounterfactualBundle(
        bundle_id=bundle_id,
        task_id=task["task_id"],
        family=task["family"],
        shared_structure={
            "base_question": "Does the rule remain stable when contradiction is absent, and revise when contradiction appears?",
            "varying_factor": "contradiction_presence",
        },
        shared_checks=[
            {"name": "initial_answer", "path": "response.initial_answer", "expected": shared_initial_target},
            {
                "name": "initial_rule_tag",
                "path": "response.metadata.initial_rule_tag",
                "expected": base_spec.expected["canonical_initial_rule"],
            },
        ],
        variant_path="response.revised_answer",
        branches=[
            CounterfactualBranch(
                branch_id=f"{bundle_id}__confirm",
                label="confirming_evidence",
                changed_factor="contradiction_absent",
                expected_revision_kind="none",
                spec=confirming_spec,
                notes=["The same probe should keep the original rule when the new evidence confirms it."],
            ),
            CounterfactualBranch(
                branch_id=f"{bundle_id}__shift",
                label="contradicting_evidence",
                changed_factor="contradiction_present",
                expected_revision_kind="rule",
                spec=shift_spec,
                notes=["The same probe should force a rule revision once the contradiction appears."],
            ),
        ],
    )


def _build_shift_transfer_bundle(task: dict[str, Any]) -> CounterfactualBundle:
    source_vocab = task["metadata"]["source_vocab"]
    direct_vocab = task["metadata"]["transfer_vocab"]
    alternate_vocab = _rotated_vocab(direct_vocab, steps=2)
    rule_name = task["metadata"]["internal_rule"]["name"]
    worked_source = task["examples"][0]
    source_input = task["query"]["source_query"]["input"]
    source_target = task["answer"]["source_target"]

    initial_turn = EpisodeTurnSpec(
        turn_id="turn_1_source_rule",
        prompt={
            "probe_input": source_input,
            "request": "Infer the source-form rule and answer the source query.",
            "examples": task["examples"],
        },
        accepted_answers=[source_target],
        expected_confidence=0.74,
        expected_rule=rule_name,
    )

    direct_example = {
        "phase": "representation_shift_example",
        "input": _remap_tokens(worked_source["input"], source_vocab, direct_vocab),
        "output": _remap_tokens(worked_source["output"], source_vocab, direct_vocab),
    }
    alternate_example = {
        "phase": "representation_shift_example",
        "input": _remap_tokens(worked_source["input"], source_vocab, alternate_vocab),
        "output": _remap_tokens(worked_source["output"], source_vocab, alternate_vocab),
    }

    direct_turn = EpisodeTurnSpec(
        turn_id="turn_2_direct_transfer",
        event={
            "event_type": "representation_shift",
            "message": "The codebook changes, but the latent rule stays fixed.",
            "example": direct_example,
        },
        prompt={
            "probe_input": task["query"]["transfer_query"]["input"],
            "request": "Apply the same latent rule in the first transfer representation.",
        },
        accepted_answers=[task["answer"]["transfer_target"]],
        expected_confidence=0.8,
        expected_rule=rule_name,
    )
    alternate_turn = EpisodeTurnSpec(
        turn_id="turn_2_alternate_transfer",
        event={
            "event_type": "representation_shift_alternate",
            "message": "The same latent rule is expressed through a different nearby remapping.",
            "example": alternate_example,
        },
        prompt={
            "probe_input": _remap_tokens(source_input, source_vocab, alternate_vocab),
            "request": "Keep the rule fixed while revising the representation hypothesis.",
        },
        accepted_answers=[_remap_tokens(source_target, source_vocab, alternate_vocab)],
        expected_confidence=0.76,
        expected_rule=rule_name,
    )

    direct_spec = EpisodeSpec(
        task_id=task["task_id"],
        family=task["family"],
        episode_type="counterfactual_direct_transfer",
        initial_context={
            "instruction": task["context"]["instruction"],
            "source_representation": task["context"]["source_representation"],
            "transfer_representation": task["context"]["transfer_representation"],
        },
        turns=[initial_turn, direct_turn],
        expected={
            "accepted_initial_targets": [source_target],
            "accepted_revised_targets": [task["answer"]["transfer_target"]],
            "should_update": True,
            "contradiction_expected": False,
            "update_mode": "preserve_rule",
            "canonical_initial_rule": rule_name,
            "canonical_revised_rule": rule_name,
            "expected_initial_confidence": 0.74,
            "expected_revised_confidence": 0.8,
            "cue_turn": 2,
        },
    )
    alternate_spec = EpisodeSpec(
        task_id=task["task_id"],
        family=task["family"],
        episode_type="counterfactual_alternate_transfer",
        initial_context={
            "instruction": task["context"]["instruction"],
            "source_representation": task["context"]["source_representation"],
            "transfer_representation": "alternate_counterfactual_codebook",
        },
        turns=[initial_turn, alternate_turn],
        expected={
            "accepted_initial_targets": [source_target],
            "accepted_revised_targets": [_remap_tokens(source_target, source_vocab, alternate_vocab)],
            "should_update": True,
            "contradiction_expected": False,
            "update_mode": "preserve_rule",
            "canonical_initial_rule": rule_name,
            "canonical_revised_rule": rule_name,
            "expected_initial_confidence": 0.74,
            "expected_revised_confidence": 0.76,
            "cue_turn": 2,
        },
    )

    bundle_id = f"{task['task_id']}__counterfactual"
    return CounterfactualBundle(
        bundle_id=bundle_id,
        task_id=task["task_id"],
        family=task["family"],
        shared_structure={
            "base_question": "Does the model preserve the latent rule across nearby remapping variants?",
            "varying_factor": "representation_variant",
        },
        shared_checks=[
            {"name": "initial_answer", "path": "response.initial_answer", "expected": source_target},
            {"name": "initial_rule_tag", "path": "response.metadata.initial_rule_tag", "expected": rule_name},
            {"name": "revised_rule_tag", "path": "response.metadata.revised_rule_tag", "expected": rule_name},
        ],
        variant_path="response.revised_answer",
        branches=[
            CounterfactualBranch(
                branch_id=f"{bundle_id}__direct",
                label="direct_transfer",
                changed_factor="direct_codebook",
                expected_revision_kind="representation",
                spec=direct_spec,
                notes=["The model should change the surface answer while preserving the same rule tag."],
            ),
            CounterfactualBranch(
                branch_id=f"{bundle_id}__alternate",
                label="alternate_transfer",
                changed_factor="alternate_codebook",
                expected_revision_kind="representation",
                spec=alternate_spec,
                notes=["The alternate branch changes only the codebook, not the latent structure."],
            ),
        ],
    )


def _build_social_bundle(task: dict[str, Any]) -> CounterfactualBundle:
    agents = task["context"]["agents"]
    object_name = task["metadata"]["object_name"]
    initial_location = task["answer"]["belief_of_false_belief_agent"]
    actual_location = task["answer"]["actual_location"]
    witness = task["metadata"]["witness"]
    false_belief_agent = task["metadata"]["false_belief_agent"]
    deceiver = task["metadata"]["deceiver"]
    witness_statement = _find_statement(task, witness)
    deceptive_statement = _find_statement(task, deceiver)

    initial_answer = {
        "actual_location": initial_location,
        "belief_of_false_belief_agent": initial_location,
    }
    private_beliefs_turn_2 = {agent: initial_location for agent in agents}
    private_beliefs_turn_2[witness] = actual_location
    private_beliefs_turn_3 = dict(private_beliefs_turn_2)
    private_beliefs_turn_4 = dict(private_beliefs_turn_2)

    public_beliefs_turn_2 = {agent: actual_location for agent in agents}
    public_beliefs_turn_3 = dict(public_beliefs_turn_2)
    public_beliefs_turn_4 = dict(public_beliefs_turn_2)

    private_final = {
        "actual_location": actual_location,
        "belief_of_false_belief_agent": initial_location,
        "trusted_agent": witness,
    }
    public_final = {
        "actual_location": actual_location,
        "belief_of_false_belief_agent": actual_location,
        "trusted_agent": witness,
    }

    common_turn_1 = EpisodeTurnSpec(
        turn_id="turn_1_shared_state",
        event={"event_type": "shared_observation", "location": initial_location},
        prompt={
            "request": f"Track the initial state for the {object_name}.",
            "question": f"Where is the {object_name}, and where would {false_belief_agent} look right now?",
        },
        accepted_answers=[initial_answer],
        expected_confidence=0.72,
        expected_rule="belief_tracking",
        expected_beliefs={agent: initial_location for agent in agents},
    )

    private_turns = [
        common_turn_1,
        EpisodeTurnSpec(
            turn_id="turn_2_private_move",
            event={
                "event_type": "private_information_move",
                "message": f"The {object_name} moved to the {actual_location}. Only {witness} saw it.",
            },
            prompt={
                "request": "Update the world state after the private move.",
                "question": f"Where is the {object_name} now, and where does {false_belief_agent} still think it is?",
            },
            accepted_answers=[private_final],
            expected_confidence=0.58,
            expected_rule="belief_tracking",
            expected_beliefs=private_beliefs_turn_2,
            expected_trust_top=witness,
        ),
        EpisodeTurnSpec(
            turn_id="turn_3_conflicting_testimony",
            event={
                "event_type": "conflicting_testimony",
                "truthful_statement": witness_statement,
                "deceptive_statement": deceptive_statement,
            },
            prompt={
                "request": "Separate the true world state from stale agent belief under conflict.",
                "question": f"Where is the {object_name}, where does {false_belief_agent} think it is, and who is trustworthy?",
            },
            accepted_answers=[private_final],
            expected_confidence=0.74,
            expected_rule="belief_tracking",
            expected_beliefs=private_beliefs_turn_3,
            expected_trust_top=witness,
        ),
        EpisodeTurnSpec(
            turn_id="turn_4_final_query",
            event={"event_type": "final_query", "message": "Give the final social-belief summary."},
            prompt={
                "request": "Provide the final actual location, stale belief, and trusted agent.",
                "question": f"Where is the {object_name}, where does {false_belief_agent} believe it is, and who should be trusted?",
            },
            accepted_answers=[private_final],
            expected_confidence=0.82,
            expected_rule="belief_tracking",
            expected_beliefs=private_beliefs_turn_4,
            expected_trust_top=witness,
        ),
    ]

    public_turns = [
        common_turn_1,
        EpisodeTurnSpec(
            turn_id="turn_2_public_move",
            event={
                "event_type": "public_information_move",
                "message": f"The {object_name} moved to the {actual_location}, and everyone saw it.",
            },
            prompt={
                "request": "Update the world state after the public move.",
                "question": f"Where is the {object_name} now, and where does {false_belief_agent} think it is?",
            },
            accepted_answers=[public_final],
            expected_confidence=0.64,
            expected_rule="belief_tracking",
            expected_beliefs=public_beliefs_turn_2,
            expected_trust_top=witness,
        ),
        EpisodeTurnSpec(
            turn_id="turn_3_conflicting_testimony",
            event={
                "event_type": "conflicting_testimony",
                "truthful_statement": witness_statement,
                "deceptive_statement": deceptive_statement,
            },
            prompt={
                "request": "Keep the now-public world state separate from deceptive testimony.",
                "question": f"Where is the {object_name}, where does {false_belief_agent} think it is, and who is trustworthy?",
            },
            accepted_answers=[public_final],
            expected_confidence=0.78,
            expected_rule="belief_tracking",
            expected_beliefs=public_beliefs_turn_3,
            expected_trust_top=witness,
        ),
        EpisodeTurnSpec(
            turn_id="turn_4_final_query",
            event={"event_type": "final_query", "message": "Give the final social-belief summary."},
            prompt={
                "request": "Provide the final actual location, updated belief, and trusted agent.",
                "question": f"Where is the {object_name}, where does {false_belief_agent} believe it is, and who should be trusted?",
            },
            accepted_answers=[public_final],
            expected_confidence=0.84,
            expected_rule="belief_tracking",
            expected_beliefs=public_beliefs_turn_4,
            expected_trust_top=witness,
        ),
    ]

    private_spec = EpisodeSpec(
        task_id=task["task_id"],
        family=task["family"],
        episode_type="counterfactual_private_information",
        initial_context={
            "instruction": task["context"]["instruction"],
            "setting": task["context"]["setting"],
            "agents": agents,
        },
        turns=private_turns,
        expected={
            "accepted_initial_targets": [initial_answer],
            "accepted_revised_targets": [private_final],
            "should_update": True,
            "contradiction_expected": True,
            "update_mode": "belief_track",
            "canonical_initial_rule": "social_reasoning",
            "canonical_revised_rule": "social_reasoning",
            "expected_initial_confidence": 0.72,
            "expected_revised_confidence": 0.82,
            "cue_turn": 2,
            "reliable_agent": witness,
            "deceptive_agent": deceiver,
            "false_belief_agent": false_belief_agent,
            "initial_location": initial_location,
            "actual_location": actual_location,
        },
    )
    public_spec = EpisodeSpec(
        task_id=task["task_id"],
        family=task["family"],
        episode_type="counterfactual_public_information",
        initial_context={
            "instruction": task["context"]["instruction"],
            "setting": task["context"]["setting"],
            "agents": agents,
        },
        turns=public_turns,
        expected={
            "accepted_initial_targets": [initial_answer],
            "accepted_revised_targets": [public_final],
            "should_update": True,
            "contradiction_expected": True,
            "update_mode": "belief_track",
            "canonical_initial_rule": "social_reasoning",
            "canonical_revised_rule": "social_reasoning",
            "expected_initial_confidence": 0.72,
            "expected_revised_confidence": 0.84,
            "cue_turn": 2,
            "reliable_agent": witness,
            "deceptive_agent": deceiver,
            "false_belief_agent": false_belief_agent,
            "initial_location": initial_location,
            "actual_location": actual_location,
        },
    )

    bundle_id = f"{task['task_id']}__counterfactual"
    return CounterfactualBundle(
        bundle_id=bundle_id,
        task_id=task["task_id"],
        family=task["family"],
        shared_structure={
            "base_question": "Does the model keep world state stable while changing only the agent's information state?",
            "varying_factor": "private_information_access",
        },
        shared_checks=[
            {"name": "initial_answer", "path": "response.initial_answer", "expected": initial_answer},
            {"name": "actual_location", "path": "response.revised_answer.actual_location", "expected": actual_location},
            {"name": "trusted_agent", "path": "response.revised_answer.trusted_agent", "expected": witness},
        ],
        variant_path="response.revised_answer",
        branches=[
            CounterfactualBranch(
                branch_id=f"{bundle_id}__private",
                label="private_information",
                changed_factor="one_agent_private_information",
                expected_revision_kind="belief",
                spec=private_spec,
                notes=["Only one agent sees the move, so false-belief structure should remain."],
            ),
            CounterfactualBranch(
                branch_id=f"{bundle_id}__public",
                label="public_information",
                changed_factor="no_private_information",
                expected_revision_kind="belief",
                spec=public_spec,
                notes=["Everyone sees the move, so the false-belief slot should collapse."],
            ),
        ],
    )


def build_counterfactual_bundle(task: dict[str, Any]) -> CounterfactualBundle:
    """Create a small counterfactual bundle from one supported task."""
    family = task["family"]
    if family not in SUPPORTED_COUNTERFACTUAL_FAMILIES:
        raise ValueError(f"Counterfactual branching does not support family: {family}")
    if family == "hidden_rule":
        return _build_hidden_rule_bundle(task)
    if family == "shift_transfer":
        return _build_shift_transfer_bundle(task)
    return _build_social_bundle(task)


def generate_counterfactual_bundles(
    tasks: list[dict[str, Any]],
    *,
    families: set[str] | None = None,
    max_per_family: int | None = None,
) -> list[CounterfactualBundle]:
    """Generate deterministic branch bundles for a subset of tasks."""
    selected_families = families or SUPPORTED_COUNTERFACTUAL_FAMILIES
    family_counts: dict[str, int] = {family: 0 for family in sorted(selected_families)}
    bundles: list[CounterfactualBundle] = []
    for task in tasks:
        family = task["family"]
        if family not in selected_families:
            continue
        if max_per_family is not None and family_counts[family] >= max_per_family:
            continue
        bundles.append(build_counterfactual_bundle(task))
        family_counts[family] += 1
    return bundles


def save_counterfactual_bundles(path: Path, bundles: list[CounterfactualBundle]) -> None:
    """Persist generated branch bundles to JSON."""
    save_json(path, [bundle.to_dict() for bundle in bundles])


def _metric_payload(bundle_rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "counterfactual_update_fidelity": counterfactual_update_fidelity(bundle_rows),
        "invariant_preservation_score": invariant_preservation_score(bundle_rows),
        "branch_belief_coherence": branch_belief_coherence(bundle_rows),
        "cross_branch_consistency": cross_branch_consistency(bundle_rows),
        "counterfactual_confidence_calibration": counterfactual_confidence_calibration(bundle_rows),
    }


def _bundle_row(bundle: CounterfactualBundle, branch_sessions: list[dict[str, Any]]) -> dict[str, Any]:
    branches = []
    for branch, session in zip(bundle.branches, branch_sessions):
        trust_trace = session["derived"].get("trust_trace", [])
        belief_trace = session["derived"].get("belief_consistency_trace", [])
        branches.append(
            {
                "branch_id": branch.branch_id,
                "label": branch.label,
                "changed_factor": branch.changed_factor,
                "expected_revision_kind": branch.expected_revision_kind,
                "expected_final_target": branch.spec.expected["accepted_revised_targets"][0],
                "expected_turn_confidences": _expected_turn_confidences(branch.spec),
                "session": session,
                "branch_summary": {
                    "revised_correct": session["derived"]["revised_correct"],
                    "answer_changed": session["derived"]["answer_changed"],
                    "rule_changed": session["derived"]["rule_changed"],
                    "contradiction_detected": session["response"]["contradiction_detected"],
                    "final_trust_top": trust_trace[-1] if trust_trace else None,
                    "final_belief_consistency": belief_trace[-1] if belief_trace else None,
                },
            }
        )

    bundle_row = {
        "bundle_id": bundle.bundle_id,
        "task_id": bundle.task_id,
        "family": bundle.family,
        "shared_structure": bundle.shared_structure,
        "shared_checks": bundle.shared_checks,
        "variant_path": bundle.variant_path,
        "branches": branches,
    }
    metrics = _metric_payload([bundle_row])
    bundle_row["metrics"] = metrics
    bundle_row["counterfactual_composite"] = round(sum(metrics.values()) / len(metrics), 4)
    return bundle_row


def evaluate_counterfactual_bundles(
    bundles: list[CounterfactualBundle],
    responder,
    *,
    run_name: str,
    adapter_description: dict[str, Any] | None = None,
    progress_callback: Callable[[int, int, int, bool], None] | None = None,
) -> dict[str, Any]:
    """Run and score counterfactual branch bundles."""
    bundle_rows = []
    bundles_completed = 0
    branches_completed = 0
    num_errors = 0
    for bundle in bundles:
        branch_sessions = []
        for branch in bundle.branches:
            try:
                branch_sessions.append(run_episode_spec(branch.spec, responder))
            except Exception:
                num_errors += 1
                if progress_callback is not None:
                    progress_callback(bundles_completed, branches_completed, num_errors, False)
                raise
            branches_completed += 1
            if progress_callback is not None:
                progress_callback(bundles_completed, branches_completed, num_errors, False)
        bundle_rows.append(_bundle_row(bundle, branch_sessions))
        bundles_completed += 1
        if progress_callback is not None:
            progress_callback(bundles_completed, branches_completed, num_errors, False)

    family_metrics: dict[str, dict[str, float]] = {}
    for family in sorted({bundle["family"] for bundle in bundle_rows}):
        family_rows = [bundle for bundle in bundle_rows if bundle["family"] == family]
        family_metrics[family] = _metric_payload(family_rows)

    top_gaps = sorted(bundle_rows, key=lambda row: (row["counterfactual_composite"], row["bundle_id"]))[:5]
    return {
        "run_name": run_name,
        "adapter": adapter_description,
        "num_bundles": len(bundle_rows),
        "num_branches": sum(len(bundle["branches"]) for bundle in bundle_rows),
        "families": sorted({bundle["family"] for bundle in bundle_rows}),
        "overall_metrics": _metric_payload(bundle_rows),
        "family_metrics": family_metrics,
        "top_counterfactual_gaps": [
            {
                "bundle_id": row["bundle_id"],
                "task_id": row["task_id"],
                "family": row["family"],
                "counterfactual_composite": row["counterfactual_composite"],
                "shared_structure": row["shared_structure"],
            }
            for row in top_gaps
        ],
        "bundle_rows": bundle_rows,
    }


def write_counterfactual_artifacts(run_dir: Path, summary: dict[str, Any]) -> None:
    """Write JSON and markdown outputs for one counterfactual evaluation run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "counterfactual_summary.json", summary)

    lines = [
        f"# Counterfactual Highlights: {summary['run_name']}",
        "",
        f"- counterfactual_update_fidelity: {summary['overall_metrics']['counterfactual_update_fidelity']}",
        f"- invariant_preservation_score: {summary['overall_metrics']['invariant_preservation_score']}",
        f"- branch_belief_coherence: {summary['overall_metrics']['branch_belief_coherence']}",
        f"- cross_branch_consistency: {summary['overall_metrics']['cross_branch_consistency']}",
        f"- counterfactual_confidence_calibration: {summary['overall_metrics']['counterfactual_confidence_calibration']}",
        "",
        "## Lowest-Scoring Bundles",
    ]
    for row in summary["top_counterfactual_gaps"]:
        lines.append(
            f"- `{row['bundle_id']}` | `{row['family']}` | composite={row['counterfactual_composite']}"
        )
        lines.append(
            f"  Why interesting: nearby branches vary `{row['shared_structure']['varying_factor']}` while holding the base episode fixed."
        )

    (run_dir / "counterfactual_highlights.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
