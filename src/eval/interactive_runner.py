"""Turn-based and episode-style interactive evaluation for AGUS."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from src.generators.common import RuleSpec
from src.scoring.metrics import (
    attention_recovery_score,
    belief_state_consistency,
    belief_trajectory_quality,
    confidence_recalibration_score,
    contradiction_sensitivity,
    cue_utilization_score,
    deception_sensitivity,
    distractor_capture_rate,
    episode_cognitive_flexibility_score,
    exact_match,
    hypothesis_update_score,
    multi_turn_adaptation_score,
    online_adaptation_gain,
    trust_revision_score,
)
from src.schemas.response_schema import InteractiveResponse, TurnResponse
from src.utils.seeds import make_rng

SUPPORTED_INTERACTIVE_FAMILIES = {
    "hidden_rule",
    "shift_transfer",
    "metacog_revision",
    "attention_distractors",
    "social_miniworlds",
}


@dataclass(frozen=True)
class EpisodeTurnSpec:
    """One turn inside an interactive evaluation episode."""

    turn_id: str
    prompt: dict[str, Any]
    event: dict[str, Any] | None = None
    accepted_answers: list[Any] = field(default_factory=list)
    expected_confidence: float = 0.0
    expected_rule: Any = None
    expected_attended_signals: list[str] = field(default_factory=list)
    expected_ignored_signals: list[str] = field(default_factory=list)
    expected_beliefs: dict[str, Any] = field(default_factory=dict)
    expected_trust_top: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize one episode turn."""
        return asdict(self)


@dataclass(frozen=True)
class EpisodeSpec:
    """Interactive evaluation spec for one task."""

    task_id: str
    family: str
    episode_type: str
    initial_context: dict[str, Any]
    turns: list[EpisodeTurnSpec]
    expected: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the episode spec."""
        return asdict(self)


TurnResponder = Callable[[EpisodeSpec, EpisodeTurnSpec, list[dict[str, Any]]], dict[str, Any]]


def _rule_from_payload(payload: dict[str, Any]) -> RuleSpec:
    return RuleSpec(name=payload["name"], params=payload["params"])


def _remap_tokens(tokens: list[str], source_vocab: list[str], target_vocab: list[str]) -> list[str]:
    index_map = {token: idx for idx, token in enumerate(source_vocab)}
    return [target_vocab[index_map[token]] for token in tokens]


def _accepted_match(answer: Any, accepted_targets: list[Any]) -> bool:
    return any(exact_match(answer, target) for target in accepted_targets)


def _infer_domain_size(task: dict[str, Any], fallback: int = 10) -> int:
    """Infer numeric domain size from a task payload."""
    values: list[int] = []

    def walk(node: Any) -> None:
        if isinstance(node, int):
            values.append(node)
            return
        if isinstance(node, list):
            for item in node:
                walk(item)
        if isinstance(node, dict):
            for item in node.values():
                walk(item)

    walk(task.get("examples", []))
    walk(task.get("query", {}))
    walk(task.get("answer", {}))
    if not values:
        return fallback
    return max(fallback, max(values) + 1)


def _parse_task_index(task_id: str) -> int:
    return int(task_id.rsplit("_", 1)[-1])


def _make_wrong_answer(target: Any) -> Any:
    """Create a deterministic wrong answer of the same broad shape."""
    if isinstance(target, list):
        if len(target) > 1:
            return list(reversed(target))
        if target:
            value = target[0]
            if isinstance(value, int):
                return [value + 1]
            return [f"{value}_alt"]
        return target
    if isinstance(target, dict):
        wrong = dict(target)
        for key, value in wrong.items():
            wrong[key] = _make_wrong_answer(value)
            break
        return wrong
    if isinstance(target, int):
        return target + 1
    if isinstance(target, str):
        return f"{target}_alt"
    return target


def _top_agent(scores: dict[str, float]) -> str | None:
    if not scores:
        return None
    return max(scores.items(), key=lambda item: (item[1], item[0]))[0]


def _find_statement(task: dict[str, Any], agent: str) -> dict[str, Any] | None:
    for example in task["examples"]:
        if example.get("phase") == "statement" and example.get("agent") == agent:
            return example
    return None


def _build_hidden_rule_episode(task: dict[str, Any]) -> EpisodeSpec:
    induction_examples = [example for example in task["examples"] if example["phase"] == "induction"]
    shift_feedback = next(example for example in task["examples"] if example["phase"] == "shift_feedback")
    induction_rule = _rule_from_payload(task["metadata"]["internal_rules"]["induction"])
    probe_input = task["query"]["shift_queries"][0]["input"]
    domain_size = len(task["context"]["symbol_space"])
    initial_target = induction_rule.apply(probe_input, domain_size)
    revised_target = task["answer"]["shift_targets"][0]

    turns = [
        EpisodeTurnSpec(
            turn_id="turn_1_initial_hypothesis",
            prompt={
                "probe_input": probe_input,
                "request": "Infer the current rule from the induction examples and answer the probe.",
                "examples": induction_examples,
            },
            accepted_answers=[initial_target],
            expected_confidence=0.65,
            expected_rule=task["metadata"]["internal_rules"]["induction"]["name"],
        ),
        EpisodeTurnSpec(
            turn_id="turn_2_rule_shift",
            event={
                "event_type": "rule_shift_contradiction",
                "message": "A new observation conflicts with the earlier pattern.",
                "example": shift_feedback,
            },
            prompt={
                "probe_input": probe_input,
                "request": "Revise the rule and answer the same probe after seeing the contradiction.",
            },
            accepted_answers=[revised_target],
            expected_confidence=0.82,
            expected_rule=task["metadata"]["internal_rules"]["shifted"]["name"],
        ),
    ]

    return EpisodeSpec(
        task_id=task["task_id"],
        family=task["family"],
        episode_type="two_turn_rule_shift",
        initial_context={
            "instruction": task["context"]["instruction"],
            "symbol_space": task["context"]["symbol_space"],
            "sequence_length": task["context"]["sequence_length"],
        },
        turns=turns,
        expected={
            "accepted_initial_targets": [initial_target],
            "accepted_revised_targets": [revised_target],
            "should_update": True,
            "contradiction_expected": True,
            "update_mode": "change_rule",
            "canonical_initial_rule": task["metadata"]["internal_rules"]["induction"]["name"],
            "canonical_revised_rule": task["metadata"]["internal_rules"]["shifted"]["name"],
            "expected_initial_confidence": 0.65,
            "expected_revised_confidence": 0.82,
            "cue_turn": 2,
        },
    )


def _build_shift_transfer_episode(task: dict[str, Any]) -> EpisodeSpec:
    source_vocab = task["metadata"]["source_vocab"]
    transfer_vocab = task["metadata"]["transfer_vocab"]
    bridge_vocab = task["metadata"].get("bridge_vocab")
    worked_source = task["examples"][0]
    worked_transfer = {
        "phase": "representation_remap_example",
        "input": _remap_tokens(worked_source["input"], source_vocab, transfer_vocab),
        "output": _remap_tokens(worked_source["output"], source_vocab, transfer_vocab),
    }

    turns = [
        EpisodeTurnSpec(
            turn_id="turn_1_source_rule",
            prompt={
                "probe_input": task["query"]["source_query"]["input"],
                "request": "Answer the source-form query, explain the rule, and provide confidence.",
                "examples": task["examples"],
            },
            accepted_answers=[task["answer"]["source_target"]],
            expected_confidence=0.74,
            expected_rule=task["metadata"]["internal_rule"]["name"],
        )
    ]

    episode_type = "two_turn_representation_shift"
    cue_turn = 2
    if bridge_vocab and task["query"].get("bridge_query") and task["answer"].get("bridge_target"):
        worked_bridge = {
            "phase": "bridge_representation_example",
            "input": _remap_tokens(worked_source["input"], source_vocab, bridge_vocab),
            "output": _remap_tokens(worked_source["output"], source_vocab, bridge_vocab),
        }
        turns.append(
            EpisodeTurnSpec(
                turn_id="turn_2_bridge_representation",
                event={
                    "event_type": "bridge_representation_shift",
                    "message": "The codebook changes once, but the latent rule stays fixed.",
                    "example": worked_bridge,
                },
                prompt={
                    "probe_input": task["query"]["bridge_query"]["input"],
                    "request": "Maintain the same rule while revising the representation hypothesis.",
                    "examples": [worked_bridge],
                },
                accepted_answers=[task["answer"]["bridge_target"]],
                expected_confidence=0.72,
                expected_rule=task["metadata"]["internal_rule"]["name"],
            )
        )
        episode_type = "three_turn_representation_composition"
        cue_turn = 3

    turns.append(
        EpisodeTurnSpec(
            turn_id=f"turn_{len(turns) + 1}_representation_shift",
            event={
                "event_type": "representation_shift",
                "message": "The symbols are remapped again, but the latent structure is unchanged.",
                "example": worked_transfer,
            },
            prompt={
                "probe_input": task["query"]["transfer_query"]["input"],
                "request": "Apply the same latent rule in the new representation.",
                "examples": [worked_transfer],
            },
            accepted_answers=[task["answer"]["transfer_target"]],
            expected_confidence=0.82 if episode_type != "two_turn_representation_shift" else 0.8,
            expected_rule=task["metadata"]["internal_rule"]["name"],
        )
    )

    return EpisodeSpec(
        task_id=task["task_id"],
        family=task["family"],
        episode_type=episode_type,
        initial_context={
            "instruction": task["context"]["instruction"],
            "source_representation": task["context"]["source_representation"],
            "transfer_representation": task["context"]["transfer_representation"],
            "bridge_representation": task["context"].get("bridge_representation"),
        },
        turns=turns,
        expected={
            "accepted_initial_targets": [task["answer"]["source_target"]],
            "accepted_revised_targets": [task["answer"]["transfer_target"]],
            "should_update": True,
            "contradiction_expected": False,
            "update_mode": "preserve_rule",
            "canonical_initial_rule": task["metadata"]["internal_rule"]["name"],
            "canonical_revised_rule": task["metadata"]["internal_rule"]["name"],
            "expected_initial_confidence": 0.74,
            "expected_revised_confidence": turns[-1].expected_confidence,
            "cue_turn": cue_turn,
            "bridge_expected": bridge_vocab is not None,
        },
    )


def _build_metacog_episode(task: dict[str, Any]) -> EpisodeSpec:
    ambiguous_examples = [example for example in task["examples"] if example["phase"] == "ambiguous_evidence"]
    corrective_example = next(example for example in task["examples"] if example["phase"] == "corrective_evidence")
    turns = [
        EpisodeTurnSpec(
            turn_id="turn_1_ambiguous_hypothesis",
            prompt={
                "probe_input": task["query"]["initial_query"]["input"],
                "request": "Give a tentative answer, hypothesis, and confidence under ambiguity.",
                "examples": ambiguous_examples,
            },
            accepted_answers=task["answer"]["acceptable_initial_targets"],
            expected_confidence=float(task["metadata"]["expected_initial_certainty"]),
            expected_rule=[rule["name"] for rule in task["metadata"]["candidate_rules"]],
        ),
        EpisodeTurnSpec(
            turn_id="turn_2_corrective_revision",
            event={
                "event_type": "corrective_evidence",
                "message": "A new example disambiguates the competing hypotheses.",
                "example": corrective_example,
            },
            prompt={
                "probe_input": task["query"]["revision_prompt"]["revise_same_input"],
                "request": "Revise your answer, confidence, and hypothesis after the corrective evidence.",
            },
            accepted_answers=[task["answer"]["revised_target"]],
            expected_confidence=0.9,
            expected_rule=task["metadata"]["actual_rule"]["name"],
        ),
    ]

    return EpisodeSpec(
        task_id=task["task_id"],
        family=task["family"],
        episode_type="two_turn_belief_revision",
        initial_context={
            "instruction": task["context"]["instruction"],
            "response_fields": task["context"]["response_fields"],
        },
        turns=turns,
        expected={
            "accepted_initial_targets": task["answer"]["acceptable_initial_targets"],
            "accepted_revised_targets": [task["answer"]["revised_target"]],
            "should_update": True,
            "contradiction_expected": True,
            "update_mode": "disambiguate_rule",
            "canonical_initial_rule": [rule["name"] for rule in task["metadata"]["candidate_rules"]],
            "canonical_revised_rule": task["metadata"]["actual_rule"]["name"],
            "expected_initial_confidence": float(task["metadata"]["expected_initial_certainty"]),
            "expected_revised_confidence": 0.9,
            "cue_turn": 2,
        },
    )


def _build_attention_episode(task: dict[str, Any]) -> EpisodeSpec:
    rule = _rule_from_payload(task["metadata"]["internal_rule"])
    domain_size = _infer_domain_size(task)
    examples = task["examples"]
    query_record = task["query"]["record"]
    target = task["answer"]["target"]
    decoy_target = rule.apply(query_record["distractor_sequence"], domain_size)
    alternate_distractor = examples[0]["distractor_sequence"]
    reference_example = examples[0]
    if exact_match(alternate_distractor, query_record["distractor_sequence"]) and len(examples) > 1:
        alternate_distractor = examples[1]["distractor_sequence"]

    controlled_cue = {
        "phase": "controlled_cue",
        "signal_sequence": reference_example["signal_sequence"],
        "distractor_sequence": alternate_distractor,
        "output": reference_example["output"],
        "message": "The output stays tied to the signal even when the distractor changes, but this is not the query row.",
    }

    turns = [
        EpisodeTurnSpec(
            turn_id="turn_1_mixed_context",
            prompt={
                "query_record": query_record,
                "examples": examples[:2],
                "request": (
                    "Answer the query, state which fields you are attending to, and provide confidence "
                    "under mixed relevant and irrelevant information."
                ),
            },
            accepted_answers=[target],
            expected_confidence=0.62,
            expected_rule=task["metadata"]["internal_rule"]["name"],
            expected_attended_signals=["signal_sequence"],
            expected_ignored_signals=["distractor_sequence"],
        ),
        EpisodeTurnSpec(
            turn_id="turn_2_noise_escalation",
            event={
                "event_type": "noise_escalation",
                "message": "A new high-salience decoy appears and suggests the distractor field may matter.",
                "example": examples[2],
                "decoy_hypothesis_output": decoy_target,
            },
            prompt={
                "query_record": query_record,
                "request": "Update your answer and confidence under increased clutter.",
            },
            accepted_answers=[target],
            expected_confidence=0.48 if task["distractor_level"] >= 2 else 0.56,
            expected_rule=task["metadata"]["internal_rule"]["name"],
            expected_attended_signals=["signal_sequence"],
            expected_ignored_signals=["distractor_sequence"],
        ),
    ]
    cue_delay_level = int(task["metadata"].get("cue_delay_level", 1))
    cue_turn = 3
    if cue_delay_level > 1:
        turns.append(
            EpisodeTurnSpec(
                turn_id="turn_3_partial_decoy_summary",
                event={
                    "event_type": "stale_pattern_summary",
                    "message": "A compressed recap emphasizes the wrong field before the causal cue arrives.",
                    "example": examples[3],
                },
                prompt={
                    "query_record": query_record,
                    "request": "Update your answer under a misleading summary without overcommitting.",
                },
                accepted_answers=[target],
                expected_confidence=0.44,
                expected_rule=task["metadata"]["internal_rule"]["name"],
                expected_attended_signals=["signal_sequence"],
                expected_ignored_signals=["distractor_sequence"],
            )
        )
        cue_turn = 4

    turns.append(
        EpisodeTurnSpec(
            turn_id=f"turn_{cue_turn}_disambiguating_cue",
            event={
                "event_type": "disambiguating_cue",
                "message": "A controlled test reveals which field actually matters.",
                "controlled_example": controlled_cue,
            },
            prompt={
                "query_record": query_record,
                "request": "Revise your answer, attended signals, and confidence after the cue.",
            },
            accepted_answers=[target],
            expected_confidence=0.72,
            expected_rule=task["metadata"]["internal_rule"]["name"],
            expected_attended_signals=["signal_sequence"],
            expected_ignored_signals=["distractor_sequence"],
        )
    )
    turns.append(
        EpisodeTurnSpec(
            turn_id=f"turn_{cue_turn + 1}_compressed_summary",
            event={
                "event_type": "compressed_summary_challenge",
                "message": "Now answer again from a compressed recap while preserving the correct attentional focus.",
                "summary": {
                    "relevant_field": "signal_sequence",
                    "query_signal_sequence": query_record["signal_sequence"],
                    "query_distractor_sequence": query_record["distractor_sequence"],
                },
            },
            prompt={
                "query_record": query_record,
                "request": "Give the final answer and name the causally relevant field.",
            },
            accepted_answers=[target],
            expected_confidence=0.82,
            expected_rule=task["metadata"]["internal_rule"]["name"],
            expected_attended_signals=["signal_sequence"],
            expected_ignored_signals=["distractor_sequence"],
        )
    )

    return EpisodeSpec(
        task_id=task["task_id"],
        family=task["family"],
        episode_type="attention_recovery_episode",
        initial_context={
            "instruction": task["context"]["instruction"],
            "distractor_level": task["distractor_level"],
        },
        turns=turns,
        expected={
            "accepted_initial_targets": [target],
            "accepted_revised_targets": [target],
            "should_update": True,
            "contradiction_expected": True,
            "update_mode": "refocus_signal",
            "canonical_initial_rule": task["metadata"]["internal_rule"]["name"],
            "canonical_revised_rule": task["metadata"]["internal_rule"]["name"],
            "expected_initial_confidence": turns[0].expected_confidence,
            "expected_revised_confidence": turns[-1].expected_confidence,
            "cue_turn": cue_turn,
            "trap_signal": "distractor_sequence",
            "relevant_signal": "signal_sequence",
            "decoy_target": decoy_target,
            "distractor_level": task["distractor_level"],
        },
    )


def _build_social_episode(task: dict[str, Any]) -> EpisodeSpec:
    task_index = _parse_task_index(task["task_id"])
    episode_type = ("false_belief_update", "trust_revision", "incentive_revision")[task_index % 3]
    agents = task["context"]["agents"]
    object_name = task["metadata"]["object_name"]
    initial_location = task["answer"]["belief_of_false_belief_agent"]
    actual_location = task["answer"]["actual_location"]
    witness = task["metadata"]["witness"]
    false_belief_agent = task["metadata"]["false_belief_agent"]
    deceiver = task["metadata"]["deceiver"]
    rumor_agent = task["metadata"]["rumor_agent"]
    witness_statement = _find_statement(task, witness)
    deceiver_statement = _find_statement(task, deceiver)
    rumor_statement = _find_statement(task, rumor_agent) if rumor_agent else None

    beliefs_turn_1 = {agent: initial_location for agent in agents}
    beliefs_turn_2 = {agent: initial_location for agent in agents}
    beliefs_turn_2[witness] = actual_location
    beliefs_turn_3 = dict(beliefs_turn_2)
    beliefs_turn_4 = dict(beliefs_turn_3)
    if rumor_agent is not None:
        beliefs_turn_4[rumor_agent] = actual_location

    trust_scores_final = {agent: 0.3 for agent in agents}
    trust_scores_final[witness] = 0.9
    trust_scores_final[deceiver] = 0.1
    if rumor_agent is not None:
        trust_scores_final[rumor_agent] = 0.55

    if episode_type == "false_belief_update":
        turns = [
            EpisodeTurnSpec(
                turn_id="turn_1_shared_observation",
                event={"event_type": "shared_observation", "location": initial_location},
                prompt={
                    "request": f"Track the initial world state for the {object_name}.",
                    "question": f"Where would each agent look for the {object_name} right now?",
                },
                accepted_answers=[{"actual_location": initial_location, "belief_of_false_belief_agent": initial_location}],
                expected_confidence=0.72,
                expected_rule="belief_tracking",
                expected_beliefs=beliefs_turn_1,
            ),
            EpisodeTurnSpec(
                turn_id="turn_2_hidden_move",
                event={
                    "event_type": "hidden_move",
                    "message": f"The {object_name} moved to the {actual_location}. Only {witness} saw it.",
                },
                prompt={
                    "request": "Update the world state and belief state after the hidden move.",
                    "question": f"Where is the {object_name} actually, and where does {false_belief_agent} still think it is?",
                },
                accepted_answers=[{"actual_location": actual_location, "belief_of_false_belief_agent": initial_location}],
                expected_confidence=0.58,
                expected_rule="belief_tracking",
                expected_beliefs=beliefs_turn_2,
            ),
            EpisodeTurnSpec(
                turn_id="turn_3_conflicting_testimony",
                event={
                    "event_type": "conflicting_testimony",
                    "truthful_statement": witness_statement,
                    "deceptive_statement": deceiver_statement,
                },
                prompt={
                    "request": "Separate world state from stale belief state under conflicting testimony.",
                    "question": (
                        f"Who should a newcomer trust, and where does {false_belief_agent} still believe the "
                        f"{object_name} is?"
                    ),
                },
                accepted_answers=[
                    {
                        "actual_location": actual_location,
                        "belief_of_false_belief_agent": initial_location,
                        "trusted_agent": witness,
                    }
                ],
                expected_confidence=0.7,
                expected_rule="belief_tracking",
                expected_beliefs=beliefs_turn_3,
                expected_trust_top=witness,
            ),
            EpisodeTurnSpec(
                turn_id="turn_4_final_social_query",
                event={
                    "event_type": "final_query",
                    "message": "Answer with the current world state, stale belief, and best informant.",
                    "rumor_statement": rumor_statement,
                },
                prompt={
                    "request": "Give the final social inference summary.",
                    "question": (
                        f"Where is the {object_name}, where does {false_belief_agent} think it is, "
                        f"and who is most trustworthy?"
                    ),
                },
                accepted_answers=[
                    {
                        "actual_location": actual_location,
                        "belief_of_false_belief_agent": initial_location,
                        "trusted_agent": witness,
                    }
                ],
                expected_confidence=0.82,
                expected_rule="belief_tracking",
                expected_beliefs=beliefs_turn_4,
                expected_trust_top=witness,
            ),
        ]
    elif episode_type == "trust_revision":
        turns = [
            EpisodeTurnSpec(
                turn_id="turn_1_conflicting_reports",
                event={
                    "event_type": "conflicting_reports",
                    "truthful_statement": witness_statement,
                    "deceptive_statement": deceiver_statement,
                },
                prompt={
                    "request": "Judge the current location and decide whom to trust under conflicting reports.",
                    "question": f"Where is the {object_name}, and who currently seems most trustworthy?",
                },
                accepted_answers=[{"actual_location": actual_location, "trusted_agent": witness}],
                expected_confidence=0.56,
                expected_rule="trust_revision",
                expected_beliefs=beliefs_turn_2,
                expected_trust_top=witness,
            ),
            EpisodeTurnSpec(
                turn_id="turn_2_incentive_reveal",
                event={
                    "event_type": "incentive_reveal",
                    "message": f"{deceiver} benefits if others search the wrong place.",
                },
                prompt={
                    "request": "Revise your trust assessment after learning the incentive structure.",
                    "question": f"Who should be trusted now for the {object_name}'s location?",
                },
                accepted_answers=[{"actual_location": actual_location, "trusted_agent": witness}],
                expected_confidence=0.74,
                expected_rule="trust_revision",
                expected_beliefs=beliefs_turn_3,
                expected_trust_top=witness,
            ),
            EpisodeTurnSpec(
                turn_id="turn_3_corroboration",
                event={
                    "event_type": "corroboration",
                    "message": "A second report is consistent with the truthful witness.",
                    "rumor_statement": rumor_statement,
                },
                prompt={
                    "request": "Incorporate corroboration and update trust if needed.",
                    "question": f"Where is the {object_name}, and who is the best informant now?",
                },
                accepted_answers=[{"actual_location": actual_location, "trusted_agent": witness}],
                expected_confidence=0.8,
                expected_rule="trust_revision",
                expected_beliefs=beliefs_turn_4,
                expected_trust_top=witness,
            ),
            EpisodeTurnSpec(
                turn_id="turn_4_final_trust_query",
                event={
                    "event_type": "final_query",
                    "message": "Give the final trust decision and likely deceptive action.",
                },
                prompt={
                    "request": "Answer with the final trust decision and likely action of the deceptive agent.",
                    "question": f"Who is trustworthy, and what is {deceiver} likely trying to do?",
                },
                accepted_answers=[
                    {
                        "actual_location": actual_location,
                        "trusted_agent": witness,
                        "likely_action": "mislead_search",
                    }
                ],
                expected_confidence=0.86,
                expected_rule="trust_revision",
                expected_beliefs=beliefs_turn_4,
                expected_trust_top=witness,
            ),
        ]
    else:
        turns = [
            EpisodeTurnSpec(
                turn_id="turn_1_hidden_move_and_claim",
                event={
                    "event_type": "hidden_move",
                    "message": f"The {object_name} moved to the {actual_location}, observed only by {witness}.",
                    "truthful_statement": witness_statement,
                },
                prompt={
                    "request": "Infer the current world state from partial observation.",
                    "question": f"Where is the {object_name}, and whom would you currently trust?",
                },
                accepted_answers=[{"actual_location": actual_location, "trusted_agent": witness}],
                expected_confidence=0.6,
                expected_rule="intent_revision",
                expected_beliefs=beliefs_turn_2,
                expected_trust_top=witness,
            ),
            EpisodeTurnSpec(
                turn_id="turn_2_conflicting_claim",
                event={
                    "event_type": "conflicting_statement",
                    "deceptive_statement": deceiver_statement,
                },
                prompt={
                    "request": "Incorporate the conflicting claim before incentives are fully known.",
                    "question": f"Where is the {object_name}, and what do you think {deceiver} believes?",
                },
                accepted_answers=[{"actual_location": actual_location, "trusted_agent": witness}],
                expected_confidence=0.52,
                expected_rule="intent_revision",
                expected_beliefs=beliefs_turn_3,
                expected_trust_top=witness,
            ),
            EpisodeTurnSpec(
                turn_id="turn_3_intent_reveal",
                event={
                    "event_type": "intent_reveal",
                    "message": f"{deceiver} benefits if others are sent to the wrong location.",
                },
                prompt={
                    "request": "Revise your inference about intent and likely action.",
                    "question": f"What is {deceiver} likely trying to do, and who should be trusted?",
                },
                accepted_answers=[
                    {
                        "actual_location": actual_location,
                        "trusted_agent": witness,
                        "likely_action": "mislead_search",
                    }
                ],
                expected_confidence=0.78,
                expected_rule="intent_revision",
                expected_beliefs=beliefs_turn_3,
                expected_trust_top=witness,
            ),
            EpisodeTurnSpec(
                turn_id="turn_4_final_social_query",
                event={
                    "event_type": "final_query",
                    "message": "Separate world state, stale belief state, and deceptive intent.",
                },
                prompt={
                    "request": "Give the final state, belief, trust, and intent summary.",
                    "question": (
                        f"Where is the {object_name}, where does {false_belief_agent} think it is, "
                        f"who should be trusted, and what is {deceiver} likely to do?"
                    ),
                },
                accepted_answers=[
                    {
                        "actual_location": actual_location,
                        "belief_of_false_belief_agent": initial_location,
                        "trusted_agent": witness,
                        "likely_action": "mislead_search",
                    }
                ],
                expected_confidence=0.85,
                expected_rule="intent_revision",
                expected_beliefs=beliefs_turn_4,
                expected_trust_top=witness,
            ),
        ]

    return EpisodeSpec(
        task_id=task["task_id"],
        family=task["family"],
        episode_type=episode_type,
        initial_context={
            "instruction": task["context"]["instruction"],
            "setting": task["context"]["setting"],
            "agents": agents,
        },
        turns=turns,
        expected={
            "accepted_initial_targets": turns[0].accepted_answers,
            "accepted_revised_targets": turns[-1].accepted_answers,
            "should_update": True,
            "contradiction_expected": True,
            "update_mode": "belief_track",
            "canonical_initial_rule": "social_reasoning",
            "canonical_revised_rule": "social_reasoning",
            "expected_initial_confidence": turns[0].expected_confidence,
            "expected_revised_confidence": turns[-1].expected_confidence,
            "cue_turn": 2,
            "reliable_agent": witness,
            "deceptive_agent": deceiver,
            "false_belief_agent": false_belief_agent,
            "initial_location": initial_location,
            "actual_location": actual_location,
            "social_episode_type": episode_type,
        },
    )


def build_interaction_spec(task: dict[str, Any]) -> EpisodeSpec:
    """Build a deterministic interactive protocol from a static task."""
    family = task["family"]
    if family not in SUPPORTED_INTERACTIVE_FAMILIES:
        raise ValueError(f"Interactive evaluation does not yet support family: {family}")
    if family == "hidden_rule":
        return _build_hidden_rule_episode(task)
    if family == "shift_transfer":
        return _build_shift_transfer_episode(task)
    if family == "metacog_revision":
        return _build_metacog_episode(task)
    if family == "attention_distractors":
        return _build_attention_episode(task)
    return _build_social_episode(task)


def _build_turn_response(turn: EpisodeTurnSpec, payload: dict[str, Any]) -> TurnResponse:
    """Normalize a raw turn payload into the response schema."""
    trust_scores = {agent: float(score) for agent, score in payload.get("trust_scores_by_agent", {}).items()}
    return TurnResponse(
        turn_id=turn.turn_id,
        answer=payload.get("answer"),
        confidence=float(payload.get("confidence", 0.0)),
        rule_explanation=str(payload.get("rule_explanation", "")),
        evidence_acknowledged=bool(payload.get("evidence_acknowledged", turn.event is not None)),
        contradiction_detected=bool(payload.get("contradiction_detected", False)),
        attended_signals=list(payload.get("attended_signals", [])),
        ignored_signals=list(payload.get("ignored_signals", [])),
        trust_scores_by_agent=trust_scores,
        inferred_agent_beliefs=dict(payload.get("inferred_agent_beliefs", {})),
        revision_events=list(payload.get("revision_events", [])),
        metadata=dict(payload.get("metadata", {})),
    )


def _derive_turn_fields(turn: EpisodeTurnSpec, response: TurnResponse) -> dict[str, Any]:
    """Compute per-turn diagnostics."""
    expected_beliefs = turn.expected_beliefs
    belief_hits = []
    for agent, belief in expected_beliefs.items():
        belief_hits.append(exact_match(response.inferred_agent_beliefs.get(agent), belief))
    beliefs_consistent = sum(belief_hits) / len(belief_hits) if belief_hits else None

    expected_trust_top = turn.expected_trust_top
    observed_trust_top = _top_agent(response.trust_scores_by_agent)
    return {
        "answer_correct": _accepted_match(response.answer, turn.accepted_answers) if turn.accepted_answers else False,
        "signal_focus_correct": all(signal in response.attended_signals for signal in turn.expected_attended_signals)
        if turn.expected_attended_signals
        else None,
        "ignored_distractor": all(signal in response.ignored_signals for signal in turn.expected_ignored_signals)
        if turn.expected_ignored_signals
        else None,
        "belief_consistency": beliefs_consistent,
        "trust_top_correct": observed_trust_top == expected_trust_top if expected_trust_top else None,
        "observed_trust_top": observed_trust_top,
    }


def _derive_session_fields(spec: EpisodeSpec, response: InteractiveResponse, turn_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute reusable derived fields from an interactive session."""
    initial_rule_tag = response.metadata.get("initial_rule_tag", response.initial_rule_explanation)
    revised_rule_tag = response.metadata.get("revised_rule_tag", response.revised_rule_explanation)
    turn_correctness = [record["derived"]["answer_correct"] for record in turn_records]
    initial_correct = turn_correctness[0]
    revised_correct = turn_correctness[-1]
    answer_changed = response.initial_answer != response.revised_answer
    rule_changed = str(initial_rule_tag).strip().lower() != str(revised_rule_tag).strip().lower()

    derived = {
        "initial_correct": initial_correct,
        "revised_correct": revised_correct,
        "turn_correctness": turn_correctness,
        "answer_changed": answer_changed,
        "rule_changed": rule_changed,
        "belief_updated": answer_changed or rule_changed or response.revised_confidence > response.initial_confidence,
        "num_turns": len(turn_records),
    }

    if spec.family == "attention_distractors":
        cue_turn = int(spec.expected["cue_turn"])
        trap_signal = spec.expected["trap_signal"]
        relevant_signal = spec.expected["relevant_signal"]
        decoy_target = spec.expected["decoy_target"]
        capture_turns = []
        recovery_turn = None
        for idx, record in enumerate(turn_records, start=1):
            model_response = record["model_response"]
            captured = (
                trap_signal in model_response["attended_signals"]
                or exact_match(model_response["answer"], decoy_target)
                or relevant_signal in model_response["ignored_signals"]
            )
            if captured:
                capture_turns.append(idx)
            if idx >= cue_turn and recovery_turn is None:
                aligned = bool(record["derived"]["answer_correct"]) and bool(record["derived"]["signal_focus_correct"])
                if aligned:
                    recovery_turn = idx
        derived["cue_turn"] = cue_turn
        derived["capture_turns"] = capture_turns
        derived["recovery_turn"] = recovery_turn

    if spec.family == "social_miniworlds":
        trust_trace = [record["derived"]["observed_trust_top"] for record in turn_records]
        derived["trust_trace"] = trust_trace
        derived["belief_consistency_trace"] = [record["derived"]["belief_consistency"] for record in turn_records]

    return derived


def make_placeholder_responder(seed: int = 101) -> TurnResponder:
    """Create a deterministic baseline responder for interactive demos."""
    rng = make_rng(seed)

    def respond(spec: EpisodeSpec, turn: EpisodeTurnSpec, prior_turns: list[dict[str, Any]]) -> dict[str, Any]:
        turn_index = len(prior_turns)
        accepted = turn.accepted_answers[0] if turn.accepted_answers else None
        metadata = {"rule_tag": turn.expected_rule if not isinstance(turn.expected_rule, list) else turn.expected_rule[0]}

        if spec.family == "hidden_rule":
            if turn_index == 0:
                correct = rng.random() > 0.12
                return {
                    "answer": accepted if correct else _make_wrong_answer(accepted),
                    "confidence": turn.expected_confidence,
                    "rule_explanation": spec.expected["canonical_initial_rule"],
                    "metadata": {"rule_tag": spec.expected["canonical_initial_rule"]},
                }
            correct = rng.random() > 0.1
            return {
                "answer": accepted if correct else _make_wrong_answer(accepted),
                "confidence": turn.expected_confidence,
                "rule_explanation": spec.expected["canonical_revised_rule"],
                "evidence_acknowledged": True,
                "contradiction_detected": True,
                "revision_events": ["rule_shift_update"],
                "metadata": {"rule_tag": spec.expected["canonical_revised_rule"]},
            }

        if spec.family == "shift_transfer":
            if turn_index == 0:
                return {
                    "answer": accepted,
                    "confidence": turn.expected_confidence,
                    "rule_explanation": spec.expected["canonical_initial_rule"],
                    "metadata": {"rule_tag": spec.expected["canonical_initial_rule"]},
                }
            correct = rng.random() > 0.18
            return {
                "answer": accepted if correct else _make_wrong_answer(accepted),
                "confidence": turn.expected_confidence,
                "rule_explanation": spec.expected["canonical_revised_rule"],
                "evidence_acknowledged": True,
                "contradiction_detected": False,
                "revision_events": ["representation_remap"],
                "metadata": {"rule_tag": spec.expected["canonical_revised_rule"]},
            }

        if spec.family == "metacog_revision":
            if turn_index == 0:
                candidate_rules = spec.expected["canonical_initial_rule"]
                chosen_rule = candidate_rules[0]
                return {
                    "answer": accepted,
                    "confidence": turn.expected_confidence,
                    "rule_explanation": chosen_rule,
                    "metadata": {"rule_tag": chosen_rule},
                }
            correct = rng.random() > 0.08
            return {
                "answer": accepted if correct else turn.accepted_answers[-1],
                "confidence": turn.expected_confidence,
                "rule_explanation": spec.expected["canonical_revised_rule"],
                "evidence_acknowledged": True,
                "contradiction_detected": True,
                "revision_events": ["hypothesis_disambiguated"],
                "metadata": {"rule_tag": spec.expected["canonical_revised_rule"]},
            }

        if spec.family == "attention_distractors":
            true_target = accepted
            decoy_target = spec.expected["decoy_target"]
            cue_turn = int(spec.expected["cue_turn"])
            if turn_index == 0:
                captured = spec.expected["distractor_level"] >= 2 or rng.random() < 0.4
                return {
                    "answer": decoy_target if captured else true_target,
                    "confidence": 0.7 if captured else turn.expected_confidence,
                    "rule_explanation": spec.expected["canonical_initial_rule"],
                    "attended_signals": ["distractor_sequence"] if captured else ["signal_sequence"],
                    "ignored_signals": ["signal_sequence"] if captured else ["distractor_sequence"],
                    "revision_events": ["captured_by_distractor"] if captured else ["initial_signal_focus"],
                    "metadata": {"rule_tag": spec.expected["canonical_initial_rule"]},
                }
            if turn_index < cue_turn - 1:
                prior_captured = "distractor_sequence" in prior_turns[-1]["attended_signals"]
                captured = prior_captured or spec.expected["distractor_level"] >= 2
                return {
                    "answer": decoy_target if captured else true_target,
                    "confidence": turn.expected_confidence,
                    "rule_explanation": spec.expected["canonical_initial_rule"],
                    "evidence_acknowledged": True,
                    "contradiction_detected": captured,
                    "attended_signals": ["distractor_sequence"] if captured else ["signal_sequence"],
                    "ignored_signals": ["signal_sequence"] if captured else ["distractor_sequence"],
                    "revision_events": ["noise_escalation"],
                    "metadata": {"rule_tag": spec.expected["canonical_initial_rule"]},
                }
            if turn_index == cue_turn - 1:
                return {
                    "answer": true_target,
                    "confidence": turn.expected_confidence,
                    "rule_explanation": spec.expected["canonical_revised_rule"],
                    "evidence_acknowledged": True,
                    "contradiction_detected": True,
                    "attended_signals": ["signal_sequence"],
                    "ignored_signals": ["distractor_sequence"],
                    "revision_events": ["attention_recovery", "cue_used"],
                    "metadata": {"rule_tag": spec.expected["canonical_revised_rule"]},
                }
            return {
                "answer": true_target,
                "confidence": turn.expected_confidence,
                "rule_explanation": spec.expected["canonical_revised_rule"],
                "evidence_acknowledged": True,
                "contradiction_detected": False,
                "attended_signals": ["signal_sequence"],
                "ignored_signals": ["distractor_sequence"],
                "revision_events": ["maintained_signal_focus"],
                "metadata": {"rule_tag": spec.expected["canonical_revised_rule"]},
            }

        reliable = spec.expected["reliable_agent"]
        deceptive = spec.expected["deceptive_agent"]
        false_belief_agent = spec.expected["false_belief_agent"]
        actual_location = spec.expected["actual_location"]
        initial_location = spec.expected["initial_location"]

        if spec.episode_type == "false_belief_update":
            trust_scores = {agent: 0.4 for agent in spec.initial_context["agents"]}
            trust_scores[reliable] = 0.6
            trust_scores[deceptive] = 0.35
            if turn_index >= 2:
                trust_scores[reliable] = 0.9
                trust_scores[deceptive] = 0.1
            beliefs = turn.expected_beliefs
            answer = accepted
            if turn_index == 0:
                answer = {"actual_location": initial_location, "belief_of_false_belief_agent": initial_location}
            return {
                "answer": answer,
                "confidence": turn.expected_confidence,
                "rule_explanation": "social_reasoning",
                "evidence_acknowledged": turn_index > 0,
                "contradiction_detected": turn_index >= 2,
                "trust_scores_by_agent": trust_scores,
                "inferred_agent_beliefs": beliefs,
                "revision_events": (["belief_update", "trust_revised"] if turn_index >= 2 else ["belief_update"] if turn_index > 0 else []),
                "metadata": {"rule_tag": "social_reasoning"},
            }

        if spec.episode_type == "trust_revision":
            trust_scores = {agent: 0.35 for agent in spec.initial_context["agents"]}
            trust_scores[reliable] = 0.55 if turn_index == 0 else 0.88
            trust_scores[deceptive] = 0.58 if turn_index == 0 else 0.12
            answer = accepted
            if turn_index == 0:
                answer = {"actual_location": actual_location, "trusted_agent": deceptive}
            return {
                "answer": answer,
                "confidence": turn.expected_confidence,
                "rule_explanation": "trust_revision",
                "evidence_acknowledged": turn_index > 0,
                "contradiction_detected": turn_index >= 1,
                "trust_scores_by_agent": trust_scores,
                "inferred_agent_beliefs": turn.expected_beliefs,
                "revision_events": ["trust_revised"] if turn_index >= 1 else [],
                "metadata": {"rule_tag": "social_reasoning"},
            }

        trust_scores = {agent: 0.35 for agent in spec.initial_context["agents"]}
        trust_scores[reliable] = 0.62 if turn_index < 2 else 0.9
        trust_scores[deceptive] = 0.4 if turn_index == 0 else (0.48 if turn_index == 1 else 0.08)
        answer = accepted
        if turn_index == 1:
            answer = {"actual_location": actual_location, "trusted_agent": reliable}
        return {
            "answer": answer,
            "confidence": turn.expected_confidence,
            "rule_explanation": "intent_revision",
            "evidence_acknowledged": turn_index > 0,
            "contradiction_detected": turn_index >= 1,
            "trust_scores_by_agent": trust_scores,
            "inferred_agent_beliefs": turn.expected_beliefs,
            "revision_events": ["intent_revised"] if turn_index >= 2 else [],
            "metadata": {"rule_tag": "social_reasoning"},
        }

    return respond


def run_episode_spec(spec: EpisodeSpec, responder: TurnResponder) -> dict[str, Any]:
    """Run an interactive session from a prebuilt episode spec."""
    prior_turn_payloads: list[dict[str, Any]] = []
    turn_records: list[dict[str, Any]] = []
    turn_responses: list[TurnResponse] = []

    for turn in spec.turns:
        raw_payload = responder(spec, turn, prior_turn_payloads)
        turn_response = _build_turn_response(turn, raw_payload)
        turn_responses.append(turn_response)
        turn_payload = turn_response.to_dict()
        prior_turn_payloads.append(turn_payload)
        turn_records.append(
            {
                "turn_spec": turn.to_dict(),
                "model_response": turn_payload,
                "derived": _derive_turn_fields(turn, turn_response),
            }
        )

    first_turn = turn_responses[0]
    last_turn = turn_responses[-1]
    final_trust = {}
    for turn_response in reversed(turn_responses):
        if turn_response.trust_scores_by_agent:
            final_trust = turn_response.trust_scores_by_agent
            break
    final_beliefs = {}
    for turn_response in reversed(turn_responses):
        if turn_response.inferred_agent_beliefs:
            final_beliefs = turn_response.inferred_agent_beliefs
            break

    response = InteractiveResponse(
        task_id=spec.task_id,
        initial_answer=first_turn.answer,
        initial_confidence=first_turn.confidence,
        initial_rule_explanation=first_turn.rule_explanation,
        revised_answer=last_turn.answer,
        revised_confidence=last_turn.confidence,
        revised_rule_explanation=last_turn.rule_explanation,
        evidence_acknowledged=any(turn.evidence_acknowledged for turn in turn_responses[1:]),
        contradiction_detected=any(turn.contradiction_detected for turn in turn_responses),
        turns=turn_responses,
        turn_confidences=[turn.confidence for turn in turn_responses],
        belief_state_trace=[turn.inferred_agent_beliefs for turn in turn_responses if turn.inferred_agent_beliefs],
        attended_signals=[turn.attended_signals for turn in turn_responses],
        ignored_signals=[turn.ignored_signals for turn in turn_responses],
        trust_scores_by_agent=final_trust,
        inferred_agent_beliefs=final_beliefs,
        revision_events=[event for turn in turn_responses for event in turn.revision_events],
        metadata={
            "initial_rule_tag": first_turn.metadata.get("rule_tag"),
            "revised_rule_tag": last_turn.metadata.get("rule_tag"),
            "episode_type": spec.episode_type,
        },
    )

    return {
        "task_id": spec.task_id,
        "family": spec.family,
        "episode_type": spec.episode_type,
        "initial_context": spec.initial_context,
        "turns": turn_records,
        "expected": spec.expected,
        "response": response.to_dict(),
        "derived": _derive_session_fields(spec, response, turn_records),
    }


def run_interactive_session(task: dict[str, Any], responder: TurnResponder) -> dict[str, Any]:
    """Run an interactive session for one task."""
    spec = build_interaction_spec(task)
    return run_episode_spec(spec, responder)


def run_interactive_sessions(tasks: list[dict[str, Any]], responder_factory: Callable[[int], TurnResponder]) -> list[dict[str, Any]]:
    """Run interactive sessions for multiple tasks."""
    sessions = []
    for idx, task in enumerate(tasks):
        responder = responder_factory(idx)
        sessions.append(run_interactive_session(task, responder))
    return sessions


def summarize_interactive_sessions(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute dynamic interactive metrics over completed sessions."""
    return {
        "hypothesis_update_score": hypothesis_update_score(sessions),
        "contradiction_sensitivity": contradiction_sensitivity(sessions),
        "confidence_recalibration_score": confidence_recalibration_score(sessions),
        "online_adaptation_gain": online_adaptation_gain(sessions),
        "belief_trajectory_quality": belief_trajectory_quality(sessions),
        "attention_recovery_score": attention_recovery_score(sessions),
        "distractor_capture_rate": distractor_capture_rate(sessions),
        "cue_utilization_score": cue_utilization_score(sessions),
        "trust_revision_score": trust_revision_score(sessions),
        "belief_state_consistency": belief_state_consistency(sessions),
        "deception_sensitivity": deception_sensitivity(sessions),
        "multi_turn_adaptation_score": multi_turn_adaptation_score(sessions),
        "episode_cognitive_flexibility_score": episode_cognitive_flexibility_score(sessions),
        "num_sessions": len(sessions),
    }
