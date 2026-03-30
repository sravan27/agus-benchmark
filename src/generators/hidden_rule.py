"""Hidden Rule Induction task family."""

from __future__ import annotations

from dataclasses import dataclass

from src.generators.common import distinct_rule, random_rule, sample_unique_sequences
from src.schemas.task_schema import AGUSTask
from src.utils.seeds import make_rng


@dataclass(frozen=True)
class HiddenRuleConfig:
    """Generation settings for hidden-rule tasks."""

    count: int = 100
    seed: int = 11
    sequence_length: int = 4
    domain_size: int = 10
    induction_examples: int = 3
    induction_queries: int = 2
    shift_examples: int = 2
    shift_queries: int = 4


def _difficulty_from_rule(rule_name: str, shift_queries: int) -> str:
    if rule_name in {"rotate_left", "mirror_anchor"} or shift_queries >= 4:
        return "medium"
    return "easy"


def generate_hidden_rule_tasks(cfg: HiddenRuleConfig) -> list[dict]:
    """Generate tasks where a latent rule changes midstream."""
    rng = make_rng(cfg.seed)
    tasks: list[dict] = []

    for idx in range(cfg.count):
        initial_rule = random_rule(rng, domain_size=cfg.domain_size)
        shifted_rule = distinct_rule(rng, initial_rule, domain_size=cfg.domain_size)

        total_initial = cfg.induction_examples + cfg.induction_queries
        induction_inputs = sample_unique_sequences(
            rng,
            total_initial,
            cfg.sequence_length,
            cfg.domain_size,
        )
        induction_examples = induction_inputs[: cfg.induction_examples]
        induction_queries = induction_inputs[cfg.induction_examples :]

        total_shift = cfg.shift_examples + cfg.shift_queries
        shift_inputs = sample_unique_sequences(
            rng,
            total_shift,
            cfg.sequence_length,
            cfg.domain_size,
            blocked=[tuple(row) for row in induction_inputs],
        )
        shift_examples = shift_inputs[: cfg.shift_examples]
        shift_queries = shift_inputs[cfg.shift_examples :]

        examples = [
            {
                "phase": "induction",
                "input": row,
                "output": initial_rule.apply(row, cfg.domain_size),
            }
            for row in induction_examples
        ] + [
            {
                "phase": "shift_feedback",
                "input": row,
                "output": shifted_rule.apply(row, cfg.domain_size),
            }
            for row in shift_examples
        ]

        query = {
            "induction_queries": [{"input": row} for row in induction_queries],
            "shift_queries": [{"input": row} for row in shift_queries],
        }
        answer = {
            "induction_targets": [initial_rule.apply(row, cfg.domain_size) for row in induction_queries],
            "shift_targets": [shifted_rule.apply(row, cfg.domain_size) for row in shift_queries],
        }

        task = AGUSTask(
            task_id=f"hidden_rule_{idx:04d}",
            family="hidden_rule",
            difficulty=_difficulty_from_rule(initial_rule.name, cfg.shift_queries),
            context={
                "instruction": (
                    "Infer the hidden sequence transformation from the induction examples. "
                    "Then continue after the rule unexpectedly changes."
                ),
                "symbol_space": list(range(cfg.domain_size)),
                "sequence_length": cfg.sequence_length,
            },
            examples=examples,
            query=query,
            answer=answer,
            metadata={
                "internal_rules": {
                    "induction": {"name": initial_rule.name, "params": initial_rule.params},
                    "shifted": {"name": shifted_rule.name, "params": shifted_rule.params},
                },
                "adaptation_window": cfg.shift_queries,
            },
            latent_rule_summary="A hidden sequence rule governs the examples, then changes abruptly.",
            shift_type="rule_change",
            distractor_level=0,
            scoring_notes=[
                "Score induction answers with exact sequence match.",
                "Use shift query order to measure adaptation speed after the change point.",
            ],
        )
        tasks.append(task.to_dict())

    return tasks

