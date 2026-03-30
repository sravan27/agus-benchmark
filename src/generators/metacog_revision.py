"""Metacognitive Revision task family."""

from __future__ import annotations

from dataclasses import dataclass

from src.generators.common import RuleSpec, sample_palindromic_sequences, sample_unique_sequences
from src.schemas.task_schema import AGUSTask
from src.utils.seeds import make_rng


@dataclass(frozen=True)
class MetacogRevisionConfig:
    """Generation settings for metacognitive revision tasks."""

    count: int = 100
    seed: int = 37
    sequence_length: int = 5
    domain_size: int = 10
    ambiguous_examples: int = 3


def _is_palindrome(row: list[int]) -> bool:
    return row == list(reversed(row))


def generate_metacog_revision_tasks(cfg: MetacogRevisionConfig) -> list[dict]:
    """Generate tasks that reward uncertainty awareness and revision."""
    rng = make_rng(cfg.seed)
    tasks: list[dict] = []

    for idx in range(cfg.count):
        k = rng.randrange(1, cfg.domain_size - 1)
        candidate_a = RuleSpec(name="add_const", params={"k": k})
        candidate_b = RuleSpec(name="reverse_add", params={"k": k})
        actual_rule = candidate_b if idx % 2 == 0 else candidate_a

        ambiguous_rows = sample_palindromic_sequences(
            rng,
            cfg.ambiguous_examples,
            cfg.sequence_length,
            cfg.domain_size,
        )
        blocked = [tuple(row) for row in ambiguous_rows]
        candidate_rows: list[list[int]] = []
        while len(candidate_rows) < 2:
            proposal = sample_unique_sequences(
                rng,
                1,
                cfg.sequence_length,
                cfg.domain_size,
                blocked=blocked + [tuple(row) for row in candidate_rows],
            )[0]
            if _is_palindrome(proposal):
                continue
            candidate_rows.append(proposal)

        initial_probe = candidate_rows[0]
        disambiguating_row = candidate_rows[1]

        acceptable_initial_targets = [
            candidate_a.apply(initial_probe, cfg.domain_size),
            candidate_b.apply(initial_probe, cfg.domain_size),
        ]
        correction_output = actual_rule.apply(disambiguating_row, cfg.domain_size)
        revised_target = actual_rule.apply(initial_probe, cfg.domain_size)

        examples = [
            {
                "phase": "ambiguous_evidence",
                "input": row,
                "output": candidate_a.apply(row, cfg.domain_size),
            }
            for row in ambiguous_rows
        ] + [
            {
                "phase": "corrective_evidence",
                "input": disambiguating_row,
                "output": correction_output,
            }
        ]

        task = AGUSTask(
            task_id=f"metacog_revision_{idx:04d}",
            family="metacog_revision",
            difficulty="hard",
            context={
                "instruction": (
                    "Give an initial answer, confidence, and short rule hypothesis from the ambiguous evidence. "
                    "Then revise after seeing corrective evidence."
                ),
                "response_fields": [
                    "initial_answer",
                    "initial_confidence",
                    "initial_rule_guess",
                    "revised_answer",
                    "revised_confidence",
                    "revised_rule_guess",
                ],
            },
            examples=examples,
            query={
                "initial_query": {"input": initial_probe},
                "revision_prompt": {
                    "corrective_example": {
                        "input": disambiguating_row,
                        "output": correction_output,
                    },
                    "revise_same_input": initial_probe,
                },
            },
            answer={
                "acceptable_initial_targets": acceptable_initial_targets,
                "revised_target": revised_target,
                "should_revise": True,
            },
            metadata={
                "candidate_rules": [
                    {"name": candidate_a.name, "params": candidate_a.params},
                    {"name": candidate_b.name, "params": candidate_b.params},
                ],
                "actual_rule": {"name": actual_rule.name, "params": actual_rule.params},
                "expected_initial_certainty": 0.4,
            },
            latent_rule_summary="Two competing hypotheses fit early evidence; later evidence disambiguates them.",
            shift_type="belief_revision",
            distractor_level=0,
            scoring_notes=[
                "Initial answers may be any hypothesis consistent with the ambiguous evidence.",
                "Calibration should remain moderate before correction and improve after revision.",
            ],
        )
        tasks.append(task.to_dict())

    return tasks
