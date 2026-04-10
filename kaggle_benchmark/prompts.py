from __future__ import annotations

import json
from typing import Any


def _json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True)


def _strict_json_contract(fields: list[str]) -> list[str]:
    return [
        "Output exactly one valid JSON object.",
        "Do not output reasoning, analysis, notes, markdown, code fences, comments, or any text before or after the JSON.",
        "Do not wrap the JSON in backticks.",
        f"Use exactly these top-level fields: {', '.join(fields)}.",
        "All required fields must be present.",
    ]


def render_hidden_rule_initial_prompt(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You are solving an AGUS Learning-track hidden-rule task.",
            row["instruction"],
            "Infer the rule from the induction examples, then answer the induction queries.",
            *_strict_json_contract(["rule_hypothesis", "confidence", "predictions"]),
            f"Sequence length: {row['sequence_length']}",
            f"Symbol space: {_json_block(row['symbol_space'])}",
            f"Induction examples: {_json_block(row['induction_examples'])}",
            f"Induction queries: {_json_block(row['induction_queries'])}",
        ]
    )


def render_hidden_rule_revision_prompt(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            "New evidence shows that the earlier rule no longer holds.",
            "Revise your rule hypothesis and answer the shifted queries.",
            *_strict_json_contract(["rule_hypothesis", "confidence", "predictions"]),
            f"Shift feedback examples: {_json_block(row['shift_feedback_examples'])}",
            f"Shift queries: {_json_block(row['shift_queries'])}",
        ]
    )


def render_shift_transfer_source_prompt(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You are solving an AGUS Learning-track shift-transfer task.",
            row["instruction"],
            "Learn the latent rule in the source representation, then answer the source query.",
            *_strict_json_contract(["rule_hypothesis", "confidence", "prediction"]),
            f"Source representation note: {row['source_representation']}",
            f"Source examples: {_json_block(row['source_examples'])}",
            f"Source query: {_json_block(row['source_query'])}",
        ]
    )


def render_shift_transfer_transfer_prompt(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Keep the same latent rule, but update your surface representation hypothesis.",
            "Answer the transfer query in the remapped representation.",
            *_strict_json_contract(["rule_hypothesis", "confidence", "prediction"]),
            f"Transfer representation note: {row['transfer_representation']}",
            f"Transfer query: {_json_block(row['transfer_query'])}",
        ]
    )


def render_metacog_initial_prompt(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You are solving an AGUS Learning-track metacognitive revision task.",
            row["instruction"],
            "Use the ambiguous examples to make an initial prediction.",
            *_strict_json_contract(["answer", "confidence", "rule_hypothesis"]),
            f"Ambiguous examples: {_json_block(row['ambiguous_examples'])}",
            f"Initial query: {_json_block(row['initial_query'])}",
        ]
    )


def render_metacog_revision_prompt(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You now receive corrective evidence.",
            "Revise your answer if needed, and indicate whether you detected a contradiction.",
            *_strict_json_contract(
                ["answer", "confidence", "rule_hypothesis", "contradiction_detected"]
            ),
            f"Corrective examples: {_json_block(row['corrective_examples'])}",
            f"Revision prompt: {_json_block(row['revision_prompt'])}",
        ]
    )
