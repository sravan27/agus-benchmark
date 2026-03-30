"""Shift and Transfer task family."""

from __future__ import annotations

from dataclasses import dataclass

from src.generators.common import RuleSpec, random_rule, sample_unique_sequences, sequence_to_tokens
from src.schemas.task_schema import AGUSTask
from src.utils.seeds import make_rng

BASE_VOCABS = (
    ["circle", "triangle", "square", "star", "hex", "dot", "line", "arc", "wave", "cross"],
    ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
)
TRANSFER_VOCABS = (
    ["ka", "zu", "mi", "to", "ra", "ve", "lo", "pi", "su", "ne"],
    ["amber", "blue", "cinder", "dune", "ember", "frost", "glade", "haze", "iris", "jade"],
)
BRIDGE_STEMS = (
    ["qx", "ly", "no", "te", "va", "ru", "si", "po", "me", "da"],
    ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c0"],
)


@dataclass(frozen=True)
class ShiftTransferConfig:
    """Generation settings for shift-transfer tasks."""

    count: int = 100
    seed: int = 23
    sequence_length: int = 4
    domain_size: int = 10
    examples_per_task: int = 4
    anti_template_strength: int = 1
    remap_composition_depth: int = 2
    bridge_representation_mode: str = "alias_chain"
    anti_anchor_strength: int = 1
    latent_rule_mix: str = "anchor_resistant"


def _token_overlap(left: list[str], right: list[str]) -> float:
    matches = sum(1 for a, b in zip(left, right) if a == b)
    return matches / max(len(left), len(right))


def _sample_transfer_rule(
    rng,
    *,
    domain_size: int,
    anti_anchor_strength: int,
    latent_rule_mix: str,
) -> RuleSpec:
    if latent_rule_mix == "baseline":
        allowed = ("add_const", "rotate_left", "index_offset", "mirror_anchor")
    elif latent_rule_mix == "transfer_hard":
        allowed = ("reverse_add", "mirror_anchor", "index_offset")
    else:
        allowed = ("index_offset", "mirror_anchor", "reverse_add", "rotate_left")
    if anti_anchor_strength <= 0 and latent_rule_mix != "transfer_hard":
        allowed = ("add_const", "rotate_left", "index_offset", "mirror_anchor")
    return random_rule(rng, allowed=allowed, domain_size=domain_size)


def _build_bridge_vocab(
    source_vocab: list[str],
    stems: list[str],
    task_index: int,
    *,
    mode: str,
) -> list[str] | None:
    if mode == "none":
        return None
    if mode == "attribute_bridge":
        return [
            f"{stems[idx]}-{source_vocab[(idx + task_index) % len(source_vocab)][-2:]}"
            for idx in range(len(source_vocab))
        ]
    return [
        f"{stems[idx]}::{source_vocab[(idx + task_index) % len(source_vocab)][:2]}"
        for idx in range(len(source_vocab))
    ]


def _build_transfer_vocab(
    source_vocab: list[str],
    base_transfer: list[str],
    bridge_vocab: list[str] | None,
    *,
    profile: str,
    depth: int,
    task_index: int,
) -> list[str]:
    if depth <= 1 or bridge_vocab is None:
        if profile == "attribute_bridge":
            return [
                f"{base_transfer[idx]}-{source_vocab[(idx + task_index + 1) % len(source_vocab)][:2]}"
                for idx in range(len(base_transfer))
            ]
        return list(base_transfer)

    if profile == "composed_alias":
        tokens = [
            f"{base_transfer[idx]}::{bridge_vocab[(idx + task_index) % len(bridge_vocab)].split('::')[0]}"
            for idx in range(len(base_transfer))
        ]
    else:
        tokens = [
            f"{base_transfer[idx]}-{source_vocab[(idx + task_index + 1) % len(source_vocab)][:2]}"
            for idx in range(len(base_transfer))
        ]

    if depth >= 3:
        tokens = [f"{token}|r{(idx * 2 + task_index) % 5}" for idx, token in enumerate(tokens)]
    return tokens


def _row_anchor_penalty(
    rule: RuleSpec,
    row: list[int],
    domain_size: int,
    *,
    anti_anchor_strength: int,
) -> tuple[float, int, int]:
    target = rule.apply(row, domain_size)
    identity_overlap = _token_overlap([str(value) for value in row], [str(value) for value in target])
    reverse_overlap = _token_overlap([str(value) for value in reversed(row)], [str(value) for value in target])
    differs_from_identity = 1.0 - identity_overlap
    differs_from_reverse = 1.0 - reverse_overlap
    uniqueness = len(set(target))
    anchor_weight = max(1, anti_anchor_strength)
    return ((differs_from_identity + differs_from_reverse) * anchor_weight, uniqueness, sum(target))


def _choose_query_rows(
    rows: list[list[int]],
    *,
    rule: RuleSpec,
    domain_size: int,
    anti_anchor_strength: int,
) -> tuple[list[list[int]], list[int], list[int], list[int]]:
    example_rows = rows[:4]
    source_query = rows[4]
    bridge_query = rows[5] if len(rows) > 5 else rows[4]
    transfer_query = rows[6] if len(rows) > 6 else rows[5]

    if anti_anchor_strength <= 0:
        return (example_rows, source_query, bridge_query, transfer_query)

    ranked = sorted(
        rows[4:],
        key=lambda row: _row_anchor_penalty(rule, row, domain_size, anti_anchor_strength=anti_anchor_strength),
        reverse=True,
    )
    source_query = ranked[0]
    bridge_query = ranked[1] if len(ranked) > 1 else ranked[0]
    transfer_query = ranked[2] if len(ranked) > 2 else ranked[-1]
    return (example_rows, source_query, bridge_query, transfer_query)


def generate_shift_transfer_tasks(cfg: ShiftTransferConfig) -> list[dict]:
    """Generate tasks that preserve structure across representation changes."""
    rng = make_rng(cfg.seed)
    tasks: list[dict] = []

    for idx in range(cfg.count):
        effective_anti_anchor = max(cfg.anti_anchor_strength, cfg.anti_template_strength)
        rule = _sample_transfer_rule(
            rng,
            domain_size=cfg.domain_size,
            anti_anchor_strength=effective_anti_anchor,
            latent_rule_mix=cfg.latent_rule_mix,
        )
        source_vocab = list(BASE_VOCABS[idx % len(BASE_VOCABS)])
        base_transfer = list(TRANSFER_VOCABS[idx % len(TRANSFER_VOCABS)])
        bridge_stems = list(BRIDGE_STEMS[idx % len(BRIDGE_STEMS)])
        remap_profile = "composed_alias" if cfg.bridge_representation_mode == "alias_chain" else "attribute_bridge"
        if cfg.bridge_representation_mode == "mixed":
            remap_profile = "composed_alias" if idx % 2 == 0 else "attribute_bridge"

        bridge_vocab = None
        if cfg.remap_composition_depth > 1:
            bridge_mode = cfg.bridge_representation_mode if cfg.bridge_representation_mode != "mixed" else remap_profile
            bridge_vocab = _build_bridge_vocab(source_vocab, bridge_stems, idx, mode=bridge_mode)
        transfer_vocab = _build_transfer_vocab(
            source_vocab,
            base_transfer,
            bridge_vocab,
            profile=remap_profile,
            depth=cfg.remap_composition_depth,
            task_index=idx,
        )

        row_count = cfg.examples_per_task + 5 if effective_anti_anchor > 0 else cfg.examples_per_task + 2
        rows = sample_unique_sequences(
            rng,
            row_count,
            cfg.sequence_length,
            cfg.domain_size,
        )
        example_rows, source_query, bridge_query, transfer_query = _choose_query_rows(
            rows,
            rule=rule,
            domain_size=cfg.domain_size,
            anti_anchor_strength=effective_anti_anchor,
        )

        examples = [
            {
                "phase": "learn_source_representation",
                "input": sequence_to_tokens(row, source_vocab),
                "output": sequence_to_tokens(rule.apply(row, cfg.domain_size), source_vocab),
                "surface_profile": "source_tokens",
            }
            for row in example_rows
        ]

        bridge_query_tokens = sequence_to_tokens(bridge_query, bridge_vocab) if bridge_vocab is not None else None
        bridge_target_tokens = sequence_to_tokens(rule.apply(bridge_query, cfg.domain_size), bridge_vocab) if bridge_vocab is not None else None
        source_target = sequence_to_tokens(rule.apply(source_query, cfg.domain_size), source_vocab)
        transfer_target = sequence_to_tokens(rule.apply(transfer_query, cfg.domain_size), transfer_vocab)

        transfer_overlap = _token_overlap(source_target, transfer_target)
        difficulty = (
            "hard"
            if cfg.remap_composition_depth > 1 or rule.name in {"index_offset", "mirror_anchor", "reverse_add"}
            else "medium"
        )

        query = {
            "source_query": {"input": sequence_to_tokens(source_query, source_vocab)},
            "transfer_query": {"input": sequence_to_tokens(transfer_query, transfer_vocab)},
        }
        answer = {
            "source_target": source_target,
            "transfer_target": transfer_target,
        }
        if bridge_query_tokens is not None and bridge_target_tokens is not None:
            query["bridge_query"] = {"input": bridge_query_tokens}
            answer["bridge_target"] = bridge_target_tokens

        task = AGUSTask(
            task_id=f"shift_transfer_{idx:04d}",
            family="shift_transfer",
            difficulty=difficulty,
            context={
                "instruction": (
                    "Infer the latent transformation from examples in one representation, then preserve the same "
                    "rule through one or more codebook changes."
                ),
                "source_representation": " ".join(source_vocab[:4]),
                "transfer_representation": " ".join(transfer_vocab[:4]),
                "bridge_representation": " ".join(bridge_vocab[:4]) if bridge_vocab is not None else None,
                "representation_profile": remap_profile,
            },
            examples=examples,
            query=query,
            answer=answer,
            metadata={
                "internal_rule": {"name": rule.name, "params": rule.params},
                "source_vocab": source_vocab,
                "bridge_vocab": bridge_vocab,
                "transfer_vocab": transfer_vocab,
                "remap_profile": remap_profile,
                "bridge_representation_mode": cfg.bridge_representation_mode,
                "remap_composition_depth": cfg.remap_composition_depth,
                "anti_template_strength": cfg.anti_template_strength,
                "anti_anchor_strength": effective_anti_anchor,
                "latent_rule_mix": cfg.latent_rule_mix,
                "source_transfer_overlap": round(transfer_overlap, 4),
            },
            latent_rule_summary=(
                "The latent operator stays fixed while the surface codebook shifts through composed aliases and "
                "near-miss token regularities."
            ),
            shift_type="representation_remap",
            distractor_level=0,
            scoring_notes=[
                "Exact-match the source query to verify rule learning.",
                "Transfer turns should preserve the rule while revising the representation hypothesis.",
                "Composed remaps are intended to punish source-answer anchoring.",
            ],
        )
        tasks.append(task.to_dict())

    return tasks
