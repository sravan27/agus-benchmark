"""Attention and Distractor Control task family."""

from __future__ import annotations

from dataclasses import dataclass

from src.generators.common import RuleSpec, random_rule, sample_unique_sequences
from src.schemas.task_schema import AGUSTask
from src.utils.seeds import make_rng

NOTE_BANK = {
    "reverse_salience": (
        "Visual note: the mirrored stripe is intentionally eye-catching.",
        "Archive note: analysts once overfit to the mirrored field.",
    ),
    "rotating_decoy": (
        "Structural note: a rotating side-channel sometimes looks causal.",
        "Telemetry note: the decoy field preserves motion but not the rule.",
    ),
    "anchor_repeat": (
        "Gloss note: a repeated anchor token can look predictive without being generative.",
        "Review note: static repetitions are often tempting but stale.",
    ),
    "offset_clone": (
        "Operator note: a constant shift appears in the decoy field but drifts across records.",
        "Control note: local offsets are present, but not the governing rule.",
    ),
    "stale_output": (
        "Memory note: one field echoes an earlier answer and can feel trustworthy.",
        "History note: prior outputs are present as noise, not as the current rule.",
    ),
    "mirror_noise": (
        "Pattern note: a reflected channel exists only as a distractor.",
        "Inspection note: mirrored values are decorative in this record family.",
    ),
}

DISTRACTOR_STYLE_ORDER = (
    "reverse_salience",
    "rotating_decoy",
    "anchor_repeat",
    "offset_clone",
    "stale_output",
    "mirror_noise",
)


@dataclass(frozen=True)
class AttentionDistractorConfig:
    """Generation settings for distractor-heavy tasks."""

    count: int = 100
    seed: int = 51
    sequence_length: int = 4
    domain_size: int = 10
    anti_template_strength: int = 1
    distractor_diversity_level: int = 2
    cue_delay_level: int = 1
    adversarial_query_mode: str = "confusable"


def _sequence_overlap(left: list[int], right: list[int]) -> float:
    matches = sum(1 for a, b in zip(left, right) if a == b)
    return matches / max(len(left), len(right))


def _legacy_decoy(rule_name: str, row: list[int]) -> list[int]:
    if rule_name == "add_const":
        return list(reversed(row))
    if rule_name == "rotate_left":
        return row[1:] + row[:1]
    return [row[0] for _ in row]


def _style_pool(diversity_level: int) -> tuple[str, ...]:
    if diversity_level <= 0:
        return ("reverse_salience",)
    limit = min(len(DISTRACTOR_STYLE_ORDER), max(2, diversity_level + 2))
    return DISTRACTOR_STYLE_ORDER[:limit]


def _base_shift(rule: RuleSpec, domain_size: int) -> int:
    if "k" in rule.params:
        return max(1, int(rule.params["k"]) % domain_size)
    if "anchor" in rule.params:
        return max(1, int(rule.params["anchor"]) % domain_size)
    return 1


def _make_diverse_decoy(
    style: str,
    row: list[int],
    *,
    rule: RuleSpec,
    domain_size: int,
    example_index: int,
    prior_output: list[int] | None = None,
) -> list[int]:
    shift = (_base_shift(rule, domain_size) + example_index + 1) % domain_size or 1

    if style == "reverse_salience":
        return list(reversed(row))
    if style == "rotating_decoy":
        offset = (example_index % len(row)) + 1
        return row[offset:] + row[:offset]
    if style == "anchor_repeat":
        anchor = row[(example_index + 1) % len(row)]
        return [anchor for _ in row]
    if style == "offset_clone":
        return [int((value + shift) % domain_size) for value in row]
    if style == "stale_output" and prior_output is not None:
        return list(prior_output)
    if style == "mirror_noise":
        anchor = (_base_shift(rule, domain_size) + example_index + 3) % domain_size
        return [int((anchor - value) % domain_size) for value in row]
    return _legacy_decoy(rule.name, row)


def _make_notes(style: str, distractor_level: int, example_index: int) -> list[str]:
    notes = list(NOTE_BANK[style])
    notes.append(f"Record note {example_index + 1}: clutter level {distractor_level + 1}.")
    return notes[: distractor_level + 2]


def _choose_query_row(
    example_rows: list[list[int]],
    candidates: list[list[int]],
    *,
    rule: RuleSpec,
    domain_size: int,
    anti_template_strength: int,
    adversarial_query_mode: str,
) -> tuple[list[int], float]:
    if anti_template_strength <= 0 or adversarial_query_mode == "first_candidate":
        return (candidates[0], 0.0)

    def score(row: list[int]) -> tuple[float, int, int]:
        target = rule.apply(row, domain_size)
        best_overlap = 0.0
        mismatch_bonus = 0
        unique_bonus = len(set(target))
        for example_row in example_rows:
            overlap = _sequence_overlap(row, example_row)
            example_target = rule.apply(example_row, domain_size)
            if example_target != target:
                best_overlap = max(best_overlap, overlap)
                if overlap >= 0.5:
                    mismatch_bonus = 1
        if adversarial_query_mode == "max_confusable":
            confusion_bonus = 0.4
        elif adversarial_query_mode == "contrastive_confusable":
            confusion_bonus = 0.25 if mismatch_bonus else -0.1
        else:
            confusion_bonus = 0.15
        return (
            best_overlap + mismatch_bonus + confusion_bonus + (0.1 * anti_template_strength),
            unique_bonus,
            sum(target),
        )

    indexed = list(enumerate(candidates))
    _best_idx, best_row = max(indexed, key=lambda item: score(item[1]))
    best_score = score(best_row)[0]
    return (best_row, round(best_score, 4))


def _choose_query_distractor(
    query_row: list[int],
    example_rows: list[list[int]],
    example_outputs: list[list[int]],
    *,
    styles: tuple[str, ...],
    rule: RuleSpec,
    domain_size: int,
    anti_template_strength: int,
) -> tuple[list[int], str]:
    true_target = rule.apply(query_row, domain_size)
    candidates: list[tuple[float, str, list[int]]] = []
    for style_index, style in enumerate(styles):
        prior_output = example_outputs[(style_index - 1) % len(example_outputs)]
        distractor = _make_diverse_decoy(
            style,
            query_row,
            rule=rule,
            domain_size=domain_size,
            example_index=len(example_rows) + style_index,
            prior_output=prior_output,
        )
        decoy_target = rule.apply(distractor, domain_size)
        if decoy_target == true_target:
            continue
        confusion = max(_sequence_overlap(distractor, row) for row in example_rows)
        stale_match = max(_sequence_overlap(decoy_target, output) for output in example_outputs)
        score = confusion + stale_match + (0.15 * anti_template_strength)
        candidates.append((score, style, distractor))

    if not candidates:
        style = styles[0]
        return (
            _make_diverse_decoy(
                style,
                query_row,
                rule=rule,
                domain_size=domain_size,
                example_index=len(example_rows),
                prior_output=example_outputs[0],
            ),
            style,
        )

    _score, style, distractor = max(candidates, key=lambda item: (item[0], item[1]))
    return (distractor, style)


def generate_attention_distractor_tasks(cfg: AttentionDistractorConfig) -> list[dict]:
    """Generate tasks where relevant signal is mixed with diverse but tempting distractors."""
    rng = make_rng(cfg.seed)
    tasks: list[dict] = []

    for idx in range(cfg.count):
        distractor_level = idx % 4
        allowed_rules = ("index_offset", "mirror_anchor", "rotate_left") if cfg.anti_template_strength > 0 else (
            "add_const",
            "rotate_left",
            "index_offset",
        )
        signal_rule = random_rule(
            rng,
            allowed=allowed_rules,
            domain_size=cfg.domain_size,
        )
        candidate_pool_size = 5
        if cfg.anti_template_strength > 0:
            candidate_pool_size = 10 if cfg.adversarial_query_mode == "first_candidate" else 12
        rows = sample_unique_sequences(
            rng,
            candidate_pool_size,
            cfg.sequence_length,
            cfg.domain_size,
        )
        example_rows = rows[:4]
        query_candidates = rows[4:]
        query_row, query_confusion_score = _choose_query_row(
            example_rows,
            query_candidates,
            rule=signal_rule,
            domain_size=cfg.domain_size,
            anti_template_strength=cfg.anti_template_strength,
            adversarial_query_mode=cfg.adversarial_query_mode,
        )

        styles = _style_pool(cfg.distractor_diversity_level)
        examples = []
        example_outputs = [signal_rule.apply(row, cfg.domain_size) for row in example_rows]
        example_styles: list[str] = []
        for ex_idx, row in enumerate(example_rows):
            if cfg.distractor_diversity_level <= 0:
                style = "reverse_salience"
                distractor = _legacy_decoy(signal_rule.name, row)
            else:
                style = styles[(idx + ex_idx) % len(styles)]
                prior_output = example_outputs[ex_idx - 1] if ex_idx > 0 else example_outputs[-1]
                distractor = _make_diverse_decoy(
                    style,
                    row,
                    rule=signal_rule,
                    domain_size=cfg.domain_size,
                    example_index=ex_idx,
                    prior_output=prior_output,
                )
            example_styles.append(style)
            examples.append(
                {
                    "phase": "mixed_signal",
                    "signal_sequence": row,
                    "distractor_sequence": distractor,
                    "irrelevant_notes": _make_notes(style, distractor_level, ex_idx),
                    "output": example_outputs[ex_idx],
                    "salience_bias": "distractor_sequence" if distractor_level >= 2 or style in {"stale_output", "offset_clone"} else "balanced",
                    "distractor_profile": style,
                    "example_index": ex_idx,
                }
            )

        query_distractor, query_style = _choose_query_distractor(
            query_row,
            example_rows,
            example_outputs,
            styles=styles,
            rule=signal_rule,
            domain_size=cfg.domain_size,
            anti_template_strength=cfg.anti_template_strength,
        )

        cue_delay = max(1, cfg.cue_delay_level)
        task = AGUSTask(
            task_id=f"attention_distractors_{idx:04d}",
            family="attention_distractors",
            difficulty="hard" if distractor_level >= 2 or cfg.anti_template_strength > 0 else "medium",
            context={
                "instruction": (
                    "Infer the latent transformation from prior records. "
                    "Each record contains one causally relevant signal field plus several structured distractors."
                ),
                "relevant_field_unknown_to_solver": True,
                "distractor_modes": sorted(set(example_styles)),
            },
            examples=examples,
            query={
                "record": {
                    "signal_sequence": query_row,
                    "distractor_sequence": query_distractor,
                    "irrelevant_notes": _make_notes(query_style, distractor_level, len(example_rows)),
                    "distractor_profile": query_style,
                }
            },
            answer={
                "target": signal_rule.apply(query_row, cfg.domain_size),
                "relevant_field": "signal_sequence",
            },
            metadata={
                "internal_rule": {"name": signal_rule.name, "params": signal_rule.params},
                "distractor_pattern": "mixed_decoy_bank" if cfg.distractor_diversity_level > 0 else "structured_decoy",
                "distractor_profiles": example_styles + [query_style],
                "query_confusion_score": query_confusion_score,
                "anti_template_strength": cfg.anti_template_strength,
                "distractor_diversity_level": cfg.distractor_diversity_level,
                "cue_delay_level": cue_delay,
                "adversarial_query_mode": cfg.adversarial_query_mode,
            },
            latent_rule_summary=(
                "Only the signal field follows the hidden rule; distractors mix stale outputs, mirrored structure, "
                "offset clones, and salience-heavy decoys."
            ),
            shift_type="none",
            distractor_level=distractor_level,
            scoring_notes=[
                "Primary accuracy depends on recovering the correct output.",
                "High-salience distractors are intentionally diverse and partially correlated with the answer.",
                "Interactive scoring should reward delayed recovery after the disambiguating cue.",
            ],
        )
        tasks.append(task.to_dict())

    return tasks
