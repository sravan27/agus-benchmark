"""Shared rule primitives for benchmark generators."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence


@dataclass(frozen=True)
class RuleSpec:
    """A lightweight symbolic rule over integer sequences."""

    name: str
    params: dict[str, int]

    def apply(self, seq: Sequence[int], modulo: int) -> list[int]:
        """Apply the rule to a sequence."""
        if self.name == "add_const":
            k = self.params["k"]
            return [int((x + k) % modulo) for x in seq]
        if self.name == "reverse_add":
            k = self.params["k"]
            return [int((x + k) % modulo) for x in reversed(seq)]
        if self.name == "rotate_left":
            k = self.params["k"] % len(seq)
            seq_list = list(seq)
            return seq_list[k:] + seq_list[:k]
        if self.name == "index_offset":
            k = self.params["k"]
            return [int((x + (i + 1) * k) % modulo) for i, x in enumerate(seq)]
        if self.name == "mirror_anchor":
            anchor = self.params["anchor"]
            return [int((anchor - x) % modulo) for x in seq]
        raise ValueError(f"Unsupported rule: {self.name}")

    def summary(self) -> str:
        """Human-readable short description."""
        if self.name == "add_const":
            return "apply a constant modular increment"
        if self.name == "reverse_add":
            return "reverse order, then apply a constant modular increment"
        if self.name == "rotate_left":
            return "rotate the sequence left"
        if self.name == "index_offset":
            return "apply an index-weighted modular increment"
        if self.name == "mirror_anchor":
            return "mirror each element around a hidden anchor"
        return "apply a hidden transformation"


def sequence_to_tokens(seq: Sequence[int], vocab: Sequence[str]) -> list[str]:
    """Map integer ids into another representation."""
    return [vocab[x] for x in seq]


def sample_unique_sequences(
    rng: random.Random,
    count: int,
    sequence_length: int,
    domain_size: int,
    blocked: Sequence[tuple[int, ...]] | None = None,
) -> list[list[int]]:
    """Sample unique fixed-length integer sequences."""
    seen = set(blocked or [])
    rows: list[list[int]] = []
    while len(rows) < count:
        candidate = tuple(rng.randrange(domain_size) for _ in range(sequence_length))
        if candidate in seen:
            continue
        seen.add(candidate)
        rows.append(list(candidate))
    return rows


def sample_palindromic_sequences(
    rng: random.Random,
    count: int,
    sequence_length: int,
    domain_size: int,
) -> list[list[int]]:
    """Sample palindromic sequences for intentionally ambiguous tasks."""
    half = (sequence_length + 1) // 2
    rows: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    while len(rows) < count:
        left = [rng.randrange(domain_size) for _ in range(half)]
        if sequence_length % 2 == 0:
            seq = left + list(reversed(left))
        else:
            seq = left + list(reversed(left[:-1]))
        seq_tuple = tuple(seq)
        if seq_tuple in seen:
            continue
        seen.add(seq_tuple)
        rows.append(seq)
    return rows


def random_rule(
    rng: random.Random,
    *,
    allowed: Sequence[str] | None = None,
    domain_size: int = 10,
) -> RuleSpec:
    """Sample one symbolic rule."""
    names = tuple(allowed or ("add_const", "reverse_add", "rotate_left", "index_offset", "mirror_anchor"))
    name = rng.choice(names)
    if name == "add_const":
        return RuleSpec(name=name, params={"k": rng.randrange(1, domain_size - 1)})
    if name == "reverse_add":
        return RuleSpec(name=name, params={"k": rng.randrange(1, domain_size - 1)})
    if name == "rotate_left":
        return RuleSpec(name=name, params={"k": rng.randrange(1, 4)})
    if name == "index_offset":
        return RuleSpec(name=name, params={"k": rng.randrange(1, 3)})
    if name == "mirror_anchor":
        return RuleSpec(name=name, params={"anchor": rng.randrange(1, domain_size)})
    raise ValueError(f"Unsupported rule name: {name}")


def distinct_rule(
    rng: random.Random,
    base: RuleSpec,
    *,
    allowed: Sequence[str] | None = None,
    domain_size: int = 10,
) -> RuleSpec:
    """Sample a different rule."""
    while True:
        candidate = random_rule(rng, allowed=allowed, domain_size=domain_size)
        if candidate != base:
            return candidate
