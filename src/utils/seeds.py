"""Deterministic seed helpers."""

from __future__ import annotations

import random


def make_rng(seed: int) -> random.Random:
    """Return a dedicated deterministic random number generator."""
    return random.Random(seed)

