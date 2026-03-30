"""Common task schema used across AGUS benchmark families."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

REQUIRED_TASK_FIELDS = (
    "task_id",
    "family",
    "difficulty",
    "context",
    "examples",
    "query",
    "answer",
    "metadata",
    "latent_rule_summary",
    "shift_type",
    "distractor_level",
    "scoring_notes",
)


@dataclass
class AGUSTask:
    """Dataclass representation of one benchmark task."""

    task_id: str
    family: str
    difficulty: str
    context: dict[str, Any]
    examples: list[dict[str, Any]]
    query: dict[str, Any]
    answer: dict[str, Any]
    metadata: dict[str, Any]
    latent_rule_summary: str
    shift_type: str | None = None
    distractor_level: int = 0
    scoring_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize task to a plain dictionary."""
        return asdict(self)

