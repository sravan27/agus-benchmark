"""Schemas for interactive AGUS responses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TurnResponse:
    """One model response inside a multi-turn interactive episode."""

    turn_id: str
    answer: Any
    confidence: float
    rule_explanation: str
    evidence_acknowledged: bool = False
    contradiction_detected: bool = False
    attended_signals: list[str] = field(default_factory=list)
    ignored_signals: list[str] = field(default_factory=list)
    trust_scores_by_agent: dict[str, float] = field(default_factory=dict)
    inferred_agent_beliefs: dict[str, Any] = field(default_factory=dict)
    revision_events: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the turn response."""
        return asdict(self)


@dataclass
class InteractiveResponse:
    """Aggregated response record for an interactive episode."""

    task_id: str
    initial_answer: Any
    initial_confidence: float
    initial_rule_explanation: str
    revised_answer: Any
    revised_confidence: float
    revised_rule_explanation: str
    evidence_acknowledged: bool
    contradiction_detected: bool
    turns: list[TurnResponse] = field(default_factory=list)
    turn_confidences: list[float] = field(default_factory=list)
    belief_state_trace: list[dict[str, Any]] = field(default_factory=list)
    attended_signals: list[list[str]] = field(default_factory=list)
    ignored_signals: list[list[str]] = field(default_factory=list)
    trust_scores_by_agent: dict[str, float] = field(default_factory=dict)
    inferred_agent_beliefs: dict[str, Any] = field(default_factory=dict)
    revision_events: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the response to a dictionary."""
        return asdict(self)
