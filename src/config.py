"""Project-level configuration for AGUS benchmark generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class FamilySpec:
    """Configuration for one task family."""

    name: str
    count: int
    seed: int


@dataclass(frozen=True)
class BenchmarkConfig:
    """Default benchmark configuration."""

    project_root: Path
    generated_dir: Path
    sample_dir: Path
    family_specs: tuple[FamilySpec, ...] = field(
        default_factory=lambda: (
            FamilySpec(name="hidden_rule", count=100, seed=11),
            FamilySpec(name="shift_transfer", count=100, seed=23),
            FamilySpec(name="metacog_revision", count=100, seed=37),
            FamilySpec(name="attention_distractors", count=100, seed=51),
            FamilySpec(name="social_miniworlds", count=100, seed=67),
        )
    )


def default_config(project_root: Path) -> BenchmarkConfig:
    """Construct default project paths and family settings."""
    return BenchmarkConfig(
        project_root=project_root,
        generated_dir=project_root / "data" / "generated",
        sample_dir=project_root / "data" / "samples",
    )
