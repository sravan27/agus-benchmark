"""Convenience run profiles for AGUS local evaluation."""

from __future__ import annotations

from typing import Any


RUN_PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {
        "description": "Fast balanced sanity check across families.",
        "max_tasks": 10,
        "balanced": True,
        "per_family_max": 2,
    },
    "balanced25": {
        "description": "Interpret-friendly balanced evaluation over 25 tasks.",
        "max_tasks": 25,
        "balanced": True,
        "per_family_max": 5,
    },
    "overnight100": {
        "description": "Longer balanced run suited for overnight local evaluation.",
        "max_tasks": 100,
        "balanced": True,
        "per_family_max": 20,
    },
}


def resolve_run_profile(
    profile_name: str | None,
    *,
    max_tasks: int | None = None,
    balanced: bool | None = None,
    per_family_max: int | None = None,
) -> dict[str, Any]:
    """Apply a named run profile, allowing explicit CLI overrides."""
    if profile_name is None:
        return {
            "profile_name": None,
            "max_tasks": max_tasks,
            "balanced": bool(balanced) if balanced is not None else False,
            "per_family_max": per_family_max,
        }

    if profile_name not in RUN_PROFILES:
        raise ValueError(f"Unknown run profile: {profile_name}")

    profile = dict(RUN_PROFILES[profile_name])
    return {
        "profile_name": profile_name,
        "max_tasks": max_tasks if max_tasks is not None else profile["max_tasks"],
        "balanced": balanced if balanced is not None else profile["balanced"],
        "per_family_max": per_family_max if per_family_max is not None else profile["per_family_max"],
    }
