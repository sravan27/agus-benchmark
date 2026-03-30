"""JSON and JSONL helpers for dataset export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def ensure_parent(path: Path) -> None:
    """Create the parent directory if it does not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Any) -> None:
    """Save one JSON payload."""
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Save iterable rows to JSONL."""
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def load_json(path: Path) -> Any:
    """Load a JSON payload."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

