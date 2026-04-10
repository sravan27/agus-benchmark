from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
DATA_ROOT = PACKAGE_ROOT / "data"
PACKAGED_SLICE_FILE = DATA_ROOT / "learning_core_v1.jsonl"
PACKAGED_MANIFEST_FILE = DATA_ROOT / "manifest.json"

LEARNING_CORE_FAMILIES = (
    "hidden_rule",
    "shift_transfer",
    "metacog_revision",
)

FAMILY_SOURCE_FILES = {
    "hidden_rule": REPO_ROOT / "data" / "generated" / "hidden_rule.json",
    "shift_transfer": REPO_ROOT / "data" / "generated" / "shift_transfer.json",
    "metacog_revision": REPO_ROOT / "data" / "generated" / "metacog_revision.json",
    "attention_distractors": REPO_ROOT
    / "data"
    / "generated"
    / "attention_distractors.json",
    "social_miniworlds": REPO_ROOT
    / "data"
    / "generated"
    / "social_miniworlds.json",
}


def _load_family_source(family: str) -> list[dict[str, Any]]:
    path = FAMILY_SOURCE_FILES[family]
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _has_source_files(families: tuple[str, ...]) -> bool:
    return all(FAMILY_SOURCE_FILES[family].exists() for family in families)


def _load_packaged_rows() -> list[dict[str, Any]]:
    return load_jsonl(PACKAGED_SLICE_FILE)


def _build_from_packaged_slice(
    families: tuple[str, ...],
    *,
    per_family: int,
    offset: int,
    slice_name: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    packaged_rows = _load_packaged_rows()
    for family in families:
        family_rows = [row for row in packaged_rows if row["family"] == family]
        selected_rows = family_rows[offset : offset + per_family]
        for row in selected_rows:
            payload = dict(row)
            payload["slice_name"] = slice_name
            records.append(payload)
    return records


def _normalize_hidden_rule(task: dict[str, Any], slice_name: str) -> dict[str, Any]:
    induction_examples = [e for e in task["examples"] if e["phase"] == "induction"]
    shift_feedback = [e for e in task["examples"] if e["phase"] == "shift_feedback"]
    return {
        "task_id": task["task_id"],
        "family": task["family"],
        "difficulty": task["difficulty"],
        "slice_name": slice_name,
        "instruction": task["context"]["instruction"],
        "induction_examples": induction_examples,
        "shift_feedback_examples": shift_feedback,
        "induction_queries": task["query"]["induction_queries"],
        "induction_targets": task["answer"]["induction_targets"],
        "shift_queries": task["query"]["shift_queries"],
        "shift_targets": task["answer"]["shift_targets"],
        "sequence_length": task["context"]["sequence_length"],
        "symbol_space": task["context"]["symbol_space"],
    }


def _normalize_shift_transfer(
    task: dict[str, Any], slice_name: str
) -> dict[str, Any]:
    source_examples = [
        e for e in task["examples"] if e["phase"] == "learn_source_representation"
    ]
    return {
        "task_id": task["task_id"],
        "family": task["family"],
        "difficulty": task["difficulty"],
        "slice_name": slice_name,
        "instruction": task["context"]["instruction"],
        "source_examples": source_examples,
        "source_query": task["query"]["source_query"],
        "source_target": task["answer"]["source_target"],
        "transfer_query": task["query"]["transfer_query"],
        "transfer_target": task["answer"]["transfer_target"],
        "source_representation": task["context"]["source_representation"],
        "transfer_representation": task["context"]["transfer_representation"],
    }


def _normalize_metacog_revision(
    task: dict[str, Any], slice_name: str
) -> dict[str, Any]:
    ambiguous_examples = [
        e for e in task["examples"] if e["phase"] == "ambiguous_evidence"
    ]
    corrective_examples = [
        e for e in task["examples"] if e["phase"] == "corrective_evidence"
    ]
    return {
        "task_id": task["task_id"],
        "family": task["family"],
        "difficulty": task["difficulty"],
        "slice_name": slice_name,
        "instruction": task["context"]["instruction"],
        "ambiguous_examples": ambiguous_examples,
        "corrective_examples": corrective_examples,
        "initial_query": task["query"]["initial_query"],
        "acceptable_initial_targets": task["answer"]["acceptable_initial_targets"],
        "revision_prompt": task["query"]["revision_prompt"],
        "revised_target": task["answer"]["revised_target"],
        "should_revise": task["answer"]["should_revise"],
        "expected_initial_certainty": task["metadata"]["expected_initial_certainty"],
    }


NORMALIZERS = {
    "hidden_rule": _normalize_hidden_rule,
    "shift_transfer": _normalize_shift_transfer,
    "metacog_revision": _normalize_metacog_revision,
}


def build_slice(
    families: tuple[str, ...] = LEARNING_CORE_FAMILIES,
    *,
    per_family: int = 10,
    offset: int = 0,
    slice_name: str = "learning_core_v1",
) -> list[dict[str, Any]]:
    if not _has_source_files(families):
        return _build_from_packaged_slice(
            families,
            per_family=per_family,
            offset=offset,
            slice_name=slice_name,
        )

    records: list[dict[str, Any]] = []
    for family in families:
        items = _load_family_source(family)[offset : offset + per_family]
        for item in items:
            records.append(NORMALIZERS[family](item, slice_name))
    return records


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")
    return path


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_manifest(records: list[dict[str, Any]], data_file: str) -> dict[str, Any]:
    family_counts = Counter(record["family"] for record in records)
    return {
        "package_name": "agus_kaggle_benchmark_v1",
        "benchmark_name": "agus_learning_track_v1",
        "slice_name": records[0]["slice_name"] if records else None,
        "families": sorted(family_counts),
        "per_family_counts": dict(sorted(family_counts.items())),
        "total_rows": len(records),
        "data_file": data_file,
        "source_files": {
            family: str(FAMILY_SOURCE_FILES[family].relative_to(REPO_ROOT))
            for family in family_counts
        },
    }


def write_default_package() -> dict[str, Any]:
    records = build_slice()
    data_path = write_jsonl(records, DATA_ROOT / "learning_core_v1.jsonl")
    manifest = build_manifest(records, data_file=str(data_path.relative_to(PACKAGE_ROOT)))
    manifest_path = DATA_ROOT / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
