from pathlib import Path

from kaggle_benchmark.packaging import (
    LEARNING_CORE_FAMILIES,
    build_manifest,
    build_slice,
    write_jsonl,
)
from kaggle_benchmark.prompts import (
    render_hidden_rule_initial_prompt,
    render_metacog_revision_prompt,
    render_shift_transfer_transfer_prompt,
)


def test_build_slice_is_balanced_for_learning_core():
    records = build_slice(per_family=2, slice_name="test_slice")

    assert len(records) == 6
    counts = {}
    for record in records:
        counts[record["family"]] = counts.get(record["family"], 0) + 1
        assert record["slice_name"] == "test_slice"

    assert counts == {
        "hidden_rule": 2,
        "shift_transfer": 2,
        "metacog_revision": 2,
    }


def test_manifest_reports_learning_core_counts(tmp_path: Path):
    records = build_slice(per_family=1, slice_name="manifest_slice")
    data_path = write_jsonl(records, tmp_path / "learning_core.jsonl")
    manifest = build_manifest(records, data_file=data_path.name)

    assert manifest["families"] == sorted(LEARNING_CORE_FAMILIES)
    assert manifest["per_family_counts"] == {
        "hidden_rule": 1,
        "metacog_revision": 1,
        "shift_transfer": 1,
    }
    assert manifest["total_rows"] == 3


def test_prompt_renderers_include_family_specific_context():
    records = build_slice(per_family=1)
    by_family = {record["family"]: record for record in records}

    hidden_prompt = render_hidden_rule_initial_prompt(by_family["hidden_rule"])
    shift_prompt = render_shift_transfer_transfer_prompt(by_family["shift_transfer"])
    metacog_prompt = render_metacog_revision_prompt(by_family["metacog_revision"])

    assert "Induction examples" in hidden_prompt
    assert "Transfer query" in shift_prompt
    assert "contradiction_detected" in metacog_prompt
