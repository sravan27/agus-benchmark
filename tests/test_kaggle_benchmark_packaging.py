from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from kaggle_benchmark import benchmark_tasks
from kaggle_benchmark.agus_learning_track_notebook import resolve_slice_path, run_notebook_entrypoint
from kaggle_benchmark.packaging import (
    LEARNING_CORE_FAMILIES,
    PACKAGED_SLICE_FILE,
    build_manifest,
    build_slice,
    load_jsonl,
    write_jsonl,
)
from kaggle_benchmark.prompts import (
    render_hidden_rule_initial_prompt,
    render_metacog_revision_prompt,
    render_shift_transfer_transfer_prompt,
)
from kaggle_benchmark.structured_output import parse_structured_response, prompt_for_schema


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


def test_kaggle_task_functions_use_bool_return_type_for_sdk_inference():
    source = Path(benchmark_tasks.__file__).read_text(encoding="utf-8")

    assert "def hidden_rule_episode(" in source
    assert "def shift_transfer_episode(" in source
    assert "def metacog_revision_episode(" in source
    assert ") -> bool:" in source
    assert source.count("return True") >= 3


def test_packaged_jsonl_matches_family_task_columns():
    rows = load_jsonl(PACKAGED_SLICE_FILE)
    task_map = {
        "hidden_rule": benchmark_tasks.hidden_rule_episode,
        "shift_transfer": benchmark_tasks.shift_transfer_episode,
        "metacog_revision": benchmark_tasks.metacog_revision_episode,
    }

    for family, task_fn in task_map.items():
        row = next(record for record in rows if record["family"] == family)
        expected_columns = set(benchmark_tasks._task_parameter_names(task_fn))
        assert set(row.keys()) == expected_columns


def test_prepare_family_evaluation_data_filters_extra_columns():
    rows = build_slice(per_family=1)
    import pandas as pd

    df = pd.DataFrame(rows)
    df["extra_column"] = "ignore_me"
    prepared = benchmark_tasks._prepare_family_evaluation_data(
        df,
        "hidden_rule",
        benchmark_tasks.hidden_rule_episode,
    )

    assert "extra_column" not in prepared.columns
    assert list(prepared.columns) == benchmark_tasks._task_parameter_names(
        benchmark_tasks.hidden_rule_episode
    )


def test_notebook_entrypoint_is_callable_without_argparse(monkeypatch):
    payload = {}

    class DummyRun:
        result = 0.75

    def _fake_runner(slice_path):
        payload["slice_path"] = str(slice_path)
        return DummyRun()

    monkeypatch.setattr(
        "kaggle_benchmark.agus_learning_track_notebook.run_learning_track_benchmark",
        _fake_runner,
    )

    resolved = resolve_slice_path()
    result = run_notebook_entrypoint()

    assert payload["slice_path"] == str(resolved)
    assert result["benchmark_name"] == "agus_learning_track_v1"
    assert result["score"] == 0.75


def test_package_only_import_works_without_repo_generated_sources(tmp_path: Path):
    package_src = Path("kaggle_benchmark")
    package_copy = tmp_path / "kaggle_benchmark"
    shutil.copytree(package_src, package_copy)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(tmp_path)!r}); "
                "from kaggle_benchmark.packaging import build_slice; "
                "rows = build_slice(per_family=1, slice_name='kaggle_test'); "
                "assert len(rows) == 3; "
                "assert sorted({row['family'] for row in rows}) == ['hidden_rule','metacog_revision','shift_transfer']; "
                "assert all(row['slice_name'] == 'kaggle_test' for row in rows)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


class _DummyLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def prompt(self, prompt, schema=str):
        self.calls.append({"prompt": prompt, "schema": schema})
        return self._responses.pop(0)


def test_parse_structured_response_recovers_leading_reasoning_then_json():
    raw = (
        "I will think briefly and then answer.\n"
        '{"rule_hypothesis":"reverse the sequence","confidence":0.5,"predictions":[[3,2,1]]}'
    )

    parsed = parse_structured_response(raw, benchmark_tasks.SequenceBatchResponse)

    assert parsed.rule_hypothesis == "reverse the sequence"
    assert parsed.confidence == 0.5
    assert parsed.predictions == [[3, 2, 1]]


def test_parse_structured_response_recovers_code_fenced_json():
    raw = (
        "```json\n"
        '{"rule_hypothesis":"copy odd positions","confidence":0.75,"prediction":["A","C"]}'
        "\n```"
    )

    parsed = parse_structured_response(raw, benchmark_tasks.TokenSequenceResponse)

    assert parsed.rule_hypothesis == "copy odd positions"
    assert parsed.confidence == 0.75
    assert parsed.prediction == ["A", "C"]


def test_parse_structured_response_strips_control_characters():
    raw = (
        "\x00\x1f"
        '{"answer":[1,0,1],"confidence":0.4,"rule_hypothesis":"alternating","contradiction_detected":true}'
        "\x07"
    )

    parsed = parse_structured_response(raw, benchmark_tasks.MetacogRevisedResponse)

    assert parsed.answer == [1, 0, 1]
    assert parsed.confidence == 0.4
    assert parsed.contradiction_detected is True


def test_parse_structured_response_fails_cleanly_on_unrecoverable_output():
    raw = "analysis: maybe reverse it\nnot actually json at all"

    with pytest.raises(ValueError, match="Could not recover a valid JSON object"):
        parse_structured_response(raw, benchmark_tasks.SequenceBatchResponse)


def test_prompt_for_schema_uses_raw_text_prompt_then_recovers_json():
    llm = _DummyLLM(
        [
            'Reasoning omitted.\n{"answer":[2,2],"confidence":0.6,"rule_hypothesis":"repeat value"}'
        ]
    )

    parsed = prompt_for_schema(llm, "prompt text", benchmark_tasks.MetacogInitialResponse)

    assert llm.calls[0]["schema"] is str
    assert parsed.answer == [2, 2]
    assert parsed.confidence == 0.6
