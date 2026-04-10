import json
from pathlib import Path
from typing import Any

from kaggle_benchmark.packaging import DATA_ROOT, load_jsonl
from kaggle_benchmark.prompts import (
    render_hidden_rule_initial_prompt,
    render_hidden_rule_revision_prompt,
    render_metacog_initial_prompt,
    render_metacog_revision_prompt,
    render_shift_transfer_source_prompt,
    render_shift_transfer_transfer_prompt,
)

try:
    import pandas as pd
    import kaggle_benchmarks as kbench
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - this module is designed for Kaggle runtime.
    pd = None
    kbench = None
    BaseModel = object
    Field = None


def default_slice_path() -> Path:
    return DATA_ROOT / "learning_core_v1.jsonl"


def _assert_unit_interval(value: float) -> None:
    kbench.assertions.assert_true(
        0.0 <= value <= 1.0,
        expectation="Confidence must be a real number between 0.0 and 1.0.",
    )


if kbench is not None:

    class SequenceBatchResponse(BaseModel):
        rule_hypothesis: str = Field(min_length=1)
        confidence: float
        predictions: list[list[int]]


    class TokenSequenceResponse(BaseModel):
        rule_hypothesis: str = Field(min_length=1)
        confidence: float
        prediction: list[str]


    class MetacogInitialResponse(BaseModel):
        answer: list[int]
        confidence: float
        rule_hypothesis: str = Field(min_length=1)


    class MetacogRevisedResponse(BaseModel):
        answer: list[int]
        confidence: float
        rule_hypothesis: str = Field(min_length=1)
        contradiction_detected: bool


    @kbench.task(
        name="agus_hidden_rule_episode_v1",
        description="AGUS hidden-rule induction and shift adaptation episode.",
        store_task=False,
        store_run=False,
    )
    def hidden_rule_episode(
        llm,
        task_id: str,
        family: str,
        induction_examples: list[dict[str, Any]],
        shift_feedback_examples: list[dict[str, Any]],
        induction_queries: list[dict[str, Any]],
        induction_targets: list[list[int]],
        shift_queries: list[dict[str, Any]],
        shift_targets: list[list[int]],
        instruction: str,
        sequence_length: int,
        symbol_space: list[int],
        difficulty: str,
        slice_name: str,
    ) -> bool:
        row = {
            "task_id": task_id,
            "family": family,
            "induction_examples": induction_examples,
            "shift_feedback_examples": shift_feedback_examples,
            "induction_queries": induction_queries,
            "shift_queries": shift_queries,
            "instruction": instruction,
            "sequence_length": sequence_length,
            "symbol_space": symbol_space,
            "difficulty": difficulty,
            "slice_name": slice_name,
        }
        initial = llm.prompt(render_hidden_rule_initial_prompt(row), schema=SequenceBatchResponse)
        _assert_unit_interval(initial.confidence)
        kbench.assertions.assert_equal(
            induction_targets,
            initial.predictions,
            expectation="Initial induction predictions should exactly match the hidden-rule targets.",
        )
        revised = llm.prompt(render_hidden_rule_revision_prompt(row), schema=SequenceBatchResponse)
        _assert_unit_interval(revised.confidence)
        kbench.assertions.assert_equal(
            shift_targets,
            revised.predictions,
            expectation="Revised predictions should exactly match the shifted hidden-rule targets.",
        )
        return True


    @kbench.task(
        name="agus_shift_transfer_episode_v1",
        description="AGUS shift-transfer learning episode.",
        store_task=False,
        store_run=False,
    )
    def shift_transfer_episode(
        llm,
        task_id: str,
        family: str,
        source_examples: list[dict[str, Any]],
        source_query: dict[str, Any],
        source_target: list[str],
        transfer_query: dict[str, Any],
        transfer_target: list[str],
        instruction: str,
        source_representation: str,
        transfer_representation: str,
        difficulty: str,
        slice_name: str,
    ) -> bool:
        row = {
            "task_id": task_id,
            "family": family,
            "source_examples": source_examples,
            "source_query": source_query,
            "transfer_query": transfer_query,
            "instruction": instruction,
            "source_representation": source_representation,
            "transfer_representation": transfer_representation,
            "difficulty": difficulty,
            "slice_name": slice_name,
        }
        source = llm.prompt(render_shift_transfer_source_prompt(row), schema=TokenSequenceResponse)
        _assert_unit_interval(source.confidence)
        kbench.assertions.assert_equal(
            source_target,
            source.prediction,
            expectation="Source-representation prediction should exactly match the target sequence.",
        )
        transfer = llm.prompt(
            render_shift_transfer_transfer_prompt(row), schema=TokenSequenceResponse
        )
        _assert_unit_interval(transfer.confidence)
        kbench.assertions.assert_equal(
            transfer_target,
            transfer.prediction,
            expectation="Transfer-representation prediction should exactly match the remapped target sequence.",
        )
        return True


    @kbench.task(
        name="agus_metacog_revision_episode_v1",
        description="AGUS metacognitive revision episode.",
        store_task=False,
        store_run=False,
    )
    def metacog_revision_episode(
        llm,
        task_id: str,
        family: str,
        ambiguous_examples: list[dict[str, Any]],
        corrective_examples: list[dict[str, Any]],
        initial_query: dict[str, Any],
        acceptable_initial_targets: list[list[int]],
        revision_prompt: dict[str, Any],
        revised_target: list[int],
        should_revise: bool,
        expected_initial_certainty: float,
        instruction: str,
        difficulty: str,
        slice_name: str,
    ) -> bool:
        row = {
            "task_id": task_id,
            "family": family,
            "ambiguous_examples": ambiguous_examples,
            "corrective_examples": corrective_examples,
            "initial_query": initial_query,
            "revision_prompt": revision_prompt,
            "instruction": instruction,
            "difficulty": difficulty,
            "slice_name": slice_name,
        }
        initial = llm.prompt(render_metacog_initial_prompt(row), schema=MetacogInitialResponse)
        _assert_unit_interval(initial.confidence)
        kbench.assertions.assert_true(
            initial.answer in acceptable_initial_targets,
            expectation="Initial metacognitive answer should remain consistent with the ambiguous evidence.",
        )
        kbench.assertions.assert_true(
            initial.confidence <= max(expected_initial_certainty + 0.35, 0.75),
            expectation="Initial confidence should stay moderate under ambiguous evidence.",
        )
        revised = llm.prompt(render_metacog_revision_prompt(row), schema=MetacogRevisedResponse)
        _assert_unit_interval(revised.confidence)
        kbench.assertions.assert_equal(
            revised_target,
            revised.answer,
            expectation="Revised metacognitive answer should match the disambiguated target.",
        )
        if should_revise:
            kbench.assertions.assert_true(
                revised.contradiction_detected,
                expectation="The model should explicitly acknowledge the contradiction when corrective evidence arrives.",
            )
        return True


    @kbench.task(
        name="agus_learning_track_v1",
        description=(
            "AGUS Learning-track benchmark slice for Kaggle: hidden-rule adaptation, "
            "representation transfer, and metacognitive revision."
        ),
    )
    def agus_learning_track_v1(llm, slice_path: str) -> float:
        rows = load_jsonl(slice_path)
        df = pd.DataFrame(rows)
        family_task_map = {
            "hidden_rule": hidden_rule_episode,
            "shift_transfer": shift_transfer_episode,
            "metacog_revision": metacog_revision_episode,
        }

        total_runs = 0
        total_passed = 0
        family_scores: dict[str, dict[str, float | int]] = {}

        for family, task_fn in family_task_map.items():
            family_df = df[df["family"] == family].reset_index(drop=True)
            if family_df.empty:
                continue
            runs = task_fn.evaluate(
                evaluation_data=family_df,
                grid={"llm": [llm]},
            )
            passed = sum(run.passed for run in runs)
            total = len(runs)
            total_runs += total
            total_passed += passed
            family_scores[family] = {
                "count": total,
                "passed": passed,
                "score": round(passed / total, 4) if total else 0.0,
            }

        summary = {
            "benchmark_name": "agus_learning_track_v1",
            "slice_path": slice_path,
            "families": family_scores,
            "overall_score": round(total_passed / total_runs, 4) if total_runs else 0.0,
            "total_runs": total_runs,
        }
        print(json.dumps(summary, indent=2))
        return float(summary["overall_score"])


def run_learning_track_benchmark(slice_path: str | Path | None = None):
    if kbench is None:
        raise ImportError(
            "kaggle_benchmarks is not installed in this environment. "
            "Run this module inside a Kaggle benchmark notebook created from "
            "https://www.kaggle.com/benchmarks/tasks/new ."
        )
    resolved = Path(slice_path) if slice_path else default_slice_path()
    return agus_learning_track_v1.run(llm=kbench.llm, slice_path=str(resolved))
