"""CLI for running a simple local scoring demo."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.scoring.metrics import exact_match
from src.scoring.evaluator import evaluate_from_paths
from src.utils.io_utils import load_json, save_json
from src.utils.seeds import make_rng


def build_demo_predictions(tasks: list[dict], seed: int = 99) -> list[dict]:
    """Create a deterministic, slightly imperfect baseline prediction set."""
    rng = make_rng(seed)
    rows = []
    for task in tasks:
        family = task["family"]
        if family == "hidden_rule":
            induction = list(task["answer"]["induction_targets"])
            shift = list(task["answer"]["shift_targets"])
            if shift:
                shift[0] = list(reversed(shift[0]))
            rows.append(
                {
                    "task_id": task["task_id"],
                    "induction_predictions": induction,
                    "shift_predictions": shift,
                }
            )
        elif family == "shift_transfer":
            rows.append(
                {
                    "task_id": task["task_id"],
                    "source_prediction": task["answer"]["source_target"],
                    "transfer_prediction": task["answer"]["transfer_target"]
                    if rng.random() > 0.15
                    else list(reversed(task["answer"]["transfer_target"])),
                }
            )
        elif family == "metacog_revision":
            acceptable = task["answer"]["acceptable_initial_targets"]
            initial_choice = acceptable[0]
            revised_target = task["answer"]["revised_target"]
            rows.append(
                {
                    "task_id": task["task_id"],
                    "initial_answer": initial_choice,
                    "initial_confidence": task["metadata"]["expected_initial_certainty"],
                    "initial_rule_guess": "candidate_rule",
                    "revised_answer": revised_target if rng.random() > 0.1 else acceptable[-1],
                    "revised_confidence": 0.9,
                    "revised_rule_guess": task["metadata"]["actual_rule"]["name"],
                }
            )
        elif family == "attention_distractors":
            target = task["answer"]["target"]
            pred = target
            if task["distractor_level"] >= 2 and rng.random() < 0.25:
                pred = task["query"]["record"]["distractor_sequence"]
            rows.append(
                {
                    "task_id": task["task_id"],
                    "prediction": pred,
                    "selected_signal": "signal_sequence" if exact_match(pred, target) else "distractor_sequence",
                }
            )
        elif family == "social_miniworlds":
            actual = task["answer"]["actual_location"]
            belief = task["answer"]["belief_of_false_belief_agent"]
            trusted = task["answer"]["most_reliable_agent"]
            wrong_location = task["metadata"]["locations"][0]
            if wrong_location == actual:
                wrong_location = task["metadata"]["locations"][1]
            rows.append(
                {
                    "task_id": task["task_id"],
                    "actual_location_prediction": actual if rng.random() > 0.08 else wrong_location,
                    "belief_prediction": belief,
                    "trust_prediction": trusted if rng.random() > 0.12 else task["metadata"]["deceiver"],
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate demo AGUS predictions.")
    parser.add_argument("--tasks", type=Path, default=Path("data/generated/agus_v1_all.json"))
    parser.add_argument("--predictions", type=Path, default=Path("data/samples/demo_predictions.json"))
    parser.add_argument("--output", type=Path, default=Path("data/samples/demo_scores.json"))
    args = parser.parse_args()

    tasks = load_json(args.tasks)
    regenerate_predictions = not args.predictions.exists()
    if not regenerate_predictions:
        existing_predictions = load_json(args.predictions)
        regenerate_predictions = len(existing_predictions) != len(tasks)

    if regenerate_predictions:
        save_json(args.predictions, build_demo_predictions(tasks))

    results = evaluate_from_paths(args.tasks, args.predictions, args.output)
    save_json(args.output, results)
    print(results)


if __name__ == "__main__":
    main()
