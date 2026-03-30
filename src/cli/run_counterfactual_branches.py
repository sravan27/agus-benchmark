"""CLI for AGUS v2 counterfactual branching episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.adapters import build_adapter
from src.eval.counterfactual_branching import (
    CounterfactualProgressTracker,
    SUPPORTED_COUNTERFACTUAL_FAMILIES,
    evaluate_counterfactual_bundles,
    generate_counterfactual_bundles,
    save_counterfactual_bundles,
    write_counterfactual_artifacts,
)
from src.utils.io_utils import load_json


def _format_eta(value: float | None) -> str:
    if value is None:
        return "ETA ?"
    return f"ETA {value:.1f}s"


def _print_progress(payload: dict) -> None:
    total_bundles = max(int(payload["total_bundles"]), 1)
    total_branches = max(int(payload["total_branches"]), 1)
    completed = int(payload["branches_completed"])
    percentage = (completed / total_branches) * 100.0
    line = (
        f"[counterfactual] bundles {payload['bundles_completed']}/{total_bundles} | "
        f"branches {payload['branches_completed']}/{total_branches} | "
        f"{percentage:.1f}% | elapsed {payload['elapsed_seconds']:.1f}s | "
        f"avg/bundle {payload['avg_seconds_per_bundle']:.2f}s | "
        f"{_format_eta(payload['eta_seconds'])} | errors {payload['num_errors']}"
    )
    print(line, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and score AGUS counterfactual branch episodes.")
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/generated/agus_v1_all.json"),
        help="Path to generated AGUS tasks.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="agus_v2_counterfactual_demo",
        help="Artifact prefix for generated branches and evaluation outputs.",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="mock-noisy",
        help="Adapter name: mock-noisy, mock-oracle, mock-shallow, or ollama.",
    )
    parser.add_argument("--model", type=str, default=None, help="Local model name when --adapter ollama is used.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--host", type=str, default="http://127.0.0.1:11434")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--keep-alive", type=str, default="30m")
    parser.add_argument(
        "--families",
        nargs="+",
        default=sorted(SUPPORTED_COUNTERFACTUAL_FAMILIES),
        help="Families to branch.",
    )
    parser.add_argument("--max-per-family", type=int, default=2)
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()

    tasks = load_json(args.tasks)
    families = set(args.families)
    bundles = generate_counterfactual_bundles(tasks, families=families, max_per_family=args.max_per_family)

    branches_path = Path("data/generated/branches") / f"{args.run_name}.json"
    save_counterfactual_bundles(branches_path, bundles)

    payload = {
        "run_name": args.run_name,
        "branches_path": str(branches_path),
        "num_bundles": len(bundles),
        "num_branches": sum(len(bundle.branches) for bundle in bundles),
        "families": sorted({bundle.family for bundle in bundles}),
    }

    if not args.generate_only:
        keep_alive = None if args.keep_alive.lower() == "none" else args.keep_alive
        adapter = build_adapter(
            args.adapter,
            model=args.model,
            seed=args.seed,
            host=args.host,
            temperature=args.temperature,
            timeout_seconds=args.timeout_seconds,
            keep_alive=keep_alive,
        )
        run_dir = Path("data/evals") / args.run_name
        tracker = CounterfactualProgressTracker(
            run_name=args.run_name,
            run_dir=run_dir,
            adapter_description=adapter.describe(),
            families=payload["families"],
            total_bundles=payload["num_bundles"],
            total_branches=payload["num_branches"],
        )
        initial_progress = tracker.update(
            bundles_completed=0,
            branches_completed=0,
            num_errors=0,
            run_complete=False,
        )
        _print_progress(initial_progress)

        def progress_callback(
            bundles_completed: int,
            branches_completed: int,
            num_errors: int,
            run_complete: bool,
        ) -> None:
            progress_payload = tracker.update(
                bundles_completed=bundles_completed,
                branches_completed=branches_completed,
                num_errors=num_errors,
                run_complete=run_complete,
            )
            _print_progress(progress_payload)

        summary = evaluate_counterfactual_bundles(
            bundles,
            adapter.respond_turn,
            run_name=args.run_name,
            adapter_description=adapter.describe(),
            progress_callback=progress_callback,
        )
        write_counterfactual_artifacts(run_dir, summary)
        final_progress = tracker.update(
            bundles_completed=payload["num_bundles"],
            branches_completed=payload["num_branches"],
            num_errors=0,
            run_complete=True,
        )
        _print_progress(final_progress)
        payload["evaluation_dir"] = str(run_dir)
        payload["overall_metrics"] = summary["overall_metrics"]

    print(payload)


if __name__ == "__main__":
    main()
