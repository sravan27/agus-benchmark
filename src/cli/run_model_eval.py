"""CLI for local-first AGUS model evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.adapters import build_adapter
from src.eval.model_runner import BALANCED_SLICE_INDEX, run_model_evaluation
from src.eval.run_profiles import RUN_PROFILES, resolve_run_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local-first AGUS model evaluation.")
    parser.add_argument("--tasks", type=Path, default=Path("data/generated/agus_v1_all.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/evals"))
    parser.add_argument("--run-name", type=str, default="mock_noisy_eval")
    parser.add_argument("--adapter", type=str, default="mock-noisy")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--host", type=str, default="http://127.0.0.1:11434")
    parser.add_argument("--keep-alive", type=str, default="30m")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--profile", type=str, choices=sorted(RUN_PROFILES), default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--balanced", action="store_true", default=None)
    parser.add_argument(
        "--balanced-slice",
        type=str,
        default="original",
        help=(
            "Deterministic balanced-slice selector. "
            f"Built-in aliases: {', '.join(sorted(BALANCED_SLICE_INDEX))}. "
            "Also supports replication_2, replication_3, or slice_2."
        ),
    )
    parser.add_argument("--per-family-max", type=int, default=None)
    parser.add_argument("--family", action="append", default=None)
    parser.add_argument("--no-interactive", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    profile_settings = resolve_run_profile(
        args.profile,
        max_tasks=args.max_tasks,
        balanced=args.balanced,
        per_family_max=args.per_family_max,
    )
    keep_alive = None if args.keep_alive.lower() in {"none", "off", "0"} else args.keep_alive

    adapter = build_adapter(
        args.adapter,
        model=args.model,
        seed=args.seed,
        host=args.host,
        temperature=args.temperature,
        timeout_seconds=args.timeout_seconds,
        keep_alive=keep_alive,
    )
    run_dir = args.output_dir / args.run_name
    results = run_model_evaluation(
        tasks_path=args.tasks,
        adapter=adapter,
        run_dir=run_dir,
        families=set(args.family) if args.family else None,
        max_tasks=profile_settings["max_tasks"],
        include_interactive=not args.no_interactive,
        resume=not args.no_resume,
        balanced=profile_settings["balanced"],
        per_family_max=profile_settings["per_family_max"],
        balanced_slice=args.balanced_slice,
    )
    print(
        {
            "run_dir": str(run_dir),
            "adapter": adapter.describe(),
            "profile": profile_settings["profile_name"],
            "balanced_slice": results["run_composition"]["balanced_slice_name"],
            "families_planned": results["run_composition"]["families_planned"],
            "tasks_planned_per_family": results["run_composition"]["tasks_planned_per_family"],
            "interactive_sessions_planned_per_family": results["run_composition"]["interactive_sessions_planned_per_family"],
            "accuracy": results["static_summary"]["accuracy"],
            "belief_trajectory_quality": results["interactive_summary"].get("belief_trajectory_quality"),
            "failure_count": results["failure_count"],
            "num_errors": results["num_errors"],
        }
    )


if __name__ == "__main__":
    main()
