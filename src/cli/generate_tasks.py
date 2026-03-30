"""CLI for generating AGUS task datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import default_config
from src.generators.attention_distractors import AttentionDistractorConfig, generate_attention_distractor_tasks
from src.generators.hidden_rule import HiddenRuleConfig, generate_hidden_rule_tasks
from src.generators.metacog_revision import MetacogRevisionConfig, generate_metacog_revision_tasks
from src.generators.social_miniworlds import SocialMiniworldConfig, generate_social_miniworld_tasks
from src.generators.shift_transfer import ShiftTransferConfig, generate_shift_transfer_tasks
from src.utils.io_utils import save_json, save_jsonl
from src.utils.validation import validate_tasks


def generate_all(project_root: Path, count_per_family: int) -> dict[str, list[dict]]:
    """Generate all currently implemented families."""
    cfg = default_config(project_root)
    datasets = {
        "hidden_rule": generate_hidden_rule_tasks(HiddenRuleConfig(count=count_per_family, seed=cfg.family_specs[0].seed)),
        "shift_transfer": generate_shift_transfer_tasks(
            ShiftTransferConfig(count=count_per_family, seed=cfg.family_specs[1].seed)
        ),
        "metacog_revision": generate_metacog_revision_tasks(
            MetacogRevisionConfig(count=count_per_family, seed=cfg.family_specs[2].seed)
        ),
        "attention_distractors": generate_attention_distractor_tasks(
            AttentionDistractorConfig(count=count_per_family, seed=cfg.family_specs[3].seed)
        ),
        "social_miniworlds": generate_social_miniworld_tasks(
            SocialMiniworldConfig(count=count_per_family, seed=cfg.family_specs[4].seed)
        ),
    }
    for tasks in datasets.values():
        validate_tasks(tasks)
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AGUS benchmark tasks.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--count-per-family", type=int, default=100)
    args = parser.parse_args()

    datasets = generate_all(args.project_root, args.count_per_family)
    generated_dir = args.project_root / "data" / "generated"

    combined = []
    for family, rows in datasets.items():
        save_json(generated_dir / f"{family}.json", rows)
        save_jsonl(generated_dir / f"{family}.jsonl", rows)
        combined.extend(rows)

    save_json(generated_dir / "agus_v1_all.json", combined)
    save_jsonl(generated_dir / "agus_v1_all.jsonl", combined)

    for family, rows in datasets.items():
        print(f"{family}: {len(rows)} tasks")
    print(f"combined: {len(combined)} tasks")


if __name__ == "__main__":
    main()
