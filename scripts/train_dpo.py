
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from distill.training.train_dpo import load_dpo_training_plan, train_dpo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DPO stage for a distilled model.")
    parser.add_argument(
        "--config",
        default="configs/dpo.yaml",
        help="Path to DPO config YAML.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved DPO training plan without starting training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dry_run:
        plan = load_dpo_training_plan(args.config)
        print(json.dumps(asdict(plan), indent=2, sort_keys=True))
        return

    train_dpo(args.config)


if __name__ == "__main__":
    main()
