from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from distill.training.train_logit_distill import (
    load_logit_distillation_plan,
    train_logit_distill,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train local logit distillation stage.")
    parser.add_argument(
        "--config",
        default="configs/logit_distill.yaml",
        help="Path to logit distillation config YAML.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved logit distillation plan without training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dry_run:
        plan = load_logit_distillation_plan(args.config)
        print(json.dumps(asdict(plan), indent=2, sort_keys=True))
        return

    train_logit_distill(args.config)


if __name__ == "__main__":
    main()
