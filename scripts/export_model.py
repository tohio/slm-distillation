from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from distill.export.export_model import export_model, load_export_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a final distilled model variant.")
    parser.add_argument(
        "--config",
        default="configs/export.yaml",
        help="Path to export config YAML.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved export plan without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dry_run:
        plan = load_export_plan(args.config)
        print(json.dumps(asdict(plan), indent=2, sort_keys=True))
        return

    plan = export_model(args.config)
    print(f"Model: {plan.model_name}")
    print(f"Export repo: {plan.export_repo}")
    print(f"Model card: {plan.model_card_path}")


if __name__ == "__main__":
    main()
