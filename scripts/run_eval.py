from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from distill.eval.run_eval import (
    load_evaluation_plan,
    run_evaluation,
    validate_evaluation_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate base and distilled checkpoints."
    )
    parser.add_argument(
        "--config",
        default="configs/eval.yaml",
        help="Path to evaluation config YAML.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the evaluation plan without loading inputs.",
    )
    mode.add_argument(
        "--validate-inputs",
        action="store_true",
        help="Resolve models and inspect the evaluation dataset.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override generation.limit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        result = load_evaluation_plan(args.config)
    elif args.validate_inputs:
        result = validate_evaluation_inputs(
            args.config,
            limit=args.limit,
        )
    else:
        result = run_evaluation(
            args.config,
            limit=args.limit,
        )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
