from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from distill.training.train_response_distill import (
    load_response_training_plan,
    train_response_distill,
    validate_response_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or validate the response-distillation stage."
    )
    parser.add_argument(
        "--config",
        default="configs/response_distill.yaml",
        help="Path to response-distillation config YAML.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved response training plan without loading inputs.",
    )
    mode.add_argument(
        "--validate-inputs",
        action="store_true",
        help="Resolve the model and inspect the configured dataset.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of dataset rows to inspect with --validate-inputs.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override training.max_steps for a bounded smoke run.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Override training.max_train_samples.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Trainer checkpoint directory to resume.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dry_run:
        plan = load_response_training_plan(args.config)
        print(json.dumps(asdict(plan), indent=2, sort_keys=True))
        return

    if args.validate_inputs:
        result = validate_response_inputs(
            args.config,
            limit=args.limit,
        )
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return

    result = train_response_distill(
        args.config,
        max_steps=args.max_steps,
        max_train_samples=args.max_train_samples,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
