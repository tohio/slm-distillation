from __future__ import annotations

import argparse

from distill.datasets.build_distill_dataset import build_distill_dataset
from distill.utils.config import load_response_distill_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build response distillation dataset.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional response distillation config YAML.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Validated teacher JSONL path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output distillation JSONL path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.config:
        config = load_response_distill_config(args.config)
        input_path = args.input or config.data.validated_path
        output_path = args.output or config.data.distill_dataset_path
    else:
        if not args.input or not args.output:
            raise SystemExit("--input and --output are required when --config is not used")
        input_path = args.input
        output_path = args.output

    count = build_distill_dataset(input_path, output_path)

    print(f"Wrote records: {count}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
