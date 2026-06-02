from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from distill.generation.prompt_builder import build_prompts_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build scalable prompt records for teacher generation.")
    parser.add_argument(
        "--config",
        default="configs/prompts.yaml",
        help="Path to prompt build config YAML.",
    )
    parser.add_argument(
        "--target-records",
        type=int,
        default=None,
        help="Number of prompt records to write.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSONL path override.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print result as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_prompts_from_config(
        args.config,
        target_records=args.target_records,
        output_path=args.output,
    )

    if args.json:
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return

    print(f"Output: {result.output_path}")
    print(f"Records: {result.records}")
    print("Categories:")
    for category, count in result.category_counts.items():
        print(f"  {category}: {count}")


if __name__ == "__main__":
    main()
