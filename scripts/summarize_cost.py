from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from distill.generation.records import load_teacher_response_records
from distill.utils.config import load_teachers_config, teacher_to_pricing
from distill.utils.pricing import cost_per_accepted_sample, estimate_cost


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize teacher generation cost.")
    parser.add_argument(
        "--teachers",
        default="configs/teachers.yaml",
        help="Path to teachers config YAML.",
    )
    parser.add_argument(
        "--teacher",
        required=True,
        help="Teacher key from teachers config.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Raw or validated teacher response JSONL.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON path for cost summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    teachers_config = load_teachers_config(args.teachers)

    if args.teacher not in teachers_config.teachers:
        raise SystemExit(f"Unknown teacher: {args.teacher}")

    teacher = teachers_config.teachers[args.teacher]
    pricing = teacher_to_pricing(teacher)

    records = load_teacher_response_records(args.input)

    input_tokens = sum(record.input_tokens or 0 for record in records)
    output_tokens = sum(record.output_tokens or 0 for record in records)

    estimate = estimate_cost(
        pricing=pricing,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    summary = {
        **asdict(estimate),
        "records": len(records),
        "cost_per_record_usd": (
            cost_per_accepted_sample(estimate.total_cost_usd, len(records))
            if records
            else None
        ),
    }

    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
