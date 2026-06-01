from __future__ import annotations

import argparse

from distill.generation.records import load_teacher_response_records
from distill.utils.config import load_response_distill_config
from distill.validation.filters import split_validated_records
from distill.validation.schema import write_validated_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw teacher responses.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to response distillation config YAML.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional raw teacher JSONL path. Defaults to config data.raw_teacher_path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional validated JSONL path. Defaults to config data.validated_path.",
    )
    parser.add_argument(
        "--rejected",
        default=None,
        help="Optional rejected JSONL path. Defaults to config data.rejected_path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_response_distill_config(args.config)

    input_path = args.input or config.data.raw_teacher_path
    output_path = args.output or config.data.validated_path
    rejected_path = args.rejected or config.data.rejected_path

    records = load_teacher_response_records(input_path)

    accepted, rejected = split_validated_records(
        records,
        require_non_empty_output=config.validation.require_non_empty_output,
        reject_refusals_when_not_expected=(
            config.validation.reject_refusals_when_not_expected
        ),
        reject_code_fences_for_function_body_tasks=(
            config.validation.reject_code_fences_for_function_body_tasks
        ),
    )

    accepted_count = write_validated_jsonl(output_path, accepted)
    rejected_count = write_validated_jsonl(rejected_path, rejected)

    print(f"Loaded: {len(records)}")
    print(f"Accepted: {accepted_count}")
    print(f"Rejected: {rejected_count}")
    print(f"Wrote validated: {output_path}")
    print(f"Wrote rejected: {rejected_path}")


if __name__ == "__main__":
    main()
