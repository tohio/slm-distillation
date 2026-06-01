from __future__ import annotations

from distill.datasets.formatters import load_validated_records, write_distill_jsonl


def build_distill_dataset(input_path: str, output_path: str) -> int:
    records = load_validated_records(input_path)
    return write_distill_jsonl(output_path, records)
