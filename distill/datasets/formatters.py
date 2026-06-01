from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class DistillTrainingRecord:
    instruction: str
    input: str | None
    output: str
    metadata: dict[str, Any]


def parse_validated_record(raw: dict[str, Any], line_number: int) -> DistillTrainingRecord:
    prompt = raw.get("prompt")
    response = raw.get("response")
    metadata = raw.get("metadata", {})
    validation = raw.get("validation", {})

    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"line {line_number}: field 'prompt' must be a non-empty string")

    if not isinstance(response, str) or not response:
        raise ValueError(f"line {line_number}: field 'response' must be a non-empty string")

    if not isinstance(metadata, dict):
        raise ValueError(f"line {line_number}: field 'metadata' must be an object")

    if not isinstance(validation, dict):
        raise ValueError(f"line {line_number}: field 'validation' must be an object")

    if validation.get("accepted") is not True:
        raise ValueError(f"line {line_number}: validated record is not accepted")

    return DistillTrainingRecord(
        instruction=prompt,
        input=None,
        output=response,
        metadata={
            **metadata,
            "prompt_id": raw.get("prompt_id"),
            "category": raw.get("category"),
            "teacher": raw.get("teacher"),
            "provider": raw.get("provider"),
            "input_tokens": raw.get("input_tokens"),
            "output_tokens": raw.get("output_tokens"),
        },
    )


def load_validated_records(path: str | Path) -> list[DistillTrainingRecord]:
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"Validated file not found: {input_path}")

    records: list[DistillTrainingRecord] = []

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON") from exc

            if not isinstance(raw, dict):
                raise ValueError(f"line {line_number}: record must be a JSON object")

            records.append(parse_validated_record(raw, line_number))

    return records


def write_distill_jsonl(
    path: str | Path,
    records: Iterable[DistillTrainingRecord],
) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = {
                "instruction": record.instruction,
                "input": record.input,
                "output": record.output,
                "metadata": record.metadata,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1

    return count
