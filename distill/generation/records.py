from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class TeacherResponseRecord:
    prompt_id: str
    category: str
    prompt: str
    teacher: str
    provider: str
    response: str
    input_tokens: int | None
    output_tokens: int | None
    metadata: dict[str, Any]
    error: str | None = None


def write_jsonl(path: str | Path, records: Iterable[TeacherResponseRecord]) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            count += 1

    return count


def parse_teacher_response_record(
    raw: dict[str, Any],
    line_number: int,
) -> TeacherResponseRecord:
    required = [
        "prompt_id",
        "category",
        "prompt",
        "teacher",
        "provider",
        "response",
        "metadata",
    ]

    for field in required:
        if field not in raw:
            raise ValueError(f"line {line_number}: missing required field '{field}'")

    for field in ["prompt_id", "category", "prompt", "teacher", "provider", "response"]:
        if not isinstance(raw[field], str):
            raise ValueError(f"line {line_number}: field '{field}' must be a string")

    if not isinstance(raw["metadata"], dict):
        raise ValueError(f"line {line_number}: field 'metadata' must be an object")

    input_tokens = raw.get("input_tokens")
    output_tokens = raw.get("output_tokens")

    if input_tokens is not None and not isinstance(input_tokens, int):
        raise ValueError(f"line {line_number}: field 'input_tokens' must be int or null")

    if output_tokens is not None and not isinstance(output_tokens, int):
        raise ValueError(f"line {line_number}: field 'output_tokens' must be int or null")

    return TeacherResponseRecord(
        prompt_id=raw["prompt_id"],
        category=raw["category"],
        prompt=raw["prompt"],
        teacher=raw["teacher"],
        provider=raw["provider"],
        response=raw["response"],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        metadata=raw["metadata"],
        error=raw.get("error"),
    )


def load_teacher_response_records(path: str | Path) -> list[TeacherResponseRecord]:
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"Teacher response file not found: {input_path}")

    records: list[TeacherResponseRecord] = []

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

            records.append(parse_teacher_response_record(raw, line_number))

    return records
