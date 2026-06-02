from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


@dataclass(frozen=True)
class PromptRecord:
    id: str
    category: str
    prompt: str
    metadata: dict


def parse_prompt_record(raw: dict, line_number: int) -> PromptRecord:
    if not isinstance(raw, dict):
        raise ValueError(f"line {line_number}: prompt record must be a JSON object")

    record_id = raw.get("id")
    category = raw.get("category")
    prompt = raw.get("prompt")
    metadata = raw.get("metadata", {})

    if not isinstance(record_id, str) or not record_id:
        raise ValueError(f"line {line_number}: field 'id' must be a non-empty string")

    if not isinstance(category, str) or not category:
        raise ValueError(
            f"line {line_number}: field 'category' must be a non-empty string"
        )

    if not isinstance(prompt, str) or not prompt:
        raise ValueError(
            f"line {line_number}: field 'prompt' must be a non-empty string"
        )

    if not isinstance(metadata, dict):
        raise ValueError(f"line {line_number}: field 'metadata' must be an object")

    return PromptRecord(
        id=record_id,
        category=category,
        prompt=prompt,
        metadata=metadata,
    )


def iter_prompt_records(path: str | Path) -> Iterator[PromptRecord]:
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{input_path}: line {line_number}: invalid JSON") from exc

            yield parse_prompt_record(raw, line_number)


def load_prompt_records(path: str | Path) -> list[PromptRecord]:
    return list(iter_prompt_records(path))


def load_merged_prompt_records(paths: Sequence[str | Path]) -> list[PromptRecord]:
    if not paths:
        raise ValueError("at least one prompt file path is required")

    records: list[PromptRecord] = []
    seen_ids: set[str] = set()

    for path in paths:
        for record in iter_prompt_records(path):
            if record.id in seen_ids:
                raise ValueError(f"duplicate prompt id found: {record.id}")

            seen_ids.add(record.id)
            records.append(record)

    return records
