from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class PreferenceRecord:
    prompt: str
    chosen: str
    rejected: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RejectedPreferenceRecord:
    prompt: str
    chosen: str | None
    rejected: str | None
    reason: str
    metadata: dict[str, Any]


def parse_preference_record(raw: dict[str, Any], line_number: int) -> PreferenceRecord:
    prompt = raw.get("prompt")
    chosen = raw.get("chosen")
    rejected = raw.get("rejected")
    metadata = raw.get("metadata", {})

    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"line {line_number}: field 'prompt' must be a non-empty string")
    if not isinstance(chosen, str) or not chosen:
        raise ValueError(f"line {line_number}: field 'chosen' must be a non-empty string")
    if not isinstance(rejected, str) or not rejected:
        raise ValueError(f"line {line_number}: field 'rejected' must be a non-empty string")
    if not isinstance(metadata, dict):
        raise ValueError(f"line {line_number}: field 'metadata' must be an object")

    return PreferenceRecord(
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        metadata=metadata,
    )


def load_preference_records(path: str | Path) -> list[PreferenceRecord]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Preference dataset not found: {input_path}")

    records: list[PreferenceRecord] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{input_path}: line {line_number}: invalid JSON") from exc
            records.append(parse_preference_record(raw, line_number))
    return records


def write_preference_records(path: str | Path, records: Iterable[PreferenceRecord]) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
            count += 1
    return count


def write_rejected_preference_records(
    path: str | Path,
    records: Iterable[RejectedPreferenceRecord],
) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
            count += 1
    return count
