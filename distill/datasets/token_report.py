from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distill.generation.token_budget import estimate_text_tokens


@dataclass(frozen=True)
class DatasetTokenReport:
    path: str
    records: int
    estimated_tokens: int


def _collect_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(_collect_text(item) for item in value.values())
    if isinstance(value, list):
        return "\n".join(_collect_text(item) for item in value)
    return ""


def report_jsonl_tokens(
    path: str | Path,
    *,
    chars_per_token: float = 4.0,
) -> DatasetTokenReport:
    input_path = Path(path)
    records = 0
    estimated_tokens = 0

    if not input_path.exists():
        return DatasetTokenReport(
            path=str(input_path),
            records=0,
            estimated_tokens=0,
        )

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{input_path}: line {line_number}: invalid JSON") from exc

            records += 1
            estimated_tokens += estimate_text_tokens(
                _collect_text(raw),
                chars_per_token=chars_per_token,
            )

    return DatasetTokenReport(
        path=str(input_path),
        records=records,
        estimated_tokens=estimated_tokens,
    )


def report_paths(
    paths: list[str | Path],
    *,
    chars_per_token: float = 4.0,
) -> list[DatasetTokenReport]:
    return [
        report_jsonl_tokens(path, chars_per_token=chars_per_token)
        for path in paths
    ]
