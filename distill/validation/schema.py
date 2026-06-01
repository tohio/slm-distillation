from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from distill.validation.filters import ValidatedTeacherResponse


def write_validated_jsonl(
    path: str | Path,
    records: Iterable[ValidatedTeacherResponse],
) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for item in records:
            payload = asdict(item.record)
            payload["validation"] = asdict(item.validation)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1

    return count
