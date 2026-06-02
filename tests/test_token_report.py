import json
from pathlib import Path

from distill.datasets.token_report import report_jsonl_tokens


def test_report_jsonl_tokens(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps({"instruction": "hello", "output": "world"}) + "\n",
        encoding="utf-8",
    )

    report = report_jsonl_tokens(path)

    assert report.records == 1
    assert report.estimated_tokens >= 1
