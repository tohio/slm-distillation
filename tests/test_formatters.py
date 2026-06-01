from pathlib import Path

import pytest

from distill.datasets.build_distill_dataset import build_distill_dataset
from distill.datasets.formatters import (
    load_validated_records,
    parse_validated_record,
    write_distill_jsonl,
)


def accepted_raw() -> dict:
    return {
        "prompt_id": "p1",
        "category": "instruction",
        "prompt": "Say hello.",
        "teacher": "deepseek/deepseek-v4-flash",
        "provider": "openrouter",
        "response": "hello",
        "input_tokens": 3,
        "output_tokens": 1,
        "metadata": {"source": "test"},
        "validation": {"accepted": True, "reason": None},
    }


def test_parse_validated_record_accepts_valid_record() -> None:
    record = parse_validated_record(accepted_raw(), line_number=1)

    assert record.instruction == "Say hello."
    assert record.input is None
    assert record.output == "hello"
    assert record.metadata["prompt_id"] == "p1"
    assert record.metadata["teacher"] == "deepseek/deepseek-v4-flash"


def test_parse_validated_record_rejects_unaccepted_record() -> None:
    raw = accepted_raw()
    raw["validation"] = {"accepted": False, "reason": "empty_response"}

    with pytest.raises(ValueError, match="not accepted"):
        parse_validated_record(raw, line_number=1)


def test_write_and_load_distill_records(tmp_path: Path) -> None:
    input_path = tmp_path / "validated.jsonl"
    output_path = tmp_path / "distill.jsonl"

    input_path.write_text(
        (
            '{"prompt_id":"p1","category":"instruction","prompt":"Say hello.",'
            '"teacher":"deepseek/deepseek-v4-flash","provider":"openrouter",'
            '"response":"hello","input_tokens":3,"output_tokens":1,'
            '"metadata":{"source":"test"},'
            '"validation":{"accepted":true,"reason":null}}\n'
        ),
        encoding="utf-8",
    )

    records = load_validated_records(input_path)
    count = write_distill_jsonl(output_path, records)

    assert count == 1
    text = output_path.read_text(encoding="utf-8")
    assert '"instruction": "Say hello."' in text
    assert '"output": "hello"' in text


def test_build_distill_dataset(tmp_path: Path) -> None:
    input_path = tmp_path / "validated.jsonl"
    output_path = tmp_path / "distill.jsonl"

    input_path.write_text(
        (
            '{"prompt_id":"p1","category":"instruction","prompt":"Say hello.",'
            '"teacher":"teacher","provider":"provider","response":"hello",'
            '"input_tokens":3,"output_tokens":1,"metadata":{},'
            '"validation":{"accepted":true,"reason":null}}\n'
        ),
        encoding="utf-8",
    )

    count = build_distill_dataset(str(input_path), str(output_path))

    assert count == 1
    assert output_path.exists()
