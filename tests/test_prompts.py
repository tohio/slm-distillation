from pathlib import Path

import pytest

from distill.generation.prompts import load_prompt_records, parse_prompt_record


def test_parse_prompt_record_accepts_valid_record() -> None:
    record = parse_prompt_record(
        {
            "id": "example-1",
            "category": "instruction",
            "prompt": "Explain virtual environments.",
            "metadata": {"source": "test"},
        },
        line_number=1,
    )

    assert record.id == "example-1"
    assert record.category == "instruction"
    assert record.prompt == "Explain virtual environments."
    assert record.metadata == {"source": "test"}


def test_parse_prompt_record_rejects_missing_prompt() -> None:
    with pytest.raises(ValueError, match="prompt"):
        parse_prompt_record(
            {
                "id": "example-1",
                "category": "instruction",
            },
            line_number=1,
        )


def test_load_prompt_records_reads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        '{"id":"p1","category":"instruction","prompt":"Say hello.","metadata":{}}\n'
        '{"id":"p2","category":"code","prompt":"Write a function.","metadata":{"x":1}}\n',
        encoding="utf-8",
    )

    records = load_prompt_records(path)

    assert len(records) == 2
    assert records[0].id == "p1"
    assert records[1].category == "code"


def test_load_prompt_records_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{bad json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        load_prompt_records(path)


def test_load_prompt_records_reads_seed_file() -> None:
    records = load_prompt_records("data/prompts/instruction_seed.jsonl")

    assert len(records) >= 1
    assert records[0].id.startswith("instruction-")
