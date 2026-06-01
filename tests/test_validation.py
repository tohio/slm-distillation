from pathlib import Path

from distill.generation.records import TeacherResponseRecord
from distill.validation.filters import split_validated_records, validate_teacher_response
from distill.validation.schema import write_validated_jsonl


def make_record(response: str, metadata: dict | None = None) -> TeacherResponseRecord:
    return TeacherResponseRecord(
        prompt_id="p1",
        category="instruction",
        prompt="Say hello.",
        teacher="deepseek/deepseek-v4-flash",
        provider="openrouter",
        response=response,
        input_tokens=3,
        output_tokens=1,
        metadata=metadata or {},
    )


def test_validate_teacher_response_accepts_normal_response() -> None:
    result = validate_teacher_response(make_record("hello"))

    assert result.accepted is True
    assert result.reason is None


def test_validate_teacher_response_rejects_empty_response() -> None:
    result = validate_teacher_response(make_record("   "))

    assert result.accepted is False
    assert result.reason == "empty_response"


def test_validate_teacher_response_rejects_unexpected_refusal() -> None:
    result = validate_teacher_response(make_record("I cannot answer that."))

    assert result.accepted is False
    assert result.reason == "unexpected_refusal"


def test_validate_teacher_response_allows_expected_refusal() -> None:
    result = validate_teacher_response(
        make_record("I cannot answer that.", metadata={"allow_refusal": True})
    )

    assert result.accepted is True


def test_validate_teacher_response_rejects_code_fence_for_function_body() -> None:
    result = validate_teacher_response(
        make_record(
            "```python\nreturn items[0]\n```",
            metadata={"task_type": "function_body"},
        )
    )

    assert result.accepted is False
    assert result.reason == "code_fence_in_function_body"


def test_split_validated_records() -> None:
    accepted, rejected = split_validated_records(
        [
            make_record("hello"),
            make_record(""),
        ]
    )

    assert len(accepted) == 1
    assert len(rejected) == 1
    assert rejected[0].validation.reason == "empty_response"


def test_write_validated_jsonl(tmp_path: Path) -> None:
    accepted, _ = split_validated_records([make_record("hello")])
    path = tmp_path / "validated.jsonl"

    count = write_validated_jsonl(path, accepted)

    assert count == 1
    text = path.read_text(encoding="utf-8")
    assert '"validation"' in text
    assert '"accepted": true' in text
