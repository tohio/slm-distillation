import json
from pathlib import Path

import pytest

from distill.utils.tokenizer_compat import (
    assert_tokenizer_compatible,
    check_tokenizer_compatibility,
)


def write_tokenizer(path: Path, vocab: dict[str, int], added_tokens=None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": {
            "vocab": vocab,
        },
        "added_tokens": added_tokens or [],
    }
    (path / "tokenizer.json").write_text(json.dumps(payload), encoding="utf-8")


def test_check_tokenizer_compatibility_accepts_matching_tokenizers(tmp_path: Path) -> None:
    teacher = tmp_path / "teacher"
    student = tmp_path / "student"

    vocab = {"<pad>": 0, "hello": 1, "world": 2}
    write_tokenizer(teacher, vocab)
    write_tokenizer(student, vocab)

    result = check_tokenizer_compatibility(teacher, student)

    assert result.compatible is True
    assert result.reason == "compatible"


def test_check_tokenizer_compatibility_rejects_vocab_mismatch(tmp_path: Path) -> None:
    teacher = tmp_path / "teacher"
    student = tmp_path / "student"

    write_tokenizer(teacher, {"hello": 1})
    write_tokenizer(student, {"hello": 2})

    result = check_tokenizer_compatibility(teacher, student)

    assert result.compatible is False
    assert result.reason == "vocab_mismatch"


def test_assert_tokenizer_compatible_raises_on_mismatch(tmp_path: Path) -> None:
    teacher = tmp_path / "teacher"
    student = tmp_path / "student"

    write_tokenizer(teacher, {"hello": 1})
    write_tokenizer(student, {"hello": 2})

    with pytest.raises(ValueError, match="vocab_mismatch"):
        assert_tokenizer_compatible(teacher, student)


def test_check_tokenizer_compatibility_rejects_special_token_mismatch(
    tmp_path: Path,
) -> None:
    teacher = tmp_path / "teacher"
    student = tmp_path / "student"
    vocab = {"hello": 1}

    write_tokenizer(teacher, vocab, added_tokens=[{"id": 0, "content": "<a>"}])
    write_tokenizer(student, vocab, added_tokens=[{"id": 0, "content": "<b>"}])

    result = check_tokenizer_compatibility(teacher, student)

    assert result.compatible is False
    assert result.reason == "special_tokens_mismatch"
