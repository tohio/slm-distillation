from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TokenizerCompatibilityResult:
    compatible: bool
    teacher_tokenizer_path: str
    student_tokenizer_path: str
    reason: str


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Tokenizer file must contain a JSON object: {path}")

    return data


def _candidate_tokenizer_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]

    return [
        path / "tokenizer.json",
        path / "vocab.json",
    ]


def _find_tokenizer_file(path: str | Path) -> Path:
    base = Path(path)

    for candidate in _candidate_tokenizer_files(base):
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No tokenizer.json or vocab.json found under tokenizer path: {base}"
    )


def _load_tokenizer_payload(path: str | Path) -> dict[str, Any]:
    tokenizer_file = _find_tokenizer_file(path)
    return _load_json(tokenizer_file)


def _extract_vocab(payload: dict[str, Any]) -> dict[str, int]:
    if isinstance(payload.get("model"), dict) and isinstance(
        payload["model"].get("vocab"), dict
    ):
        vocab = payload["model"]["vocab"]
    elif isinstance(payload.get("vocab"), dict):
        vocab = payload["vocab"]
    else:
        raise ValueError("Tokenizer payload does not contain a vocab object")

    result: dict[str, int] = {}
    for token, token_id in vocab.items():
        if not isinstance(token, str) or not isinstance(token_id, int):
            raise ValueError("Tokenizer vocab must map string tokens to integer ids")
        result[token] = token_id

    return result


def _extract_special_tokens(payload: dict[str, Any]) -> dict[str, Any]:
    special_tokens: dict[str, Any] = {}

    if isinstance(payload.get("added_tokens"), list):
        special_tokens["added_tokens"] = payload["added_tokens"]

    if isinstance(payload.get("special_tokens"), dict):
        special_tokens["special_tokens"] = payload["special_tokens"]

    if isinstance(payload.get("bos_token"), str):
        special_tokens["bos_token"] = payload["bos_token"]

    if isinstance(payload.get("eos_token"), str):
        special_tokens["eos_token"] = payload["eos_token"]

    if isinstance(payload.get("pad_token"), str):
        special_tokens["pad_token"] = payload["pad_token"]

    if isinstance(payload.get("unk_token"), str):
        special_tokens["unk_token"] = payload["unk_token"]

    return special_tokens


def check_tokenizer_compatibility(
    teacher_tokenizer_path: str | Path,
    student_tokenizer_path: str | Path,
) -> TokenizerCompatibilityResult:
    teacher_payload = _load_tokenizer_payload(teacher_tokenizer_path)
    student_payload = _load_tokenizer_payload(student_tokenizer_path)

    teacher_vocab = _extract_vocab(teacher_payload)
    student_vocab = _extract_vocab(student_payload)

    if teacher_vocab != student_vocab:
        return TokenizerCompatibilityResult(
            compatible=False,
            teacher_tokenizer_path=str(teacher_tokenizer_path),
            student_tokenizer_path=str(student_tokenizer_path),
            reason="vocab_mismatch",
        )

    teacher_special_tokens = _extract_special_tokens(teacher_payload)
    student_special_tokens = _extract_special_tokens(student_payload)

    if teacher_special_tokens != student_special_tokens:
        return TokenizerCompatibilityResult(
            compatible=False,
            teacher_tokenizer_path=str(teacher_tokenizer_path),
            student_tokenizer_path=str(student_tokenizer_path),
            reason="special_tokens_mismatch",
        )

    return TokenizerCompatibilityResult(
        compatible=True,
        teacher_tokenizer_path=str(teacher_tokenizer_path),
        student_tokenizer_path=str(student_tokenizer_path),
        reason="compatible",
    )


def assert_tokenizer_compatible(
    teacher_tokenizer_path: str | Path,
    student_tokenizer_path: str | Path,
) -> TokenizerCompatibilityResult:
    result = check_tokenizer_compatibility(
        teacher_tokenizer_path=teacher_tokenizer_path,
        student_tokenizer_path=student_tokenizer_path,
    )

    if not result.compatible:
        raise ValueError(
            "Tokenizer compatibility check failed: "
            f"{result.reason} "
            f"(teacher={result.teacher_tokenizer_path}, "
            f"student={result.student_tokenizer_path})"
        )

    return result
