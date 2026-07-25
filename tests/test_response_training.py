from __future__ import annotations

from typing import Any

import pytest

from distill.data.response import CanonicalResponseRecord
from distill.training.train_response_distill import encode_response_record
from distill.utils.config import ResponseFormattingConfig


class FakeTokenizer:
    eos_token_id = 99
    eos_token = "<eos>"
    pad_token_id = None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize is True
        if add_generation_prompt:
            assert messages[-1]["role"] == "user"
            return [1, 10, 11]
        assert messages[-1]["role"] == "assistant"
        return [1, 10, 11, 20, 21, 22, 99]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
    ) -> dict[str, list[int]]:
        assert truncation is False
        if add_special_tokens:
            return {"input_ids": [1, 12, 13]}
        return {"input_ids": [20, 21, 22]}


def record() -> CanonicalResponseRecord:
    return CanonicalResponseRecord(
        id="row-one",
        prompt="Question",
        response="Answer",
        metadata={},
    )


def test_encode_response_record_masks_chat_prompt_tokens() -> None:
    encoded = encode_response_record(
        record(),
        tokenizer=FakeTokenizer(),
        formatting=ResponseFormattingConfig(
            mode="chat",
            system_prompt=None,
        ),
        max_length=16,
    )

    assert encoded["input_ids"] == [1, 10, 11, 20, 21, 22, 99]
    assert encoded["attention_mask"] == [1] * 7
    assert encoded["labels"] == [-100, -100, -100, 20, 21, 22, 99]


def test_encode_response_record_supports_plain_formatting() -> None:
    encoded = encode_response_record(
        record(),
        tokenizer=FakeTokenizer(),
        formatting=ResponseFormattingConfig(
            mode="plain",
            system_prompt=None,
        ),
        max_length=16,
    )

    assert encoded["input_ids"] == [1, 12, 13, 20, 21, 22, 99]
    assert encoded["labels"][:3] == [-100, -100, -100]


def test_encode_response_record_preserves_response_loss_when_truncated() -> None:
    encoded = encode_response_record(
        record(),
        tokenizer=FakeTokenizer(),
        formatting=ResponseFormattingConfig(
            mode="chat",
            system_prompt=None,
        ),
        max_length=3,
    )

    assert encoded["input_ids"] == [20, 21, 99]
    assert encoded["labels"] == [20, 21, 99]


def test_encode_response_record_requires_chat_template() -> None:
    class PlainTokenizer(FakeTokenizer):
        apply_chat_template: Any = None

    with pytest.raises(ValueError, match="formatting.mode='chat'"):
        encode_response_record(
            record(),
            tokenizer=PlainTokenizer(),
            formatting=ResponseFormattingConfig(
                mode="chat",
                system_prompt=None,
            ),
            max_length=16,
        )
