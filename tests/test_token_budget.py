import pytest

from distill.generation.prompts import PromptRecord
from distill.generation.token_budget import (
    estimate_text_tokens,
    select_prompts_for_token_target,
)


def prompts(count: int) -> list[PromptRecord]:
    return [
        PromptRecord(
            id=f"p{i}",
            category="instruction",
            prompt=f"prompt {i}",
            metadata={},
        )
        for i in range(count)
    ]


def test_estimate_text_tokens() -> None:
    assert estimate_text_tokens("abcd") == 1
    assert estimate_text_tokens("abcd" * 10) == 10


def test_select_prompts_for_token_target_without_repeat() -> None:
    selected, plan = select_prompts_for_token_target(
        prompts(10),
        target_tokens=1000,
        estimated_tokens_per_record=250,
        allow_repeat=False,
    )

    assert len(selected) == 4
    assert plan is not None
    assert plan.selected_records == 4
    assert plan.estimated_total_tokens == 1000
    assert plan.repeated is False


def test_select_prompts_for_token_target_requires_repeat_when_short() -> None:
    with pytest.raises(ValueError, match="requires more prompts"):
        select_prompts_for_token_target(
            prompts(2),
            target_tokens=1000,
            estimated_tokens_per_record=250,
            allow_repeat=False,
        )


def test_select_prompts_for_token_target_can_repeat() -> None:
    selected, plan = select_prompts_for_token_target(
        prompts(2),
        target_tokens=1000,
        estimated_tokens_per_record=250,
        allow_repeat=True,
    )

    assert len(selected) == 4
    assert selected[2].id == "p0__repeat_000001"
    assert selected[2].metadata["source_prompt_id"] == "p0"
    assert plan is not None
    assert plan.repeated is True
