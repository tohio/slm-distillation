from __future__ import annotations

from dataclasses import dataclass, replace

from distill.generation.prompts import PromptRecord


@dataclass(frozen=True)
class TokenBudgetPlan:
    target_tokens: int
    estimated_tokens_per_record: int
    selected_records: int
    estimated_total_tokens: int
    repeated: bool


def estimate_text_tokens(text: str, *, chars_per_token: float = 4.0) -> int:
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be greater than zero")
    return max(1, int(round(len(text) / chars_per_token)))


def _repeat_prompt_record(record: PromptRecord, repeat_index: int) -> PromptRecord:
    metadata = {
        **record.metadata,
        "source_prompt_id": record.id,
        "repeat_index": repeat_index,
    }
    return replace(
        record,
        id=f"{record.id}__repeat_{repeat_index:06d}",
        metadata=metadata,
    )


def select_prompts_for_token_target(
    prompts: list[PromptRecord],
    *,
    target_tokens: int | None,
    estimated_tokens_per_record: int,
    allow_repeat: bool,
) -> tuple[list[PromptRecord], TokenBudgetPlan | None]:
    if target_tokens is None:
        return prompts, None

    if target_tokens <= 0:
        raise ValueError("target_tokens must be greater than zero")

    if estimated_tokens_per_record <= 0:
        raise ValueError("estimated_tokens_per_record must be greater than zero")

    if not prompts:
        raise ValueError("cannot select prompts for token target from an empty prompt list")

    needed_records = (target_tokens + estimated_tokens_per_record - 1) // estimated_tokens_per_record

    if needed_records <= len(prompts):
        selected = prompts[:needed_records]
        return selected, TokenBudgetPlan(
            target_tokens=target_tokens,
            estimated_tokens_per_record=estimated_tokens_per_record,
            selected_records=len(selected),
            estimated_total_tokens=len(selected) * estimated_tokens_per_record,
            repeated=False,
        )

    if not allow_repeat:
        available_estimate = len(prompts) * estimated_tokens_per_record
        raise ValueError(
            "token target requires more prompts than are available: "
            f"target_tokens={target_tokens}, "
            f"estimated_tokens_per_record={estimated_tokens_per_record}, "
            f"available_prompts={len(prompts)}, "
            f"available_estimated_tokens={available_estimate}. "
            "Add more prompt seeds or pass --allow-repeat-prompts."
        )

    selected: list[PromptRecord] = []
    repeat_index = 0
    while len(selected) < needed_records:
        for prompt in prompts:
            if len(selected) >= needed_records:
                break
            if repeat_index == 0:
                selected.append(prompt)
            else:
                selected.append(_repeat_prompt_record(prompt, repeat_index))
        repeat_index += 1

    return selected, TokenBudgetPlan(
        target_tokens=target_tokens,
        estimated_tokens_per_record=estimated_tokens_per_record,
        selected_records=len(selected),
        estimated_total_tokens=len(selected) * estimated_tokens_per_record,
        repeated=True,
    )
