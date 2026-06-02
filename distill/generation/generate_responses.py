from __future__ import annotations

import time
from typing import Iterable

from distill.generation.prompts import PromptRecord
from distill.generation.records import TeacherResponseRecord, write_jsonl
from distill.providers.base import GenerationRequest, GenerationResponse


def _generate_with_retries(
    provider,
    request: GenerationRequest,
    max_retries: int,
    retry_delay_seconds: float,
) -> GenerationResponse:
    attempts = 0

    while True:
        try:
            return provider.generate(request)
        except Exception:
            attempts += 1

            if attempts > max_retries:
                raise

            if retry_delay_seconds > 0:
                time.sleep(retry_delay_seconds)


def generate_teacher_records(
    prompts: Iterable[PromptRecord],
    provider,
    teacher_model: str,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    max_retries: int = 0,
    retry_delay_seconds: float = 0.0,
    continue_on_error: bool = False,
) -> list[TeacherResponseRecord]:
    records: list[TeacherResponseRecord] = []

    for prompt in prompts:
        request = GenerationRequest(
            prompt=prompt.prompt,
            model=teacher_model,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        try:
            response: GenerationResponse = _generate_with_retries(
                provider=provider,
                request=request,
                max_retries=max_retries,
                retry_delay_seconds=retry_delay_seconds,
            )

            records.append(
                TeacherResponseRecord(
                    prompt_id=prompt.id,
                    category=prompt.category,
                    prompt=prompt.prompt,
                    teacher=response.model,
                    provider=response.provider,
                    response=response.text,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    metadata=prompt.metadata,
                )
            )
        except Exception as exc:
            if not continue_on_error:
                raise

            provider_name = getattr(provider, "provider_name", "unknown")
            records.append(
                TeacherResponseRecord(
                    prompt_id=prompt.id,
                    category=prompt.category,
                    prompt=prompt.prompt,
                    teacher=teacher_model,
                    provider=provider_name,
                    response="",
                    input_tokens=None,
                    output_tokens=None,
                    metadata=prompt.metadata,
                    error=str(exc),
                )
            )

    return records


def write_teacher_records(
    output_path: str,
    records: Iterable[TeacherResponseRecord],
) -> int:
    return write_jsonl(output_path, records)
