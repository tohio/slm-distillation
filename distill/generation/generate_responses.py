from __future__ import annotations

from typing import Iterable

from distill.generation.prompts import PromptRecord
from distill.generation.records import TeacherResponseRecord, write_jsonl
from distill.providers.base import GenerationRequest, GenerationResponse


def generate_teacher_records(
    prompts: Iterable[PromptRecord],
    provider,
    teacher_model: str,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
) -> list[TeacherResponseRecord]:
    records: list[TeacherResponseRecord] = []

    for prompt in prompts:
        response: GenerationResponse = provider.generate(
            GenerationRequest(
                prompt=prompt.prompt,
                model=teacher_model,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
            )
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

    return records


def write_teacher_records(
    output_path: str,
    records: Iterable[TeacherResponseRecord],
) -> int:
    return write_jsonl(output_path, records)
