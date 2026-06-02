from __future__ import annotations

import json
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distill.generation.hosted_controls import (
    AdaptiveRequestController,
    HostedGenerationControls,
    RetryableProviderExhaustedError,
    generate_with_hosted_retries,
)
from distill.generation.prompts import PromptRecord
from distill.providers.base import GenerationRequest


@dataclass(frozen=True)
class HostedGenerationResult:
    written: int
    errors: int
    output_path: str


def teacher_response_record(
    *,
    prompt: PromptRecord,
    teacher_model: str,
    provider_name: str,
    response_text: str,
    input_tokens: int | None,
    output_tokens: int | None,
    error: str | None,
) -> dict[str, Any]:
    return {
        "prompt_id": prompt.id,
        "category": prompt.category,
        "prompt": prompt.prompt,
        "teacher": teacher_model,
        "provider": provider_name,
        "response": response_text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "metadata": prompt.metadata,
        "error": error,
    }


def error_teacher_response_record(
    *,
    prompt: PromptRecord,
    teacher_model: str,
    provider_name: str,
    error: str,
) -> dict[str, Any]:
    return teacher_response_record(
        prompt=prompt,
        teacher_model=teacher_model,
        provider_name=provider_name,
        response_text="",
        input_tokens=None,
        output_tokens=None,
        error=error,
    )


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\\n")

    return len(records)


def _generate_one(
    *,
    provider: Any,
    prompt: PromptRecord,
    teacher_model: str,
    provider_name: str,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    controls: HostedGenerationControls,
    controller: AdaptiveRequestController,
) -> dict[str, Any]:
    request = GenerationRequest(
        prompt=prompt.prompt,
        model=teacher_model,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    response = generate_with_hosted_retries(
        provider=provider,
        request=request,
        controls=controls,
        controller=controller,
    )
    return teacher_response_record(
        prompt=prompt,
        teacher_model=response.model or teacher_model,
        provider_name=response.provider or provider_name,
        response_text=response.text,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        error=None,
    )


def run_hosted_generation(
    *,
    provider: Any,
    prompts: list[PromptRecord],
    output_path: str | Path,
    teacher_model: str,
    provider_name: str,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    controls: HostedGenerationControls,
    continue_on_error: bool,
) -> HostedGenerationResult:
    controller = AdaptiveRequestController(controls)
    records_by_index: dict[int, dict[str, Any]] = {}
    requeues: dict[int, int] = {}
    pending_indices = list(range(len(prompts)))
    active: dict[Any, int] = {}
    errors = 0
    stop_submitting = False

    with ThreadPoolExecutor(max_workers=max(1, controls.concurrency)) as executor:
        while pending_indices and len(active) < controls.concurrency:
            index = pending_indices.pop(0)
            prompt = prompts[index]
            active[
                executor.submit(
                    _generate_one,
                    provider=provider,
                    prompt=prompt,
                    teacher_model=teacher_model,
                    provider_name=provider_name,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    controls=controls,
                    controller=controller,
                )
            ] = index

        while active:
            done, _ = wait(set(active), return_when=FIRST_COMPLETED)
            for future in done:
                index = active.pop(future)
                prompt = prompts[index]

                try:
                    records_by_index[index] = future.result()
                except RetryableProviderExhaustedError as exc:
                    requeues[index] = requeues.get(index, 0) + 1
                    if requeues[index] <= controls.max_requeues:
                        if controls.exhausted_retryable_requeue_delay_seconds > 0:
                            time.sleep(controls.exhausted_retryable_requeue_delay_seconds)
                        active[
                            executor.submit(
                                _generate_one,
                                provider=provider,
                                prompt=prompt,
                                teacher_model=teacher_model,
                                provider_name=provider_name,
                                max_output_tokens=max_output_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                controls=controls,
                                controller=controller,
                            )
                        ] = index
                        continue

                    errors += 1
                    if not continue_on_error:
                        raise RuntimeError(
                            f"Hosted generation failed for prompt_id={prompt.id}"
                        ) from exc
                    records_by_index[index] = error_teacher_response_record(
                        prompt=prompt,
                        teacher_model=teacher_model,
                        provider_name=provider_name,
                        error=str(exc),
                    )
                except Exception as exc:
                    errors += 1
                    if not continue_on_error:
                        raise RuntimeError(
                            f"Fatal hosted generation failure for prompt_id={prompt.id}"
                        ) from exc
                    records_by_index[index] = error_teacher_response_record(
                        prompt=prompt,
                        teacher_model=teacher_model,
                        provider_name=provider_name,
                        error=str(exc),
                    )
                    stop_submitting = True

                while (
                    not stop_submitting
                    and pending_indices
                    and len(active) < controls.concurrency
                ):
                    next_index = pending_indices.pop(0)
                    next_prompt = prompts[next_index]
                    active[
                        executor.submit(
                            _generate_one,
                            provider=provider,
                            prompt=next_prompt,
                            teacher_model=teacher_model,
                            provider_name=provider_name,
                            max_output_tokens=max_output_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            controls=controls,
                            controller=controller,
                        )
                    ] = next_index

    ordered_records = [records_by_index[index] for index in sorted(records_by_index)]
    written = write_jsonl(output_path, ordered_records)

    return HostedGenerationResult(
        written=written,
        errors=errors,
        output_path=str(output_path),
    )
