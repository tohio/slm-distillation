from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distill.generation.prompts import PromptRecord
from distill.providers.base import GenerationRequest, GenerationResponse


@dataclass(frozen=True)
class HostedGenerationResult:
    output_path: str
    written: int
    skipped: int
    errors: int


@dataclass(frozen=True)
class BatchGenerationItem:
    prompt: PromptRecord
    output: str
    response: GenerationResponse | None
    error: str | None = None


def build_batch_prompt(prompts: list[PromptRecord]) -> str:
    items = [
        {"id": prompt.id, "category": prompt.category, "prompt": prompt.prompt}
        for prompt in prompts
    ]
    return (
        "You are generating supervised distillation outputs.\n"
        "Return ONLY a valid JSON array. Do not include markdown, code fences, or text outside JSON.\n"
        "The array MUST contain exactly one object for each input item, in the same order.\n"
        "Each output object MUST use this schema: {\"id\": string, \"output\": string}\n"
        "Keep each output concise, complete, and directly useful.\n"
        "Do not add long introductions, repeated disclaimers, or unrelated explanation.\n"
        "For code tasks, return the requested code and only the minimal necessary notes.\n"
        "For planning/debugging tasks, prefer compact bullets or numbered steps.\n\n"
        "Input items:\n"
        f"{json.dumps(items, ensure_ascii=False, indent=2)}"
    )


def _chunked(items: list[PromptRecord], size: int) -> list[list[PromptRecord]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _extract_json_array(text: str) -> list[Any]:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("[")
        end = stripped.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(stripped[start : end + 1])

    if not isinstance(parsed, list):
        raise ValueError("batch response must be a JSON array")
    return parsed


def parse_batch_response(*, prompts: list[PromptRecord], response_text: str) -> dict[str, str]:
    parsed = _extract_json_array(response_text)
    if len(parsed) != len(prompts):
        raise ValueError(
            f"batch response item count mismatch: expected={len(prompts)}, got={len(parsed)}"
        )

    outputs: dict[str, str] = {}
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"batch response item {index} must be an object")
        expected_id = prompts[index].id
        item_id = item.get("id")
        output = item.get("output")
        if item_id != expected_id:
            raise ValueError(
                f"batch response id mismatch at index {index}: expected={expected_id}, got={item_id}"
            )
        if not isinstance(output, str) or not output.strip():
            raise ValueError(f"batch response item {index} has empty output")
        outputs[expected_id] = output.strip()
    return outputs


def _record(
    *,
    prompt: PromptRecord,
    output: str,
    teacher_model: str,
    provider_name: str,
    response: GenerationResponse | None,
    error: str | None,
) -> dict[str, Any]:
    return {
        "id": prompt.id,
        "prompt_id": prompt.id,
        "category": prompt.category,
        "prompt": prompt.prompt,
        "output": output,
        "response": output,
        "teacher": teacher_model,
        "teacher_model": teacher_model,
        "teacher_provider": provider_name,
        "provider": provider_name,
        "model": teacher_model,
        "input_tokens": response.input_tokens if response is not None else None,
        "output_tokens": response.output_tokens if response is not None else None,
        "metadata": prompt.metadata,
        "raw": response.raw if response is not None else None,
        "error": error,
    }


def _completed_ids(output_path: Path) -> set[str]:
    ids: set[str] = set()
    if not output_path.exists():
        return ids
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt_id = row.get("prompt_id") or row.get("id")
            if isinstance(prompt_id, str):
                ids.add(prompt_id)
    return ids


def _dry_run_batch(
    prompts: list[PromptRecord],
    *,
    teacher_model: str,
    provider_name: str,
) -> list[BatchGenerationItem]:
    return [
        BatchGenerationItem(
            prompt=prompt,
            output=f"Dry run response for: {prompt.prompt}",
            response=GenerationResponse(
                text=f"Dry run response for: {prompt.prompt}",
                model=teacher_model,
                provider=provider_name,
                input_tokens=len(prompt.prompt.split()),
                output_tokens=8,
                raw={"dry_run": True, "batched": True},
            ),
        )
        for prompt in prompts
    ]


def _generate_batch_once(
    *,
    provider: Any,
    prompts: list[PromptRecord],
    teacher_model: str,
    provider_name: str,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    max_attempts: int,
    retry_delay_seconds: float,
) -> list[BatchGenerationItem]:
    if getattr(provider, "provider_name", "") == "dry_run":
        return _dry_run_batch(prompts, teacher_model=teacher_model, provider_name=provider_name)

    last_error: Exception | None = None

    for attempt in range(1, max(1, max_attempts) + 1):
        try:
            response = provider.generate(
                GenerationRequest(
                    prompt=build_batch_prompt(prompts),
                    model=teacher_model,
                    max_output_tokens=max_output_tokens * len(prompts),
                    temperature=temperature,
                    top_p=top_p,
                )
            )

            try:
                outputs = parse_batch_response(prompts=prompts, response_text=response.text)
            except Exception:
                if len(prompts) != 1:
                    raise
                plain_text = response.text.strip()
                if not plain_text:
                    raise
                outputs = {prompts[0].id: plain_text}

            return [
                BatchGenerationItem(prompt=prompt, output=outputs[prompt.id], response=response)
                for prompt in prompts
            ]
        except Exception as exc:
            last_error = exc
            if attempt < max(1, max_attempts) and retry_delay_seconds > 0:
                time.sleep(retry_delay_seconds)

    assert last_error is not None
    raise last_error


def _generate_batch_with_split(
    *,
    provider: Any,
    prompts: list[PromptRecord],
    teacher_model: str,
    provider_name: str,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    min_batch_size: int,
    continue_on_error: bool,
    max_attempts: int,
    retry_delay_seconds: float,
) -> list[BatchGenerationItem]:
    try:
        return _generate_batch_once(
            provider=provider,
            prompts=prompts,
            teacher_model=teacher_model,
            provider_name=provider_name,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
        )
    except Exception as exc:
        if len(prompts) > min_batch_size:
            midpoint = len(prompts) // 2
            return [
                *_generate_batch_with_split(
                    provider=provider,
                    prompts=prompts[:midpoint],
                    teacher_model=teacher_model,
                    provider_name=provider_name,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    min_batch_size=min_batch_size,
                    continue_on_error=continue_on_error,
                    max_attempts=max_attempts,
                    retry_delay_seconds=retry_delay_seconds,
                ),
                *_generate_batch_with_split(
                    provider=provider,
                    prompts=prompts[midpoint:],
                    teacher_model=teacher_model,
                    provider_name=provider_name,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    min_batch_size=min_batch_size,
                    continue_on_error=continue_on_error,
                    max_attempts=max_attempts,
                    retry_delay_seconds=retry_delay_seconds,
                ),
            ]
        if not continue_on_error:
            raise
        return [
            BatchGenerationItem(prompt=prompt, output="", response=None, error=str(exc))
            for prompt in prompts
        ]


def _controls_value(controls: Any, name: str, default: Any) -> Any:
    return getattr(controls, name, default) if controls is not None else default


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
    controls: Any,
    continue_on_error: bool,
    batch_size: int | None = None,
    min_batch_size: int | None = None,
    parallel_requests: int | None = None,
    progress_interval: int | None = None,
    resume: bool = True,
) -> HostedGenerationResult:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    effective_batch_size = int(batch_size or _controls_value(controls, "batch_size", 1) or 1)
    effective_min_batch_size = int(min_batch_size or _controls_value(controls, "min_batch_size", 1) or 1)
    effective_parallel = int(
        parallel_requests
        or _controls_value(controls, "parallel_requests", None)
        or _controls_value(controls, "concurrency", 1)
        or 1
    )
    effective_progress_interval = int(
        progress_interval or _controls_value(controls, "progress_interval", 25) or 25
    )
    max_attempts = int(_controls_value(controls, "max_retryable_request_attempts", 1) or 1)
    retry_delay_seconds = float(
        _controls_value(controls, "retry_backoff_initial_seconds", 0.0) or 0.0
    )

    if effective_batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if effective_min_batch_size <= 0:
        raise ValueError("min_batch_size must be greater than zero")
    if effective_min_batch_size > effective_batch_size:
        raise ValueError("min_batch_size cannot be greater than batch_size")
    if effective_parallel <= 0:
        raise ValueError("parallel_requests must be greater than zero")

    existing_ids = _completed_ids(output) if resume else set()
    selected = [prompt for prompt in prompts if prompt.id not in existing_ids]
    skipped = len(prompts) - len(selected)
    total = len(selected)

    if total == 0:
        return HostedGenerationResult(str(output), 0, skipped, 0)

    print(
        "Hosted generation plan: "
        f"prompts={len(prompts)} skipped={skipped} remaining={total} "
        f"batch_size={effective_batch_size} min_batch_size={effective_min_batch_size} "
        f"parallel_requests={effective_parallel}",
        flush=True,
    )

    batches = _chunked(selected, effective_batch_size)
    written = 0
    errors = 0
    completed = 0
    next_progress = effective_progress_interval
    start = time.time()

    with output.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=effective_parallel) as executor:
            futures = [
                executor.submit(
                    _generate_batch_with_split,
                    provider=provider,
                    prompts=batch,
                    teacher_model=teacher_model,
                    provider_name=provider_name,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    min_batch_size=effective_min_batch_size,
                    continue_on_error=continue_on_error,
                    max_attempts=max_attempts,
                    retry_delay_seconds=retry_delay_seconds,
                )
                for batch in batches
            ]

            for batch_index, future in enumerate(as_completed(futures), start=1):
                items = future.result()
                batch_size_completed = len(items)
                for item in items:
                    written += 1
                    if item.error:
                        errors += 1
                    handle.write(
                        json.dumps(
                            _record(
                                prompt=item.prompt,
                                output=item.output,
                                teacher_model=teacher_model,
                                provider_name=provider_name,
                                response=item.response,
                                error=item.error,
                            ),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                    handle.write("\n")
                    completed += 1

                handle.flush()

                if completed >= next_progress or completed >= total:
                    elapsed = max(0.001, time.time() - start)
                    print(
                        "Hosted generation progress: "
                        f"batches={batch_index}/{len(batches)} "
                        f"last_batch_records={batch_size_completed} "
                        f"completed={completed}/{total} "
                        f"written={written} "
                        f"errors={errors} "
                        f"skipped={skipped} "
                        f"elapsed_seconds={elapsed:.1f} "
                        f"records_per_second={completed / elapsed:.2f}",
                        flush=True,
                    )
                    while next_progress <= completed:
                        next_progress += effective_progress_interval

    return HostedGenerationResult(str(output), written, skipped, errors)
