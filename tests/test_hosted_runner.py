from __future__ import annotations

from dataclasses import dataclass

from distill.generation.hosted_controls import HostedGenerationControls
from distill.generation.hosted_runner import run_hosted_generation
from distill.generation.prompts import PromptRecord
from distill.providers.base import GenerationResponse


@dataclass
class FakeProvider:
    failures_before_success: int = 0

    def __post_init__(self) -> None:
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        if self.calls <= self.failures_before_success:
            raise RuntimeError("error code: 429")
        return GenerationResponse(
            text=f"answer: {request.prompt}",
            model=request.model,
            provider="openrouter",
            input_tokens=1,
            output_tokens=2,
            raw={},
        )


def test_run_hosted_generation_writes_records(tmp_path) -> None:
    prompts = [
        PromptRecord(id="p1", category="instruction", prompt="hello", metadata={}),
        PromptRecord(id="p2", category="instruction", prompt="world", metadata={}),
    ]
    output_path = tmp_path / "raw.jsonl"

    result = run_hosted_generation(
        provider=FakeProvider(),
        prompts=prompts,
        output_path=output_path,
        teacher_model="teacher/model",
        provider_name="openrouter",
        max_output_tokens=16,
        temperature=0.2,
        top_p=0.9,
        controls=HostedGenerationControls(
            concurrency=2,
            adaptive_concurrency_enabled=False,
        ),
        continue_on_error=True,
    )

    assert result.written == 2
    assert result.errors == 0
    assert output_path.read_text().count("\\n") == 2


def test_run_hosted_generation_retries_retryable_errors(tmp_path) -> None:
    prompts = [
        PromptRecord(id="p1", category="instruction", prompt="hello", metadata={}),
    ]
    output_path = tmp_path / "raw.jsonl"

    result = run_hosted_generation(
        provider=FakeProvider(failures_before_success=1),
        prompts=prompts,
        output_path=output_path,
        teacher_model="teacher/model",
        provider_name="openrouter",
        max_output_tokens=16,
        temperature=0.2,
        top_p=0.9,
        controls=HostedGenerationControls(
            concurrency=1,
            max_retryable_request_attempts=2,
            retry_backoff_initial_seconds=0.0,
            retry_backoff_max_seconds=0.0,
            retry_jitter_ratio=0.0,
            adaptive_concurrency_enabled=False,
        ),
        continue_on_error=True,
    )

    assert result.written == 1
    assert result.errors == 0


def test_run_hosted_generation_writes_error_after_requeue_exhaustion(tmp_path) -> None:
    prompts = [
        PromptRecord(id="p1", category="instruction", prompt="hello", metadata={}),
    ]
    output_path = tmp_path / "raw.jsonl"

    result = run_hosted_generation(
        provider=FakeProvider(failures_before_success=100),
        prompts=prompts,
        output_path=output_path,
        teacher_model="teacher/model",
        provider_name="openrouter",
        max_output_tokens=16,
        temperature=0.2,
        top_p=0.9,
        controls=HostedGenerationControls(
            concurrency=1,
            max_requeues=1,
            exhausted_retryable_requeue_delay_seconds=0.0,
            max_retryable_request_attempts=1,
            retry_backoff_initial_seconds=0.0,
            retry_backoff_max_seconds=0.0,
            retry_jitter_ratio=0.0,
            adaptive_concurrency_enabled=False,
        ),
        continue_on_error=True,
    )

    assert result.written == 1
    assert result.errors == 1
    assert '"error":' in output_path.read_text()
