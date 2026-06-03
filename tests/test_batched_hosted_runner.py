import json
from pathlib import Path
from types import SimpleNamespace

from distill.generation.hosted_runner import (
    build_batch_prompt,
    parse_batch_response,
    run_hosted_generation,
)
from distill.generation.prompts import PromptRecord
from distill.providers.base import GenerationResponse


class BatchEchoProvider:
    provider_name = "echo"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        body = request.prompt[request.prompt.index("[") : request.prompt.rindex("]") + 1]
        items = json.loads(body)
        output = [{"id": item["id"], "output": f"answer for {item['prompt']}"} for item in items]
        return GenerationResponse(
            text=json.dumps(output),
            model=request.model,
            provider=self.provider_name,
            input_tokens=10,
            output_tokens=20,
            raw={},
        )


def prompts(count: int) -> list[PromptRecord]:
    return [
        PromptRecord(id=f"p{i}", category="test", prompt=f"prompt {i}", metadata={})
        for i in range(count)
    ]


def test_build_batch_prompt_contains_all_ids() -> None:
    text = build_batch_prompt(prompts(3))
    assert "p0" in text
    assert "p1" in text
    assert "p2" in text
    assert "Return only a JSON object with key `items`" in text


def test_parse_batch_response_maps_outputs() -> None:
    response = json.dumps([{"id": "p0", "output": "a"}, {"id": "p1", "output": "b"}])
    assert parse_batch_response(prompts=prompts(2), response_text=response) == {"p0": "a", "p1": "b"}


def test_run_hosted_generation_batches_requests(tmp_path: Path) -> None:
    provider = BatchEchoProvider()
    output_path = tmp_path / "raw.jsonl"

    result = run_hosted_generation(
        provider=provider,
        prompts=prompts(10),
        output_path=output_path,
        teacher_model="teacher",
        provider_name="openrouter",
        max_output_tokens=64,
        temperature=0.1,
        top_p=0.9,
        controls=SimpleNamespace(concurrency=1),
        continue_on_error=True,
        batch_size=4,
        min_batch_size=1,
        parallel_requests=1,
        progress_interval=10,
    )

    assert result.written == 10
    assert provider.calls == 3
    assert output_path.read_text(encoding="utf-8").count("\n") == 10


def test_run_hosted_generation_resumes_existing_records(tmp_path: Path) -> None:
    provider = BatchEchoProvider()
    output_path = tmp_path / "raw.jsonl"
    output_path.write_text(json.dumps({"prompt_id": "p0", "output": "already done"}) + "\n", encoding="utf-8")

    result = run_hosted_generation(
        provider=provider,
        prompts=prompts(3),
        output_path=output_path,
        teacher_model="teacher",
        provider_name="openrouter",
        max_output_tokens=64,
        temperature=0.1,
        top_p=0.9,
        controls=SimpleNamespace(concurrency=1),
        continue_on_error=True,
        batch_size=2,
        min_batch_size=1,
        parallel_requests=1,
        progress_interval=10,
    )

    assert result.skipped == 1
    assert result.written == 2
    assert output_path.read_text(encoding="utf-8").count("\n") == 3
