from distill.generation.generate_responses import generate_teacher_records
from distill.generation.prompts import PromptRecord
from distill.providers.base import GenerationRequest, GenerationResponse


class FakeProvider:
    provider_name = "fake"

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        return GenerationResponse(
            text=f"answer: {request.prompt}",
            model=request.model,
            provider=self.provider_name,
            input_tokens=10,
            output_tokens=20,
            raw={"ok": True},
        )


def test_generate_teacher_records_with_fake_provider() -> None:
    prompts = [
        PromptRecord(
            id="p1",
            category="instruction",
            prompt="Say hello.",
            metadata={"source": "test"},
        )
    ]

    records = generate_teacher_records(
        prompts=prompts,
        provider=FakeProvider(),
        teacher_model="teacher/model",
        max_output_tokens=32,
        temperature=0.2,
        top_p=0.9,
    )

    assert len(records) == 1
    assert records[0].prompt_id == "p1"
    assert records[0].category == "instruction"
    assert records[0].prompt == "Say hello."
    assert records[0].teacher == "teacher/model"
    assert records[0].provider == "fake"
    assert records[0].response == "answer: Say hello."
    assert records[0].input_tokens == 10
    assert records[0].output_tokens == 20
    assert records[0].metadata == {"source": "test"}


class FailingProvider:
    provider_name = "failing"

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        raise RuntimeError("provider failed")


def test_generate_teacher_records_continues_on_error() -> None:
    prompts = [
        PromptRecord(
            id="p1",
            category="instruction",
            prompt="Say hello.",
            metadata={},
        )
    ]

    records = generate_teacher_records(
        prompts=prompts,
        provider=FailingProvider(),
        teacher_model="teacher/model",
        max_output_tokens=32,
        temperature=0.2,
        top_p=0.9,
        max_retries=0,
        continue_on_error=True,
    )

    assert len(records) == 1
    assert records[0].response == ""
    assert records[0].error == "provider failed"
    assert records[0].provider == "failing"
