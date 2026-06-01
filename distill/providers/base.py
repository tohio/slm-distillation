from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    model: str
    max_output_tokens: int
    temperature: float
    top_p: float


@dataclass(frozen=True)
class GenerationResponse:
    text: str
    model: str
    provider: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    raw: dict | None = None


class TeacherProvider(Protocol):
    provider_name: str

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        raise NotImplementedError
