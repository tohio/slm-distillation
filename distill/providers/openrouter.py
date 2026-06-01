from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from distill.providers.base import GenerationRequest, GenerationResponse


OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass(frozen=True)
class OpenRouterPayload:
    url: str
    headers: dict[str, str]
    json: dict[str, Any]


class OpenRouterProvider:
    provider_name = "openrouter"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")

    def build_payload(self, request: GenerationRequest) -> OpenRouterPayload:
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouterProvider")

        return OpenRouterPayload(
            url=OPENROUTER_CHAT_COMPLETIONS_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": request.model,
                "messages": [
                    {
                        "role": "user",
                        "content": request.prompt,
                    }
                ],
                "max_tokens": request.max_output_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
            },
        )

    def parse_response(
        self,
        payload: dict[str, Any],
        model: str,
    ) -> GenerationResponse:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("OpenRouter response missing non-empty choices")

        first = choices[0]
        if not isinstance(first, dict):
            raise ValueError("OpenRouter response choice must be a mapping")

        message = first.get("message")
        if not isinstance(message, dict):
            raise ValueError("OpenRouter response choice missing message")

        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("OpenRouter response message content must be a string")

        usage = payload.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}

        return GenerationResponse(
            text=content,
            model=model,
            provider=self.provider_name,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            raw=payload,
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        raise NotImplementedError(
            "Network generation is not implemented yet. "
            "Use build_payload and parse_response first."
        )
