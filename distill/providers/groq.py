from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from distill.providers.base import GenerationRequest, GenerationResponse
from distill.utils.env import get_env_value


GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"


@dataclass(frozen=True)
class GroqPayload:
    url: str
    headers: dict[str, str]
    json: dict[str, Any]


class GroqProvider:
    provider_name = "groq"
    supports_response_distillation = True
    supports_logit_distillation = False

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 60.0,
        env_path: str = ".env",
    ) -> None:
        self.api_key = api_key or get_env_value("GROQ_API_KEY", env_path)
        self.timeout = timeout

    def build_payload(self, request: GenerationRequest) -> GroqPayload:
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required in .env for GroqProvider")

        return GroqPayload(
            url=GROQ_CHAT_COMPLETIONS_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": request.model,
                "messages": [{"role": "user", "content": request.prompt}],
                "max_tokens": request.max_output_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
            },
        )

    def parse_response(self, payload: dict[str, Any], model: str) -> GenerationResponse:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("Groq response missing non-empty choices")

        first = choices[0]
        if not isinstance(first, dict):
            raise ValueError("Groq response choice must be a mapping")

        message = first.get("message")
        if not isinstance(message, dict):
            raise ValueError("Groq response choice missing message")

        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("Groq response message content must be a string")

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
        payload = self.build_payload(request)

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                payload.url,
                headers=payload.headers,
                json=payload.json,
            )
            response.raise_for_status()

        return self.parse_response(response.json(), model=request.model)
