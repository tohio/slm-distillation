from __future__ import annotations

from distill.providers.base import GenerationRequest, GenerationResponse


class GroqProvider:
    provider_name = "groq"
    supports_response_distillation = True
    supports_logit_distillation = False

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        raise NotImplementedError(
            "GroqProvider is active in provider scope but not implemented yet. "
            "Use OpenRouterProvider until Groq generation is implemented."
        )
