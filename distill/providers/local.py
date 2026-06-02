from __future__ import annotations

from distill.providers.base import GenerationRequest, GenerationResponse


class LocalProvider:
    provider_name = "local"
    supports_response_distillation = True
    supports_logit_distillation = True
    requires_single_gpu_teacher = True
    supported_gpu_classes = ("b300", "b200", "h200", "a100")

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        raise NotImplementedError(
            "LocalProvider is active in provider scope but not implemented yet. "
            "Local response/logit distillation will be implemented in a later step."
        )
