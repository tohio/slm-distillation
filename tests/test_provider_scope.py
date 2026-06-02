from pathlib import Path

import pytest

from distill.providers.groq import GroqProvider
from distill.providers.local import LocalProvider
from distill.providers.openrouter import OpenRouterProvider


def test_openrouter_provider_scope() -> None:
    provider = OpenRouterProvider(api_key="test-key")

    assert provider.provider_name == "openrouter"


def test_groq_provider_scope() -> None:
    provider = GroqProvider(api_key="test-key")

    assert provider.provider_name == "groq"
    assert provider.supports_response_distillation is True
    assert provider.supports_logit_distillation is False


def test_local_provider_scope() -> None:
    provider = LocalProvider()

    assert provider.provider_name == "local"
    assert provider.supports_response_distillation is True
    assert provider.supports_logit_distillation is True
    assert provider.requires_single_gpu_teacher is True
    assert provider.supported_gpu_classes == ("b300", "b200", "h200", "a100")

    with pytest.raises(NotImplementedError, match="LocalProvider"):
        provider.generate(None)  # type: ignore[arg-type]


def test_direct_deepseek_provider_removed() -> None:
    assert not Path("distill/providers/deepseek.py").exists()
