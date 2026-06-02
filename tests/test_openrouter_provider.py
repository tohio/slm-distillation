import pytest

from distill.providers.base import GenerationRequest
from distill.providers.openrouter import (
    OPENROUTER_CHAT_COMPLETIONS_URL,
    OpenRouterProvider,
)


def test_openrouter_build_payload() -> None:
    provider = OpenRouterProvider(api_key="test-key")
    request = GenerationRequest(
        prompt="Return only the word hello.",
        model="deepseek/deepseek-v4-flash",
        max_output_tokens=32,
        temperature=0.2,
        top_p=0.9,
    )

    payload = provider.build_payload(request)

    assert payload.url == OPENROUTER_CHAT_COMPLETIONS_URL
    assert payload.headers["Authorization"] == "Bearer test-key"
    assert payload.json["model"] == "deepseek/deepseek-v4-flash"
    assert payload.json["max_tokens"] == 32
    assert payload.json["temperature"] == 0.2
    assert payload.json["top_p"] == 0.9
    assert payload.json["messages"] == [
        {"role": "user", "content": "Return only the word hello."}
    ]


def test_openrouter_requires_api_key(tmp_path) -> None:
    provider = OpenRouterProvider(env_path=str(tmp_path / ".env"))

    request = GenerationRequest(
        prompt="test",
        model="deepseek/deepseek-v4-flash",
        max_output_tokens=32,
        temperature=0.2,
        top_p=0.9,
    )

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        provider.build_payload(request)


def test_openrouter_parse_response() -> None:
    provider = OpenRouterProvider(api_key="test-key")

    response = provider.parse_response(
        {
            "choices": [
                {
                    "message": {
                        "content": "hello",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 1,
            },
        },
        model="deepseek/deepseek-v4-flash",
    )

    assert response.text == "hello"
    assert response.model == "deepseek/deepseek-v4-flash"
    assert response.provider == "openrouter"
    assert response.input_tokens == 5
    assert response.output_tokens == 1
