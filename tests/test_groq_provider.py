import pytest

from distill.providers.base import GenerationRequest
from distill.providers.groq import GROQ_CHAT_COMPLETIONS_URL, GroqProvider


def test_groq_build_payload() -> None:
    provider = GroqProvider(api_key="test-key")
    request = GenerationRequest(
        prompt="Return only the word hello.",
        model="llama-3.3-70b-versatile",
        max_output_tokens=32,
        temperature=0.2,
        top_p=0.9,
    )

    payload = provider.build_payload(request)

    assert payload.url == GROQ_CHAT_COMPLETIONS_URL
    assert payload.headers["Authorization"] == "Bearer test-key"
    assert payload.json["model"] == "llama-3.3-70b-versatile"
    assert payload.json["max_tokens"] == 32
    assert payload.json["temperature"] == 0.2
    assert payload.json["top_p"] == 0.9
    assert payload.json["messages"] == [
        {"role": "user", "content": "Return only the word hello."}
    ]


def test_groq_requires_api_key(tmp_path) -> None:
    provider = GroqProvider(env_path=str(tmp_path / ".env"))

    request = GenerationRequest(
        prompt="test",
        model="llama-3.3-70b-versatile",
        max_output_tokens=32,
        temperature=0.2,
        top_p=0.9,
    )

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        provider.build_payload(request)


def test_groq_parse_response() -> None:
    provider = GroqProvider(api_key="test-key")

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
        model="llama-3.3-70b-versatile",
    )

    assert response.text == "hello"
    assert response.model == "llama-3.3-70b-versatile"
    assert response.provider == "groq"
    assert response.input_tokens == 5
    assert response.output_tokens == 1
