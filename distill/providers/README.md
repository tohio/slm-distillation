# Providers

Teacher inference provider adapters.

Providers expose a common generation interface so the rest of the pipeline does not depend on a specific hosted API or local backend.

## Files

| File | Purpose |
|---|---|
| `base.py` | Shared request/response dataclasses and provider protocol. |
| `openrouter.py` | OpenRouter chat-completions provider. |
| `groq.py` | Groq provider placeholder. |
| `deepseek.py` | DeepSeek API provider placeholder. |
| `local.py` | Local model provider placeholder. |

## Rule

Provider adapters should return normalized `GenerationResponse` objects with response text, model name, provider name, and token counts when available.
