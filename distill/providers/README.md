# Providers

Teacher inference provider adapters.

## Scope

| Provider | Response Distillation | Logit Distillation | Status |
|---|---:|---:|---|
| OpenRouter | yes | no | implemented |
| Groq | yes | no | implemented |
| Local | yes | yes | pending |
| DeepSeek direct API | no | no | unsupported |

## Files

| File | Purpose |
|---|---|
| `base.py` | Shared request/response dataclasses and provider protocol. |
| `openrouter.py` | OpenRouter hosted response-distillation provider. |
| `groq.py` | Groq hosted response-distillation provider. |
| `local.py` | Local response/logit distillation provider. |

## Provider Contract

Provider adapters return normalized `GenerationResponse` records with response text, model name, provider name, and token counts when available.

## Notes

DeepSeek teacher models are accessed through OpenRouter model slugs such as `deepseek/deepseek-v4-flash`.

Hosted providers are used for teacher-generated text targets. Full logit distillation requires local model execution.

Local logit distillation requires tokenizer compatibility between teacher and student. Local teacher models should fit on a single B300, B200, H200, or A100 GPU.
