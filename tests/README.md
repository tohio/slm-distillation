# Tests

Tests for config loading, prompt parsing, provider payloads, validation, formatting, and pricing helpers.

## Targets

```bash
make test
make test-unit
```

## Test Areas

| File | Validates |
|---|---|
| `test_config.py` | Config loaders and required fields. |
| `test_prompts.py` | Prompt JSONL parsing. |
| `test_openrouter_provider.py` | OpenRouter request/response handling. |
| `test_generate_responses.py` | Generation pipeline with fake provider. |
| `test_validation.py` | Teacher output validation filters. |
| `test_formatters.py` | Distillation dataset formatting. |
| `test_pricing.py` | Lightweight cost estimation helpers. |
