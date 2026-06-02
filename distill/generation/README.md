# Generation

Prompt loading and teacher response generation.

## Files

| File | Purpose |
|---|---|
| `prompt_builder.py` | Deterministic prompt builders for scalable distillation runs. |
| `prompts.py` | Prompt seed schemas and merged prompt loading. |
| `records.py` | Raw teacher response record schemas and JSONL IO. |
| `generate_responses.py` | Teacher response generation helpers. |
| `hosted_controls.py` | Retry, backoff, jitter, and adaptive hosted-provider controls. |
| `hosted_runner.py` | Concurrent hosted generation runner for OpenRouter and Groq. |

## Hosted Generation

Hosted generation keeps one prompt ID mapped to one raw teacher response record.

Retry-exhausted provider failures are requeued up to the configured limit. After requeue exhaustion, generation writes an error record when `continue_on_error` is enabled.

## Token Targets

Use `TARGET_TOKENS` with `make generate` or `make response-pipeline` to select prompts by estimated generated-token budget.
## Batched Hosted Generation

Hosted teacher generation groups prompt records into provider requests. Each batch asks the teacher for a JSON array with one output per prompt. Failed batches are split recursively down to `min_batch_size`. Generation writes records incrementally, supports resume by prompt ID, and prints progress.
