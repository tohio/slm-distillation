# Configs

Configuration files for teacher selection, distillation runs, validation rules, and evaluation.

## Files

| File | Purpose |
|---|---|
| `teachers.yaml` | Teacher model registry with provider, model slug, mode, and pricing metadata. |
| `response_distill.yaml` | OpenRouter response-distillation run config. |
| `response_distill_groq.yaml` | Groq response-distillation run config. |
| `preference.yaml` | Preference dataset build config for DPO pairs. |
| `logit_distill.yaml` | Local teacher/student logit-distillation config. |
| `dpo.yaml` | DPO stage config for the final distilled model. |
| `export.yaml` | Export and model-card config for the final model variant. |
| `validation.yaml` | Shared validation and filtering rules. |
| `eval.yaml` | Evaluation config for base and distilled checkpoints. |

## Hosted Generation Controls

Response-distillation configs may define provider-specific hosted generation controls under `providers.<provider>.generation`.

| Field | Purpose |
|---|---|
| `concurrency` | Maximum worker count for hosted generation. |
| `max_requeues` | Number of runner-level requeues after provider retry exhaustion. |
| `exhausted_retryable_requeue_delay_seconds` | Delay before requeueing a retry-exhausted prompt. |
| `max_retryable_request_attempts` | Per-call retry attempts for retryable provider errors. |
| `retry_backoff_initial_seconds` | Initial retry backoff. |
| `retry_backoff_max_seconds` | Maximum retry backoff. |
| `retry_jitter_ratio` | Random jitter ratio added to retry backoff. |
| `adaptive_concurrency_enabled` | Enables adaptive in-flight provider admission control. |

Hosted controls apply to OpenRouter and Groq. Local provider execution uses separate local training controls.
