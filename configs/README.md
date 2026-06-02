# Configs

Configuration files for teacher selection, distillation runs, validation rules, and evaluation.

## Files

| File | Purpose |
|---|---|
| `preference_groq.yaml` | Preference config for default Groq GPT-OSS 20B outputs. |
| `response_distill_groq_llama_3_1_8b_instant.yaml` | Groq Llama 3.1 8B Instant comparison config. |
| `response_distill_groq_qwen3_32b.yaml` | Groq Qwen3 32B comparison config. |
| `response_distill_groq_oss_120b.yaml` | Groq GPT-OSS 120B comparison config. |
| `response_distill_groq.yaml` | Default Groq response-distillation config using OpenAI GPT-OSS 20B. |
| `prompts.yaml` | Prompt build config for scalable teacher-generation inputs. |
| `teachers.yaml` | Teacher model registry with provider, model slug, mode, and pricing metadata. |
| `response_distill_openrouter.yaml` | OpenRouter response-distillation run config. |
| `response_distill_groq.yaml` | Groq response-distillation run config. |
| `preference.yaml` | Preference dataset build config for DPO pairs. |
| `logit_distill.yaml` | Local teacher/student logit-distillation config. |
| `dpo.yaml` | DPO stage config for the final distilled model. |
| `export.yaml` | Export and model-card config for the final model variant. |
| `artifacts.yaml` | Required S3 artifact handoff config. |
| `validation.yaml` | Shared validation and filtering rules. |
| `eval.yaml` | Evaluation config for base and distilled checkpoints. |

## Groq Teacher Policy

The default Groq source config uses `openai/gpt-oss-20b` for cost-sensitive scale tests.

Active Groq comparison configs:

- `openai/gpt-oss-120b`
- `qwen/qwen3-32b`
- `llama-3.1-8b-instant`

`llama-3.3-70b-versatile` is not used as an active scale config because it is too expensive at distillation-token scale.
