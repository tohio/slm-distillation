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
| `artifacts.yaml` | Required generated JSONL artifact handoff config. |
| `validation.yaml` | Shared validation and filtering rules. |
| `eval.yaml` | Evaluation config for base and distilled checkpoints. |
