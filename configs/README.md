# Configs

Configuration files for teacher selection, distillation runs, validation rules, and evaluation.

## Files

| File | Purpose |
|---|---|
| `teachers.yaml` | Defines available teacher models, providers, permissions, and pricing metadata. |
| `response_distill.yaml` | Main response-distillation run config. |
| `logit_distill.yaml` | Local teacher/student logit-distillation config. |
| `validation.yaml` | Shared validation/filtering rules. |
| `eval.yaml` | Evaluation config for base and distilled checkpoints. |

## Notes

Configs should define paths, model/provider choices, and run settings. Code should not hardcode model names, checkpoint paths, or output locations.
