# distill Package

Core Python package for SLM distillation workflows.

## Subpackages

| Directory | Purpose |
|---|---|
| `artifacts/` | Required generated JSONL artifact handoff utilities. |
| `providers/` | Hosted and local teacher provider adapters. |
| `generation/` | Prompt loading and teacher response generation. |
| `validation/` | Schema checks and quality filters. |
| `datasets/` | Conversion from validated responses to training datasets. |
| `preference/` | Preference-pair dataset builders for DPO. |
| `training/` | Response, logit, and DPO training stages. |
| `export/` | Model-card generation and export planning. |
| `eval/` | Evaluation and output comparison utilities. |
| `utils/` | Shared config, IO, pricing, and logging helpers. |
