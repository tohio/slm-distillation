# distill Package

Core package for model training, evaluation, export, and artifact handoff.

| Directory | Purpose |
|---|---|
| `data/` | Published response and preference dataset loading and schema conversion |
| `models/` | Local and Hugging Face model reference resolution |
| `training/` | Response, DPO, and logit training stages |
| `eval/` | Base and distilled checkpoint evaluation |
| `export/` | Model-card generation and model export |
| `artifacts/` | Run artifact packaging, verification, and S3 handoff |
| `utils/` | Configuration, environment, and tokenizer compatibility helpers |
