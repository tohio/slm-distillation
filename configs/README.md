# Configurations

Runtime configuration for training, evaluation, export, and artifact handoff.

| File | Purpose |
|---|---|
| `response_distill.yaml` | Response student, dataset schema mapping, and outputs |
| `dpo.yaml` | DPO source checkpoint, published preference dataset, hyperparameters, and outputs |
| `logit_distill.yaml` | Local teacher/student checkpoints, compatibility rules, and hyperparameters |
| `eval.yaml` | Evaluation settings |
| `export.yaml` | Final model export and provenance metadata |
| `artifacts.yaml` | Model artifact packaging and S3 handoff |

Hosted teacher and dataset-generation configurations belong in
`slm-synthetic-data`, not this repository.
