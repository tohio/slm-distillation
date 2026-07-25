# Configurations

Runtime configuration for training, evaluation, export, and artifact handoff.

| File | Purpose |
|---|---|
| `response_distill.yaml` | Response student, dataset schema, formatting, hyperparameters, and outputs |
| `response_distill_smoke.yaml` | Isolated bounded response training |
| `dpo.yaml` | DPO source checkpoint, preference schema, hyperparameters, and outputs |
| `dpo_smoke.yaml` | Isolated bounded response-branch DPO |
| `logit_distill.yaml` | Local teacher/student checkpoints, compatibility rules, and hyperparameters |
| `logit_distill_smoke.yaml` | Isolated bounded logit training |
| `dpo_logit.yaml` | DPO stage consuming the full logit checkpoint |
| `dpo_logit_smoke.yaml` | DPO stage consuming the smoke logit checkpoint |
| `eval.yaml`, `eval_logit.yaml` | Full response and logit branch evaluation |
| `eval_smoke.yaml`, `eval_logit_smoke.yaml` | Evaluation of isolated smoke checkpoints |
| `export.yaml` | Final model export and provenance metadata |
| `export_logit.yaml` | Logit-branch export and provenance metadata |
| `artifacts.yaml` | Model artifact packaging and S3 handoff |
| `artifacts_logit.yaml` | Logit-branch packaging and S3 handoff |

Hosted teacher and dataset-generation configurations belong in
`slm-synthetic-data`, not this repository.
