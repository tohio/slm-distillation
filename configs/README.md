# Configurations

## Purpose

`configs` owns runtime YAML contracts for both branches, smoke/full execution,
evaluation, export, and artifact handoff. It does not store secrets or
synthetic-data generation settings.

## Contents

~~~text
configs/
├── response_distill.yaml / response_distill_smoke.yaml
├── logit_distill.yaml / logit_distill_smoke.yaml
├── dpo.yaml / dpo_smoke.yaml
├── dpo_logit.yaml / dpo_logit_smoke.yaml
├── eval.yaml / eval_smoke.yaml
├── eval_logit.yaml / eval_logit_smoke.yaml
├── export.yaml / export_logit.yaml
└── artifacts.yaml / artifacts_logit.yaml
~~~

## Key Files

| Stage | Response branch | Logit branch |
|---|---|---|
| Distillation | `response_distill.yaml` | `logit_distill.yaml` |
| Distillation smoke | `response_distill_smoke.yaml` | `logit_distill_smoke.yaml` |
| DPO | `dpo.yaml` | `dpo_logit.yaml` |
| DPO smoke | `dpo_smoke.yaml` | `dpo_logit_smoke.yaml` |
| Evaluation | `eval.yaml` | `eval_logit.yaml` |
| Evaluation smoke | `eval_smoke.yaml` | `eval_logit_smoke.yaml` |
| Export | `export.yaml` | `export_logit.yaml` |
| Artifacts | `artifacts.yaml` | `artifacts_logit.yaml` |

## How It Fits In

Each downstream config consumes the final checkpoint produced by the matching
upstream branch and run class. Make variables select alternate config files
without changing Python code.

## Usage/API

~~~bash
make train-response-dry-run RESPONSE_CONFIG=configs/response_distill.yaml
make train-logit-dry-run LOGIT_CONFIG=configs/logit_distill.yaml
make eval-logit-dry-run EVAL_LOGIT_CONFIG=configs/eval_logit.yaml
~~~

See [Configuration](../docs/configuration.md) for contracts and safe swapping.

## Conventions

- Keep model, tokenizer, and output identities branch-specific.
- Keep smoke paths under `runs/smoke/`.
- Keep full paths under `runs/`.
- Use `null` for optional config name, revision, or runtime bounds.
- Store credentials only in `.env` or process environment variables.

## Gotchas

Hosted teacher and dataset-generation settings belong to
`slm-synthetic-data`. Export and artifact configs target full checkpoints, not
smoke checkpoints.
