# Distill Package

## Purpose

`distill` owns the reusable Python implementation for model and dataset
resolution, training, evaluation, export, and artifact handoff. It does not own
CLI argument parsing, Make targets, synthetic dataset generation, or published
dataset storage.

## Contents

~~~text
distill/
├── artifacts/   Packaging, verification, and S3 handoff
├── data/        Response and preference dataset contracts
├── eval/        Generation and comparison metrics
├── export/      Model cards and Hugging Face export
├── models/      Local and Hugging Face reference resolution
├── training/    Response, logit, and DPO trainers
└── utils/       Config, environment, and tokenizer helpers
~~~

## Key Files

| Path | Responsibility |
|---|---|
| `data/response.py` | Canonical response rows and schema validation |
| `data/preference.py` | Canonical preference rows and schema validation |
| `models/resolve.py` | Local/Hugging Face model resolution |
| `training/` | Training plans, validators, dataset preparation, and trainers |
| `eval/run_eval.py` | Branch evaluation orchestration |
| `export/export_model.py` | Local export validation and optional Hub upload |
| `artifacts/handoff.py` | Checksummed artifact staging and transfer |

## How It Fits In

Scripts import this package and the Makefile invokes those scripts. YAML files
under `configs/` remain the runtime source of truth.

## Usage/API

Prefer the supported Make targets documented in [`COMMAND.md`](../COMMAND.md).
Tests may import side-effect-free plan builders, config loaders, validators,
and row converters directly.

## Conventions

- Keep imports of heavyweight ML libraries inside execution functions when
  dry-run/config paths do not need them.
- Keep config parsing typed and validation errors explicit.
- Keep response and logit branch implementations swappable through config.
- Do not embed provider calls or dataset-generation logic in this package.

## Gotchas

Logit training requires compatible teacher/student tokenizers and exactly one
visible supported GPU.
