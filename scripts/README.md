# Scripts

## Purpose

`scripts` owns CLI argument parsing and translation into `distill` package
operations. It does not contain core training, evaluation, export, or artifact
logic.

## Contents

~~~text
scripts/
├── train_response_distill.py
├── train_logit_distill.py
├── train_dpo.py
├── run_eval.py
├── export_model.py
├── pack_artifacts.py
├── verify_artifacts.py
├── unpack_artifacts.py
├── push_artifacts.py
└── pull_artifacts.py
~~~

## Key Files

| Script | Modes |
|---|---|
| `train_response_distill.py` | dry-run, validate inputs, train, bound, resume |
| `train_logit_distill.py` | dry-run, validate inputs, train, bound, resume |
| `train_dpo.py` | dry-run, validate inputs, train, bound, resume |
| `run_eval.py` | dry-run, validate inputs, evaluate with optional limit |
| `export_model.py` | dry-run, local export, explicit Hugging Face push |
| Artifact scripts | pack, verify, unpack, push, and pull |

## How It Fits In

The Makefile is the supported operator interface and invokes these scripts with
the correct branch config and runtime overrides.

## Usage/API

~~~bash
python3 scripts/train_response_distill.py --help
python3 scripts/train_logit_distill.py --help
python3 scripts/train_dpo.py --help
python3 scripts/run_eval.py --help
python3 scripts/export_model.py --help
~~~

Use [`COMMAND.md`](../COMMAND.md) for Make targets and variables.

## Conventions

- Keep defaults aligned with the response full configs.
- Keep `--dry-run` side-effect free.
- Keep `--validate-inputs` separate from training.
- Print structured JSON for plans, validation, and evaluation results.
- Require an explicit `--push-to-hub` for CLI-triggered Hub mutation.

## Gotchas

Downstream validation fails until its upstream final checkpoint exists. Direct
script use must set `PYTHONPATH=.` when the package is not installed.
