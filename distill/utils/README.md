# Utilities

## Purpose

`distill/utils` owns shared configuration, environment, tokenizer
compatibility, IO, and logging helpers. It does not own stage orchestration or
model training.

## Contents

~~~text
utils/
├── config.py
├── env.py
├── hardware.py
├── io.py
├── logging.py
└── tokenizer_compat.py
~~~

## Key Files

| File | Responsibility |
|---|---|
| `config.py` | Typed YAML loaders and stage-specific validation |
| `env.py` | `.env` and process-environment lookup |
| `hardware.py` | Single-visible-CUDA-GPU training contract |
| `tokenizer_compat.py` | Vocabulary, token-ID, and special-token comparison |
| `io.py` | Reserved shared IO helpers |
| `logging.py` | Reserved shared logging helpers |

## How It Fits In

Every stage loads validated dataclasses from `config.py`. Logit input
validation additionally uses `tokenizer_compat.py` before model training.

## Usage/API

Import the stage loader matching the config being consumed, such as
`load_response_distill_config`, `load_dpo_config`, or `load_eval_config`.
Operator-facing use should go through scripts or Make targets.

## Conventions

- Fail early on missing, mistyped, unsupported, or non-positive values.
- Keep config loaders free of model or dataset downloads.
- Read secrets from `.env` or the process environment, never YAML.
- Return structured compatibility results before raising at hard gates.

## Gotchas

`io.py` and `logging.py` are placeholders; do not document behavior they do not
yet implement.
