# Export

## Purpose

`distill/export` owns provenance model-card generation, final checkpoint
validation, and optional Hugging Face model upload. It does not train models or
package S3 artifacts.

## Contents

~~~text
export/
├── export_model.py
└── model_card.py
~~~

## Key Files

| File | Responsibility |
|---|---|
| `export_model.py` | Builds export plans, validates paths, and orchestrates Hub uploads |
| `model_card.py` | Builds and writes provenance-focused model cards |

## How It Fits In

Each export config consumes the matching branch's full DPO checkpoint and
evaluation result. The generated model card is also included by the matching
artifact config.

## Usage/API

~~~bash
make export-dry-run
make export-logit-dry-run
make export
make export-logit
~~~

Explicit upload targets are `make export-push` and
`make export-logit-push`.

## Conventions

- Validate checkpoint/tokenizer paths before writing or uploading.
- Preserve source, teacher, dataset, DPO, and evaluation provenance.
- Keep response and logit repositories/configs separate.
- Require an explicit push target or enabled config for remote mutation.

## Gotchas

Hugging Face uploads require `HF_TOKEN`. `export` and `export-logit` write
model cards but do not push with the default configs.
