# Artifact Handoff

## Purpose

`distill/artifacts` owns checksummed staging, archive creation, verification,
unpacking, and optional S3 synchronization for completed model runs. It does
not publish datasets or models to Hugging Face.

## Contents

~~~text
artifacts/
└── handoff.py
~~~

## Key Files

| File | Responsibility |
|---|---|
| `handoff.py` | Loads artifact configs, stages files, writes manifests, packs archives, verifies checksums, and transfers S3 objects |

## How It Fits In

Branch-specific artifact configs collect final distillation and DPO
checkpoints, evaluation results, and the generated model card after a full
run.

## Usage/API

~~~bash
make pack-artifacts
make verify-artifacts
make pack-artifacts-logit
make verify-artifacts-logit
~~~

Use the push/pull targets only when S3 handoff is required.

## Conventions

- Include generated model outputs, evaluation, and provenance.
- Reference published datasets by ID instead of duplicating them.
- Verify manifests before transfer or server handoff.
- Keep response and logit artifact roots separate.

## Gotchas

Packing requires every configured `required` file to exist. S3 operations
require the configured bucket/prefix variables and AWS credentials.
