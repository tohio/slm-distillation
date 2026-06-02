# Artifacts

Required generated JSONL artifact handoff utilities.

## Files

| File | Purpose |
|---|---|
| `handoff.py` | Stage, verify, pack, unpack, push, and pull generated JSONL artifacts. |

## Stored Data

Artifact handoff stores generated JSONL data:

| Directory | Purpose |
|---|---|
| `data/raw_teacher/` | Raw teacher responses. |
| `data/validated/` | Validated teacher responses. |
| `data/rejected/` | Rejected teacher responses or rejected preference pairs. |
| `data/distill/` | Response-distillation training JSONL. |
| `data/preference/` | DPO preference-pair JSONL. |

## Handoff Contract

The required training handoff files are:

    data/distill/response_distill.jsonl
    data/preference/dpo_pairs.jsonl

The manifest records file paths, sizes, and SHA256 checksums.

## Storage

Generated artifacts are stored in S3.

Set `artifact.s3_uri` in `configs/artifacts.yaml`:

    s3://YOUR_BUCKET/slm-distillation/slm-125m-deepseek-distilled/

AWS authentication uses the standard boto3 credential chain:

- instance profile
- `AWS_PROFILE`
- environment variables
- shared AWS config files
