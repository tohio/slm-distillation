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

## External Storage

The default external target is a Hugging Face dataset repo:

    tohio/slm-distillation-artifacts

Artifacts are organized under the configured run name.

## Authentication

Artifact push and pull read Hugging Face credentials from `.env`.

Required variable:

    HF_TOKEN=...

`HUGGINGFACE_HUB_TOKEN` is also supported.

Interactive `hf auth login` is not required for repository workflows.
