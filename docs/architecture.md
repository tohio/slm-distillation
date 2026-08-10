# Architecture

System boundaries, data flow, and checkpoint handoffs for both distillation
branches.

## System Flow

~~~text
slm-synthetic-data
├── tohio/slm-synthetic-distillation-sft
└── tohio/slm-synthetic-distillation-dpo
              │
              ├──────────────────────────────────────────────┐
              │                                              │
              v                                              v
  response distillation                          local logit distillation
  SmolLM2-135M student                           SmolLM2-1.7B teacher
  teacher text only                              SmolLM2-135M student
              │                                              │
              v                                              v
  response-distilled checkpoint                  logit-distilled checkpoint
              │                                              │
              └───────────────┐              ┌───────────────┘
                              v              v
                          branch-specific DPO
                              │              │
                              v              v
                          evaluation and export
~~~

## Components

| Component | Responsibility |
|---|---|
| `configs/` | Declares model references, schemas, hyperparameters, and output contracts |
| `distill/data/` | Loads and validates published response and preference datasets |
| `distill/models/` | Resolves local paths and Hugging Face model references |
| `distill/training/` | Implements response, logit, and DPO training |
| `distill/eval/` | Generates responses and scores response/preference behavior |
| `distill/export/` | Builds model cards and optionally uploads models to Hugging Face |
| `distill/artifacts/` | Packages, verifies, and transfers run artifacts |
| `scripts/` | Exposes the package operations as command-line entry points |
| `Makefile` | Provides the supported operator command surface |

## Branch Boundaries

### Response branch

Response distillation is offline. The training repository reads already
generated teacher responses from
`tohio/slm-synthetic-distillation-sft`; it does not call OpenRouter or host a
teacher model. The response checkpoint becomes the source for response-branch
DPO.

### Logit branch

Logit distillation performs teacher and student forward passes during
training. The teacher is frozen, the student is trainable, and the published
response dataset supplies prompts and hard-label responses. The resulting
checkpoint becomes the source for logit-branch DPO.

### Shared downstream stages

Both branches use the same preference dataset and equivalent DPO,
evaluation, export, and artifact code. Branch-specific configs keep checkpoint
paths and model identities separate.

## Checkpoint Contracts

| Branch | Distillation output | DPO output |
|---|---|---|
| Response full | `runs/smollm2-135m-response-distilled/response_distill/checkpoints/final` | `runs/smollm2-135m-response-distilled/dpo/checkpoints/final` |
| Response smoke | `runs/smoke/smollm2-135m-response-distilled/response_distill/checkpoints/final` | `runs/smoke/smollm2-135m-response-distilled/dpo/checkpoints/final` |
| Logit full | `runs/smollm2-135m-logit-distilled/logit_distill/checkpoints/final` | `runs/smollm2-135m-logit-distilled/dpo/checkpoints/final` |
| Logit smoke | `runs/smoke/smollm2-135m-logit-distilled/logit_distill/checkpoints/final` | `runs/smoke/smollm2-135m-logit-distilled/dpo/checkpoints/final` |

Smoke and full configs never share checkpoint directories. Evaluation configs
consume the corresponding branch and run class.

## Repository Boundary

This repository owns training, alignment, evaluation, export, and model
artifact handoff. It does not own prompt generation, hosted teacher inference,
response validation during generation, dataset construction, or dataset
publishing.

## See Also

- [Training](training.md)
- [Configuration](configuration.md)
- [Evaluation and Export](evaluation-and-export.md)
- [Logit Distillation](logit_distillation.md)
