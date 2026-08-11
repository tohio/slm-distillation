# Training

## Purpose

`distill/training` owns response distillation, local logit distillation, and
DPO training. It does not own CLI parsing, evaluation, export, or dataset
generation.

## Contents

~~~text
training/
├── train_response_distill.py
├── train_logit_distill.py
└── train_dpo.py
~~~

## Key Files

| File | Input | Output |
|---|---|---|
| `train_response_distill.py` | Published response rows and a student model | Response-distilled checkpoint |
| `train_logit_distill.py` | Local teacher/student plus published response rows | Logit-distilled checkpoint |
| `train_dpo.py` | Branch-specific distilled checkpoint and preference rows | DPO-aligned checkpoint |

Each module exposes a side-effect-free plan builder, an input validator, data
preparation, and the training operation.

## How It Fits In

Response and logit training create independent branch checkpoints. Matching
DPO configs consume those checkpoints before evaluation and export.

## Usage/API

~~~bash
make train-response-dry-run
make train-logit-dry-run
make train-dpo-dry-run
make train-dpo-logit-dry-run
~~~

Use [the training guide](../../docs/training.md) for complete smoke and
production sequences.

## Conventions

- Compute causal loss only on supervised response tokens.
- Combine DPO preference loss with chosen-response SFT anchoring.
- Save model and tokenizer together in each final checkpoint.
- Keep smoke and full outputs in separate roots.
- Treat configured final checkpoint paths as downstream contracts.
- Support bounded steps/samples and resume where the trainer exposes them.

## Gotchas

DPO validation requires its upstream final checkpoint to exist. Logit training
requires matching tokenizers and exactly one visible supported CUDA GPU. The
logit validator checks metadata, tokenizers, and data but does not load model
weights or execute CUDA.
