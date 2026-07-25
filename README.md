# SLM Distillation

Training, evaluation, and export workflows for distilled language models.

This repository consumes published datasets and model checkpoints. Synthetic
teacher generation is owned by
[`slm-synthetic-data`](https://github.com/tohio/slm-synthetic-data).

## Inputs

| Input | Default |
|---|---|
| Response dataset | `tohio/slm-synthetic-distillation-sft` |
| Preference dataset | `tohio/slm-synthetic-distillation-dpo` |
| Student checkpoint | Configurable Hugging Face model or local checkpoint |
| Logit teacher | Configurable Hugging Face model or local checkpoint |

The default production datasets contain 30,000 response-distillation records
and 15,000 DPO preference pairs.

## Workflows

### Response distillation

Response distillation is offline supervised fine-tuning:

```text
published response dataset
  -> configured student checkpoint
  -> response-distilled checkpoint
```

The teacher is not loaded or queried during training. Teacher provenance is
retained in dataset metadata and the exported model card.

### DPO

DPO aligns the response-distilled checkpoint with the published preference
dataset:

```text
response-distilled checkpoint
  + published preference dataset
  -> DPO-aligned distilled checkpoint
```

### Logit distillation

Logit distillation runs the teacher and student locally during training:

```text
local teacher logits
  + student logits
  -> distillation loss
  -> logit-distilled checkpoint
```

Teacher and student tokenizers must have identical vocabularies, token IDs, and
special tokens. The compatibility gate runs before training.

## Model Swapping

Models are selected through configuration. Training code must not contain
model-specific paths or architecture branches.

Response distillation permits unrelated teacher and student tokenizers because
the student tokenizes saved text. Logit distillation requires tokenizer
compatibility.

Initial development models:

| Role | Model |
|---|---|
| Student | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| Local logit teacher | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |

SLM checkpoints can replace the SmolLM2 student when they are available.

## Repository Scope

Included:

- response-distillation training
- DPO training
- local logit distillation
- tokenizer compatibility checks
- evaluation
- model export and model-card generation
- model artifact handoff

Excluded:

- prompt generation
- hosted teacher inference
- provider routing and retry controls
- teacher-response validation
- response or preference dataset construction
- dataset publishing

## Status

| Component | Status |
|---|---|
| Dataset production | External; published by `slm-synthetic-data` |
| Artifact handoff | Implemented |
| Tokenizer compatibility gate | Implemented |
| Model-card generation | Implemented |
| Response trainer | Not implemented |
| DPO trainer | Configuration and dry-run plan only |
| Logit trainer | Configuration and compatibility plan only |
| Evaluation | Not implemented |
| Hugging Face model export | Not implemented |

## Installation

Requirements:

- Python 3.12+
- CUDA-capable GPU for training
- Hugging Face access for model and dataset downloads
- S3 credentials only when using artifact handoff

```bash
git clone https://github.com/tohio/slm-distillation.git
cd slm-distillation

cp .env.sample .env
make install
```

## Commands

```bash
make help
make test

make train-dpo-dry-run
make train-logit-dry-run
make export-dry-run
```

Training targets raise a clear `NotImplementedError` until their trainer is
implemented. Dry-run targets validate and print the resolved configuration.

## Configuration

| File | Purpose |
|---|---|
| `configs/dpo.yaml` | DPO source model, published dataset, training, and output settings |
| `configs/logit_distill.yaml` | Local teacher/student and logit-distillation settings |
| `configs/eval.yaml` | Evaluation settings |
| `configs/export.yaml` | Final checkpoint, model-card provenance, and export settings |
| `configs/artifacts.yaml` | Model artifact packaging and S3 handoff |

## Tests

```bash
make test
```

The test suite covers configuration loading, tokenizer compatibility, artifact
handoff, model-card generation, and export planning.

## Related Projects

- [`slm`](https://github.com/tohio/slm) — SLM model training
- [`slm-synthetic-data`](https://github.com/tohio/slm-synthetic-data) — synthetic dataset generation
- [`slm-reasoning`](https://github.com/tohio/slm-reasoning) — reasoning-model workflows

## License

MIT
