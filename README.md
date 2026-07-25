# SLM Distillation

Training, evaluation, and export workflows for distilled language models.

This repository consumes published datasets and model checkpoints. Synthetic
teacher generation is owned by
[`slm-synthetic-data`](https://github.com/tohio/slm-synthetic-data).

## Inputs

| Input | Default |
|---|---|
| Student | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| Response dataset | `tohio/slm-synthetic-distillation-sft` |
| Preference dataset | `tohio/slm-synthetic-distillation-dpo` |
| Local logit teacher | Configurable local checkpoint |

The production datasets contain 30,000 response-distillation records and
15,000 DPO preference pairs.

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

## Model and Dataset Swapping

The response configuration accepts Hugging Face IDs or local model directories.
Model code does not contain architecture-specific paths.

Dataset schema fields are configurable:

```yaml
model:
  model_name_or_path: HuggingFaceTB/SmolLM2-135M-Instruct
  tokenizer_name_or_path: HuggingFaceTB/SmolLM2-135M-Instruct
  revision: main

data:
  dataset_id: tohio/slm-synthetic-distillation-sft
  dataset_config_name: null
  dataset_split: train
  id_field: id
  prompt_field: prompt
  response_field: response
  metadata_field: metadata
```

The input validator:

- resolves the model and tokenizer reference;
- loads the configured dataset split;
- verifies configured columns;
- converts rows to `id`, `prompt`, `response`, and `metadata`;
- rejects empty text and duplicate IDs.

Response distillation permits unrelated teacher and student tokenizers because
the student tokenizes saved text. Logit distillation requires tokenizer
compatibility.

## Repository Scope

Included:

- response-distillation training
- DPO training
- local logit distillation
- Hugging Face model and dataset input resolution
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
| Response config and input validation | Implemented |
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

make train-response-dry-run
make validate-response-inputs RESPONSE_DATA_LIMIT=100
make train-dpo-dry-run
make train-logit-dry-run
make export-dry-run
```

Training targets raise a clear `NotImplementedError` until their trainer is
implemented. Dry-run targets validate and print resolved configuration without
starting training.

## Configuration

| File | Purpose |
|---|---|
| `configs/response_distill.yaml` | Response student, dataset schema, and outputs |
| `configs/dpo.yaml` | DPO source model, published dataset, training, and outputs |
| `configs/logit_distill.yaml` | Local teacher/student and logit-distillation settings |
| `configs/eval.yaml` | Evaluation settings |
| `configs/export.yaml` | Final checkpoint, model-card provenance, and export settings |
| `configs/artifacts.yaml` | Model artifact packaging and S3 handoff |

## Tests

```bash
make test
```

The test suite covers response configuration, model resolution, dataset schema
conversion, DPO and logit configuration, tokenizer compatibility, artifact
handoff, model-card generation, and export planning.

## Related Projects

- [`slm`](https://github.com/tohio/slm) — SLM model training
- [`slm-synthetic-data`](https://github.com/tohio/slm-synthetic-data) — synthetic dataset generation
- [`slm-reasoning`](https://github.com/tohio/slm-reasoning) — reasoning-model workflows

## License

MIT
