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
| Local logit teacher | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |

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

The default logit branch uses SmolLM2-1.7B-Instruct as teacher and
SmolLM2-135M-Instruct as student. Teacher and student tokenizers must have
identical vocabularies, token IDs, and special tokens. The compatibility gate
runs before training.

### Evaluation and export

Each branch evaluates the base, distilled, and DPO checkpoints on two signals:

- teacher-response exact match, normalized exact match, token F1, and output
  completeness;
- chosen-over-rejected preference accuracy based on conditional log
  probabilities.

The default configs sample the published training splits for pipeline
validation. For final reporting, point `data` and `preference_data` at
independent held-out splits or datasets.

Export validates the final checkpoint, writes a provenance model card, and can
push the checkpoint and tokenizer directly to Hugging Face.

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

The response trainer:

- formats records with the tokenizer chat template or plain text;
- masks prompt tokens so loss is computed only on teacher responses;
- truncates long records while preserving supervised response tokens;
- uses dynamic padding;
- supports bounded smoke runs and checkpoint resume;
- saves the model and tokenizer to the configured final checkpoint.

The DPO configuration also accepts local directories or Hugging Face model IDs.
Preference field names are configurable and are normalized to `prompt`,
`chosen`, and `rejected`. Standard text and conversational message-list
datasets are supported.

The DPO validator rejects:

- missing configured columns;
- empty prompts or completions;
- mixed standard and conversational fields;
- identical chosen and rejected responses;
- duplicate IDs.

DPO training uses the response-distilled checkpoint as both the initial policy
and fixed reference policy. The DPO-aligned model and tokenizer are saved to
the configured final checkpoint.

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
| DPO config and input validation | Implemented |
| Artifact handoff | Implemented |
| Tokenizer compatibility gate | Implemented |
| Model-card generation | Implemented |
| Response trainer | Implemented |
| DPO trainer | Implemented |
| Logit trainer | Implemented |
| Response and preference evaluation | Implemented |
| Hugging Face model export | Implemented |

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
make train-response RESPONSE_MAX_STEPS=5 RESPONSE_MAX_TRAIN_SAMPLES=64
make train-dpo-dry-run
make validate-dpo-inputs DPO_DATA_LIMIT=100
make train-dpo DPO_MAX_STEPS=5 DPO_MAX_TRAIN_SAMPLES=64
make train-logit-dry-run
make validate-logit-inputs LOGIT_DATA_LIMIT=100
make eval-response-dry-run
make eval-logit-dry-run
make export-dry-run
make export-logit-dry-run
```

Run isolated smoke workflows first:

```bash
make train-response-smoke
make train-dpo-smoke
make eval-response-smoke

make train-logit-smoke
make train-dpo-logit-smoke
make eval-logit-smoke
```

Smoke outputs are written under `runs/smoke/`; full outputs are written under
`runs/`. The two paths never share checkpoints.

Run either full branch with:

```bash
make train-response
make train-dpo
make eval-response
make export-push

make train-logit
make train-dpo-logit
make eval-logit
make export-logit-push
```

`HF_TOKEN` is required for private inputs and Hugging Face pushes. The local
logit trainer intentionally requires exactly one supported GPU because teacher
and student are colocated. Response and DPO can use an Accelerate launcher via
`RESPONSE_LAUNCH` and `DPO_LAUNCH`.

The default response config targets SmolLM2-135M with bfloat16. When swapping
models, update the source model, tokenizer, downstream checkpoint paths, model
card provenance, and export repository together.

## Configuration

| File | Purpose |
|---|---|
| `configs/response_distill.yaml` | Response student, formatting, training, dataset schema, and outputs |
| `configs/dpo.yaml` | DPO source model, published dataset, training, and outputs |
| `configs/logit_distill.yaml` | Local teacher/student and logit-distillation settings |
| `configs/dpo_logit.yaml` | DPO stage for the logit branch |
| `configs/eval.yaml`, `configs/eval_logit.yaml` | Branch evaluation settings |
| `configs/export.yaml`, `configs/export_logit.yaml` | Branch model-card and Hugging Face export settings |
| `configs/artifacts.yaml`, `configs/artifacts_logit.yaml` | Branch artifact packaging and S3 handoff |
| `configs/*_smoke.yaml` | Isolated bounded-run checkpoints and evaluation |

## Tests

```bash
make test
```

The test suite covers response and preference configuration, model resolution,
dataset schema conversion, response-only loss masking, sequence truncation,
DPO and logit configuration, tokenizer compatibility, evaluation output,
artifact handoff, model-card generation, and Hugging Face export orchestration.

## Related Projects

- [`slm`](https://github.com/tohio/slm) — SLM model training
- [`slm-synthetic-data`](https://github.com/tohio/slm-synthetic-data) — synthetic dataset generation
- [`slm-reasoning`](https://github.com/tohio/slm-reasoning) — reasoning-model workflows

## License

MIT
