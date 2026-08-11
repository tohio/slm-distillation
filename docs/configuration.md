# Configuration

Reference for YAML ownership, branch pairing, runtime overrides, and safe model
or dataset substitution.

## Configuration Inventory

| Stage | Full response | Smoke response | Full logit | Smoke logit |
|---|---|---|---|---|
| Distillation | `response_distill.yaml` | `response_distill_smoke.yaml` | `logit_distill.yaml` | `logit_distill_smoke.yaml` |
| DPO | `dpo.yaml` | `dpo_smoke.yaml` | `dpo_logit.yaml` | `dpo_logit_smoke.yaml` |
| Evaluation | `eval.yaml` | `eval_smoke.yaml` | `eval_logit.yaml` | `eval_logit_smoke.yaml` |
| Export | `export.yaml` | — | `export_logit.yaml` | — |
| Artifacts | `artifacts.yaml` | — | `artifacts_logit.yaml` | — |

All files live under `configs/`.

## Configuration Contracts

### Training

Training configs define model/tokenizer references, dataset IDs and field
mappings, formatting, hyperparameters, and output paths. Hugging Face IDs and
local checkpoint directories use the same fields.

### DPO

DPO configs must consume the final checkpoint from the matching distillation
config. The source checkpoint and tokenizer paths are identical by default.
Preference fields are normalized to `prompt`, `chosen`, and `rejected`.
`loss_type` accepts one loss name or a list, and `loss_weights` must match the
list length when multiple objectives are configured. All default response and
logit branches use only `loss_type: sigmoid`, the original DPO objective; they
do not configure `loss_weights`. See the [DPO paper](https://arxiv.org/abs/2305.18290)
and [TRL DPO loss documentation](https://huggingface.co/docs/trl/dpo_trainer#loss-types).

### Evaluation

The first configured model is the comparison baseline. Response data supplies
prompts and references; preference data supplies chosen/rejected pairs.
Evaluation output paths must remain inside the matching branch root.

### Export and artifacts

Export configs consume the matching full DPO checkpoint and reference the
matching evaluation result. Artifact configs package the distillation final,
DPO final, evaluation directory, and generated model card for the same branch.

## Runtime Overrides

The Makefile exposes config, validation limit, bounded training, resume,
launcher, and evaluation limit variables. See [COMMAND.md](../COMMAND.md) for
the complete list.

Examples:

~~~bash
make train-response RESPONSE_CONFIG=configs/response_distill.yaml
make eval-logit EVAL_LIMIT=50
make pack-artifacts-logit ARTIFACT_LOGIT_CONFIG=configs/artifacts_logit.yaml
~~~

## Swapping Models

For response distillation, update:

1. `model.model_name_or_path` and `model.tokenizer_name_or_path`;
2. the response output model name and all response checkpoint paths;
3. the downstream response DPO, evaluation, export, and artifact paths;
4. export provenance and repository name.

For logit distillation, also update both teacher and student references and run
the tokenizer compatibility validator before training. The current logit loss
requires identical vocabulary/token IDs and special-token mappings.

When increasing student size, lower the per-device batch size and increase
gradient accumulation if needed to preserve the intended effective batch.

## Swapping Datasets

Update the dataset ID, optional config name, split, and every field mapping.
Response validation rejects missing fields, empty prompt/response text, and
duplicate IDs. Preference validation additionally rejects mixed row formats
and identical chosen/rejected responses.

The response-branch model identity describes the distillation method, so a
dataset swap does not require renaming `smollm2-135m-response-distilled`.
Update `model_card.response_dataset` in `configs/export.yaml` to preserve
dataset provenance. If the replacement response dataset uses a different
teacher, also update `model_card.teacher_model`. Provider provenance belongs
to the published dataset and is not duplicated in exported model metadata.

The default evaluation configs sample published training splits for pipeline
validation. Point both evaluation datasets at independent held-out data before
making final quality claims.

## Environment

`.env` may provide:

- `HF_TOKEN` for private Hugging Face resources, rate limits, and pushes;
- `WANDB_API_KEY`, `WANDB_PROJECT`, and `WANDB_ENTITY` for optional online
  experiment tracking;
- `WANDB_MODE=offline` for local W&B recording without online authentication;
- `S3_BUCKET` and `S3_PREFIX` for artifact handoff;
- AWS credentials and region values used by the S3 client.

Environment variables already present in the process remain usable.

## See Also

- [Architecture](architecture.md)
- [Training](training.md)
- [Evaluation and Export](evaluation-and-export.md)
- [`configs/README.md`](../configs/README.md)
