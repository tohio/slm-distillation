# Command Reference

Lookup reference for supported Make targets and runtime variables. Complete
branch sequences live in [`docs/training.md`](docs/training.md).

## General

| Target | Purpose |
|---|---|
| `make help` | Print the common command summary |
| `make install` | Install `requirements.txt` |
| `make test` | Run the full pytest suite |
| `make test-unit` | Run tests under `tests/` |

## Response Distillation

| Target | Purpose |
|---|---|
| `make validate-response-inputs` | Resolve the response model/tokenizer and inspect response data |
| `make train-response-dry-run` | Print the full response training plan without loading inputs |
| `make train-response-smoke` | Run bounded response training with the isolated smoke config |
| `make train-response` | Run the configured full response stage |

## Logit Distillation

| Target | Purpose |
|---|---|
| `make validate-logit-inputs` | Resolve teacher/student models, compare tokenizers, and inspect response data |
| `make train-logit-dry-run` | Print the full logit training plan without loading inputs |
| `make train-logit-smoke` | Run bounded logit training with the isolated smoke config |
| `make train-logit` | Run the configured full logit stage |

## DPO

| Target | Purpose |
|---|---|
| `make validate-dpo-inputs` | Validate the response-branch source checkpoint and preference data |
| `make train-dpo-dry-run` | Print the response-branch DPO plan |
| `make train-dpo-smoke` | DPO-align the response smoke checkpoint |
| `make train-dpo` | DPO-align the full response checkpoint |
| `make validate-dpo-logit-inputs` | Validate the logit-branch source checkpoint and preference data |
| `make train-dpo-logit-dry-run` | Print the logit-branch DPO plan |
| `make train-dpo-logit-smoke` | DPO-align the logit smoke checkpoint |
| `make train-dpo-logit` | DPO-align the full logit checkpoint |

## Evaluation

| Target | Purpose |
|---|---|
| `make validate-eval-response-inputs` | Validate response-branch models and evaluation datasets |
| `make eval-response-dry-run` | Print the response evaluation plan |
| `make eval-response-smoke` | Evaluate base and response smoke checkpoints |
| `make eval-response` | Evaluate base and full response checkpoints |
| `make validate-eval-logit-inputs` | Validate logit-branch models and evaluation datasets |
| `make eval-logit-dry-run` | Print the logit evaluation plan |
| `make eval-logit-smoke` | Evaluate base and logit smoke checkpoints |
| `make eval-logit` | Evaluate base and full logit checkpoints |

## Export

| Target | Purpose |
|---|---|
| `make export-dry-run` | Print the response export plan |
| `make export` | Validate the response final checkpoint and write its model card |
| `make export-push` | Export and upload the response model to Hugging Face |
| `make export-logit-dry-run` | Print the logit export plan |
| `make export-logit` | Validate the logit final checkpoint and write its model card |
| `make export-logit-push` | Export and upload the logit model to Hugging Face |

## Artifacts

| Target | Purpose |
|---|---|
| `make pack-artifacts` | Stage and archive response-branch artifacts |
| `make verify-artifacts` | Verify the response artifact manifest |
| `make push-artifacts` | Push response artifacts to S3 |
| `make pull-artifacts` | Pull response artifacts from S3 |
| `make pack-artifacts-logit` | Stage and archive logit-branch artifacts |
| `make verify-artifacts-logit` | Verify the logit artifact manifest |
| `make push-artifacts-logit` | Push logit artifacts to S3 |
| `make pull-artifacts-logit` | Pull logit artifacts from S3 |
| `make unpack-artifacts ARTIFACT=path/to/bundle.tar.gz` | Unpack a local artifact bundle |

## Configuration Variables

| Variable | Default | Used by |
|---|---|---|
| `RESPONSE_CONFIG` | `configs/response_distill.yaml` | response full dry-run, validation, and training |
| `RESPONSE_SMOKE_CONFIG` | `configs/response_distill_smoke.yaml` | response smoke training |
| `DPO_CONFIG` | `configs/dpo.yaml` | response-branch DPO full dry-run, validation, and training |
| `DPO_SMOKE_CONFIG` | `configs/dpo_smoke.yaml` | response-branch DPO smoke |
| `LOGIT_CONFIG` | `configs/logit_distill.yaml` | logit full dry-run, validation, and training |
| `LOGIT_SMOKE_CONFIG` | `configs/logit_distill_smoke.yaml` | logit smoke training |
| `DPO_LOGIT_CONFIG` | `configs/dpo_logit.yaml` | logit-branch DPO full dry-run, validation, and training |
| `DPO_LOGIT_SMOKE_CONFIG` | `configs/dpo_logit_smoke.yaml` | logit-branch DPO smoke |
| `EVAL_CONFIG` | `configs/eval.yaml` | response full evaluation and validation |
| `EVAL_SMOKE_CONFIG` | `configs/eval_smoke.yaml` | response smoke evaluation |
| `EVAL_LOGIT_CONFIG` | `configs/eval_logit.yaml` | logit full evaluation and validation |
| `EVAL_LOGIT_SMOKE_CONFIG` | `configs/eval_logit_smoke.yaml` | logit smoke evaluation |
| `EXPORT_CONFIG` | `configs/export.yaml` | response export |
| `EXPORT_LOGIT_CONFIG` | `configs/export_logit.yaml` | logit export |
| `ARTIFACT_CONFIG` | `configs/artifacts.yaml` | response artifact operations |
| `ARTIFACT_LOGIT_CONFIG` | `configs/artifacts_logit.yaml` | logit artifact operations |

## Runtime Variables

| Variable | Default | Purpose |
|---|---|---|
| `PYTHON` | `python3` | Python interpreter for scripts |
| `PYTHONPATH` | `.` | Package import root |
| `RESPONSE_DATA_LIMIT` | `100` | Rows inspected by response validation |
| `RESPONSE_MAX_STEPS` | unset | Override full response max steps |
| `RESPONSE_MAX_TRAIN_SAMPLES` | unset | Bound full response training rows |
| `RESPONSE_RESUME_FROM_CHECKPOINT` | unset | Resume response training |
| `RESPONSE_LAUNCH` | `python3` | Response launcher, including Accelerate commands |
| `RESPONSE_SMOKE_STEPS` | `5` | Response smoke steps |
| `RESPONSE_SMOKE_SAMPLES` | `64` | Response smoke rows |
| `DPO_DATA_LIMIT` | `100` | Rows inspected by DPO validation |
| `DPO_MAX_STEPS` | unset | Override either full DPO target's max steps |
| `DPO_MAX_TRAIN_SAMPLES` | unset | Bound either full DPO target's training rows |
| `DPO_RESUME_FROM_CHECKPOINT` | unset | Resume either DPO branch |
| `DPO_LAUNCH` | `python3` | DPO launcher, including Accelerate commands |
| `DPO_SMOKE_STEPS` | `5` | Either DPO smoke target's steps |
| `DPO_SMOKE_SAMPLES` | `64` | Either DPO smoke target's rows |
| `LOGIT_DATA_LIMIT` | `100` | Rows inspected by logit validation |
| `LOGIT_MAX_STEPS` | unset | Override full logit max steps |
| `LOGIT_MAX_TRAIN_SAMPLES` | unset | Bound full logit training rows |
| `LOGIT_RESUME_FROM_CHECKPOINT` | unset | Resume logit training |
| `LOGIT_SMOKE_STEPS` | `5` | Logit smoke steps |
| `LOGIT_SMOKE_SAMPLES` | `64` | Logit smoke rows |
| `EVAL_LIMIT` | unset | Override full evaluation/validation row count |
| `EVAL_SMOKE_LIMIT` | `20` | Smoke evaluation row count |
| `ARTIFACT` | required for `unpack-artifacts` | Bundle path to unpack |

Override variables on the same invocation:

~~~bash
make eval-logit EVAL_LIMIT=50
make train-response RESPONSE_MAX_STEPS=100
make unpack-artifacts ARTIFACT=artifacts/model-run.tar.gz
~~~

## Environment Variables

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Private Hugging Face access, improved Hub rate limits, and model upload |
| `WANDB_API_KEY` | Enable authenticated W&B experiment tracking |
| `WANDB_PROJECT` | W&B project name; defaults to `slm-distillation` |
| `WANDB_ENTITY` | Optional W&B team or account |
| `WANDB_MODE` | Set to `offline` for local W&B recording or `disabled` to force-disable |
| `S3_BUCKET` | Artifact handoff bucket |
| `S3_PREFIX` | Optional artifact key prefix |
| `AWS_ACCESS_KEY_ID` | S3 client credential |
| `AWS_SECRET_ACCESS_KEY` | S3 client credential |
| `AWS_DEFAULT_REGION` | S3 client region |
| `CUDA_VISIBLE_DEVICES` | Select the single GPU used by logit training |
