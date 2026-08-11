# ============================================================
# SLM Distillation
# ============================================================

.RECIPEPREFIX := >

PYTHON := python3
PYTHONPATH := .
RESPONSE_CONFIG ?= configs/response_distill.yaml
RESPONSE_SMOKE_CONFIG ?= configs/response_distill_smoke.yaml
RESPONSE_DATA_LIMIT ?= 100
RESPONSE_MAX_STEPS ?=
RESPONSE_MAX_TRAIN_SAMPLES ?=
RESPONSE_RESUME_FROM_CHECKPOINT ?=
RESPONSE_SMOKE_STEPS ?= 5
RESPONSE_SMOKE_SAMPLES ?= 64
DPO_CONFIG ?= configs/dpo.yaml
DPO_SMOKE_CONFIG ?= configs/dpo_smoke.yaml
DPO_LOGIT_CONFIG ?= configs/dpo_logit.yaml
DPO_LOGIT_SMOKE_CONFIG ?= configs/dpo_logit_smoke.yaml
DPO_DATA_LIMIT ?= 100
DPO_MAX_STEPS ?=
DPO_MAX_TRAIN_SAMPLES ?=
DPO_RESUME_FROM_CHECKPOINT ?=
DPO_SMOKE_STEPS ?= 5
DPO_SMOKE_SAMPLES ?= 64
LOGIT_CONFIG ?= configs/logit_distill.yaml
LOGIT_SMOKE_CONFIG ?= configs/logit_distill_smoke.yaml
LOGIT_DATA_LIMIT ?= 100
LOGIT_MAX_STEPS ?=
LOGIT_MAX_TRAIN_SAMPLES ?=
LOGIT_RESUME_FROM_CHECKPOINT ?=
LOGIT_SMOKE_STEPS ?= 5
LOGIT_SMOKE_SAMPLES ?= 64
EVAL_CONFIG ?= configs/eval.yaml
EVAL_SMOKE_CONFIG ?= configs/eval_smoke.yaml
EVAL_LOGIT_CONFIG ?= configs/eval_logit.yaml
EVAL_LOGIT_SMOKE_CONFIG ?= configs/eval_logit_smoke.yaml
EVAL_LIMIT ?=
EVAL_SMOKE_LIMIT ?= 20
EXPORT_CONFIG ?= configs/export.yaml
EXPORT_LOGIT_CONFIG ?= configs/export_logit.yaml
ARTIFACT_CONFIG ?= configs/artifacts.yaml
ARTIFACT_LOGIT_CONFIG ?= configs/artifacts_logit.yaml

RESPONSE_TRAIN_ARGS = $(strip $(if $(RESPONSE_MAX_STEPS),--max-steps $(RESPONSE_MAX_STEPS)) $(if $(RESPONSE_MAX_TRAIN_SAMPLES),--max-train-samples $(RESPONSE_MAX_TRAIN_SAMPLES)) $(if $(RESPONSE_RESUME_FROM_CHECKPOINT),--resume-from-checkpoint $(RESPONSE_RESUME_FROM_CHECKPOINT)))
DPO_TRAIN_ARGS = $(strip $(if $(DPO_MAX_STEPS),--max-steps $(DPO_MAX_STEPS)) $(if $(DPO_MAX_TRAIN_SAMPLES),--max-train-samples $(DPO_MAX_TRAIN_SAMPLES)) $(if $(DPO_RESUME_FROM_CHECKPOINT),--resume-from-checkpoint $(DPO_RESUME_FROM_CHECKPOINT)))
LOGIT_TRAIN_ARGS = $(strip $(if $(LOGIT_MAX_STEPS),--max-steps $(LOGIT_MAX_STEPS)) $(if $(LOGIT_MAX_TRAIN_SAMPLES),--max-train-samples $(LOGIT_MAX_TRAIN_SAMPLES)) $(if $(LOGIT_RESUME_FROM_CHECKPOINT),--resume-from-checkpoint $(LOGIT_RESUME_FROM_CHECKPOINT)))
EVAL_ARGS = $(strip $(if $(EVAL_LIMIT),--limit $(EVAL_LIMIT)))

.PHONY: help install test test-unit validate-response-inputs train-response train-response-smoke train-response-dry-run validate-dpo-inputs train-dpo train-dpo-smoke train-dpo-dry-run validate-logit-inputs train-logit train-logit-smoke train-logit-dry-run validate-dpo-logit-inputs train-dpo-logit train-dpo-logit-smoke train-dpo-logit-dry-run eval-response eval-response-smoke eval-response-dry-run validate-eval-response-inputs eval-logit eval-logit-smoke eval-logit-dry-run validate-eval-logit-inputs export export-dry-run export-push export-logit export-logit-dry-run export-logit-push verify-artifacts verify-artifacts-logit pack-artifacts pack-artifacts-logit unpack-artifacts push-artifacts push-artifacts-logit pull-artifacts pull-artifacts-logit

help:
> @echo ""
> @echo "SLM Distillation"
> @echo "================"
> @echo ""
> @echo "Setup:"
> @echo "  install                 Install Python dependencies"
> @echo ""
> @echo "Training:"
> @echo "  validate-response-inputs Validate the response model and dataset"
> @echo "  train-response          Train the response-distillation stage"
> @echo "  train-response-smoke    Run a bounded response smoke"
> @echo "  train-response-dry-run  Print the resolved response plan"
> @echo "  validate-dpo-inputs     Validate the DPO model and dataset"
> @echo "  train-dpo               Train the DPO stage"
> @echo "  train-dpo-smoke         Run a bounded response-DPO smoke"
> @echo "  train-dpo-dry-run       Print the resolved DPO plan"
> @echo "  validate-logit-inputs   Validate logit models, tokenizers, and data"
> @echo "  train-logit             Train with local teacher logits"
> @echo "  train-logit-smoke       Run a bounded logit smoke"
> @echo "  train-logit-dry-run     Print the resolved logit plan"
> @echo "  train-dpo-logit         DPO-align the logit-distilled checkpoint"
> @echo "  train-dpo-logit-smoke   Run a bounded logit-DPO smoke"
> @echo ""
> @echo "Evaluation:"
> @echo "  eval-response           Evaluate the response branch"
> @echo "  eval-response-smoke     Run bounded response evaluation"
> @echo "  eval-logit              Evaluate the logit branch"
> @echo "  eval-logit-smoke        Run bounded logit evaluation"
> @echo ""
> @echo "Export:"
> @echo "  export                  Validate the response export and model card"
> @echo "  export-push             Push the response model to Hugging Face"
> @echo "  export-dry-run          Print the resolved export plan"
> @echo "  export-logit            Validate the logit export and model card"
> @echo "  export-logit-push       Push the logit model to Hugging Face"
> @echo ""
> @echo "Artifacts:"
> @echo "  verify-artifacts        Verify an artifact manifest"
> @echo "  verify-artifacts-logit  Verify a logit artifact manifest"
> @echo "  pack-artifacts          Build a local artifact bundle"
> @echo "  pack-artifacts-logit    Build a logit artifact bundle"
> @echo "  unpack-artifacts        Unpack a local artifact bundle"
> @echo "  push-artifacts          Push run artifacts to S3"
> @echo "  pull-artifacts          Pull run artifacts from S3"
> @echo ""
> @echo "Tests:"
> @echo "  test                    Run the full test suite"
> @echo "  test-unit               Run unit tests"
> @echo ""

install:
> $(PYTHON) -m pip install -r requirements.txt

test:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest

test-unit:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests

validate-response-inputs:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_response_distill.py \
>   --config $(RESPONSE_CONFIG) \
>   --validate-inputs \
>   --limit $(RESPONSE_DATA_LIMIT)

train-response-dry-run:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_response_distill.py \
>   --config $(RESPONSE_CONFIG) \
>   --dry-run

train-response:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_response_distill.py --config $(RESPONSE_CONFIG)$(if $(RESPONSE_TRAIN_ARGS), $(RESPONSE_TRAIN_ARGS))

train-response-smoke:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_response_distill.py \
>   --config $(RESPONSE_SMOKE_CONFIG) \
>   --max-steps $(RESPONSE_SMOKE_STEPS) \
>   --max-train-samples $(RESPONSE_SMOKE_SAMPLES)

verify-artifacts:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/verify_artifacts.py \
>   --manifest artifacts/smollm2-135m-response-distilled/manifest.json

verify-artifacts-logit:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/verify_artifacts.py \
>   --manifest artifacts/smollm2-135m-logit-distilled/manifest.json

pack-artifacts:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/pack_artifacts.py \
>   --config $(ARTIFACT_CONFIG)

pack-artifacts-logit:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/pack_artifacts.py \
>   --config $(ARTIFACT_LOGIT_CONFIG)

unpack-artifacts:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/unpack_artifacts.py $(ARTIFACT)

push-artifacts:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/push_artifacts.py \
>   --config $(ARTIFACT_CONFIG)

push-artifacts-logit:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/push_artifacts.py \
>   --config $(ARTIFACT_LOGIT_CONFIG)

pull-artifacts:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/pull_artifacts.py \
>   --config $(ARTIFACT_CONFIG)

pull-artifacts-logit:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/pull_artifacts.py \
>   --config $(ARTIFACT_LOGIT_CONFIG)

train-logit:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_logit_distill.py --config $(LOGIT_CONFIG)$(if $(LOGIT_TRAIN_ARGS), $(LOGIT_TRAIN_ARGS))

train-logit-smoke:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_logit_distill.py \
>   --config $(LOGIT_SMOKE_CONFIG) \
>   --max-steps $(LOGIT_SMOKE_STEPS) \
>   --max-train-samples $(LOGIT_SMOKE_SAMPLES)

train-logit-dry-run:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_logit_distill.py \
>   --config $(LOGIT_CONFIG) \
>   --dry-run

validate-logit-inputs:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_logit_distill.py \
>   --config $(LOGIT_CONFIG) \
>   --validate-inputs \
>   --limit $(LOGIT_DATA_LIMIT)

train-dpo:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_dpo.py --config $(DPO_CONFIG)$(if $(DPO_TRAIN_ARGS), $(DPO_TRAIN_ARGS))

train-dpo-smoke:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_dpo.py \
>   --config $(DPO_SMOKE_CONFIG) \
>   --max-steps $(DPO_SMOKE_STEPS) \
>   --max-train-samples $(DPO_SMOKE_SAMPLES)

train-dpo-dry-run:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_dpo.py \
>   --config $(DPO_CONFIG) \
>   --dry-run

validate-dpo-inputs:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_dpo.py \
>   --config $(DPO_CONFIG) \
>   --validate-inputs \
>   --limit $(DPO_DATA_LIMIT)

train-dpo-logit:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_dpo.py --config $(DPO_LOGIT_CONFIG)$(if $(DPO_TRAIN_ARGS), $(DPO_TRAIN_ARGS))

train-dpo-logit-smoke:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_dpo.py \
>   --config $(DPO_LOGIT_SMOKE_CONFIG) \
>   --max-steps $(DPO_SMOKE_STEPS) \
>   --max-train-samples $(DPO_SMOKE_SAMPLES)

train-dpo-logit-dry-run:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_dpo.py \
>   --config $(DPO_LOGIT_CONFIG) \
>   --dry-run

validate-dpo-logit-inputs:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_dpo.py \
>   --config $(DPO_LOGIT_CONFIG) \
>   --validate-inputs \
>   --limit $(DPO_DATA_LIMIT)

eval-response:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_eval.py --config $(EVAL_CONFIG)$(if $(EVAL_ARGS), $(EVAL_ARGS))

eval-response-smoke:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_eval.py \
>   --config $(EVAL_SMOKE_CONFIG) \
>   --limit $(EVAL_SMOKE_LIMIT)

eval-response-dry-run:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_eval.py \
>   --config $(EVAL_CONFIG) \
>   --dry-run

validate-eval-response-inputs:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_eval.py --config $(EVAL_CONFIG) --validate-inputs$(if $(EVAL_ARGS), $(EVAL_ARGS))

eval-logit:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_eval.py --config $(EVAL_LOGIT_CONFIG)$(if $(EVAL_ARGS), $(EVAL_ARGS))

eval-logit-smoke:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_eval.py \
>   --config $(EVAL_LOGIT_SMOKE_CONFIG) \
>   --limit $(EVAL_SMOKE_LIMIT)

eval-logit-dry-run:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_eval.py \
>   --config $(EVAL_LOGIT_CONFIG) \
>   --dry-run

validate-eval-logit-inputs:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_eval.py --config $(EVAL_LOGIT_CONFIG) --validate-inputs$(if $(EVAL_ARGS), $(EVAL_ARGS))

export:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/export_model.py \
>   --config $(EXPORT_CONFIG)

export-dry-run:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/export_model.py \
>   --config $(EXPORT_CONFIG) \
>   --dry-run

export-push:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/export_model.py \
>   --config $(EXPORT_CONFIG) \
>   --push-to-hub

export-logit:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/export_model.py \
>   --config $(EXPORT_LOGIT_CONFIG)

export-logit-dry-run:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/export_model.py \
>   --config $(EXPORT_LOGIT_CONFIG) \
>   --dry-run

export-logit-push:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/export_model.py \
>   --config $(EXPORT_LOGIT_CONFIG) \
>   --push-to-hub
