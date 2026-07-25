# ============================================================
# SLM Distillation
# ============================================================

.RECIPEPREFIX := >

PYTHON := python3
PYTHONPATH := .
RESPONSE_CONFIG ?= configs/response_distill.yaml
RESPONSE_DATA_LIMIT ?= 100
RESPONSE_MAX_STEPS ?=
RESPONSE_MAX_TRAIN_SAMPLES ?=
RESPONSE_RESUME_FROM_CHECKPOINT ?=
RESPONSE_LAUNCH ?= $(PYTHON)
DPO_CONFIG ?= configs/dpo.yaml
DPO_DATA_LIMIT ?= 100
DPO_MAX_STEPS ?=
DPO_MAX_TRAIN_SAMPLES ?=
DPO_RESUME_FROM_CHECKPOINT ?=
DPO_LAUNCH ?= $(PYTHON)
LOGIT_CONFIG ?= configs/logit_distill.yaml
EXPORT_CONFIG ?= configs/export.yaml
ARTIFACT_CONFIG ?= configs/artifacts.yaml

.PHONY: help install test test-unit verify-artifacts pack-artifacts unpack-artifacts push-artifacts pull-artifacts validate-response-inputs validate-dpo-inputs train-response train-response-dry-run train-logit train-logit-dry-run train-dpo train-dpo-dry-run export export-dry-run

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
> @echo "  train-response-dry-run  Print the resolved response plan"
> @echo "  validate-dpo-inputs     Validate the DPO model and dataset"
> @echo "  train-dpo               Train the DPO stage"
> @echo "  train-dpo-dry-run       Print the resolved DPO plan"
> @echo "  train-logit             Train with local teacher logits"
> @echo "  train-logit-dry-run     Print the resolved logit plan"
> @echo ""
> @echo "Export:"
> @echo "  export                  Generate the model card and export"
> @echo "  export-dry-run          Print the resolved export plan"
> @echo ""
> @echo "Artifacts:"
> @echo "  verify-artifacts        Verify an artifact manifest"
> @echo "  pack-artifacts          Build a local artifact bundle"
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
> PYTHONPATH=$(PYTHONPATH) $(RESPONSE_LAUNCH) scripts/train_response_distill.py \
>   --config $(RESPONSE_CONFIG) \
>   $(if $(RESPONSE_MAX_STEPS),--max-steps $(RESPONSE_MAX_STEPS)) \
>   $(if $(RESPONSE_MAX_TRAIN_SAMPLES),--max-train-samples $(RESPONSE_MAX_TRAIN_SAMPLES)) \
>   $(if $(RESPONSE_RESUME_FROM_CHECKPOINT),--resume-from-checkpoint $(RESPONSE_RESUME_FROM_CHECKPOINT))

verify-artifacts:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/verify_artifacts.py \
>   --manifest artifacts/smollm2-135m-deepseek-distilled/manifest.json

pack-artifacts:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/pack_artifacts.py \
>   --config $(ARTIFACT_CONFIG)

unpack-artifacts:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/unpack_artifacts.py $(ARTIFACT)

push-artifacts:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/push_artifacts.py \
>   --config $(ARTIFACT_CONFIG)

pull-artifacts:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/pull_artifacts.py \
>   --config $(ARTIFACT_CONFIG)

train-logit:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_logit_distill.py \
>   --config $(LOGIT_CONFIG)

train-logit-dry-run:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_logit_distill.py \
>   --config $(LOGIT_CONFIG) \
>   --dry-run

train-dpo:
> PYTHONPATH=$(PYTHONPATH) $(DPO_LAUNCH) scripts/train_dpo.py \
>   --config $(DPO_CONFIG) \
>   $(if $(DPO_MAX_STEPS),--max-steps $(DPO_MAX_STEPS)) \
>   $(if $(DPO_MAX_TRAIN_SAMPLES),--max-train-samples $(DPO_MAX_TRAIN_SAMPLES)) \
>   $(if $(DPO_RESUME_FROM_CHECKPOINT),--resume-from-checkpoint $(DPO_RESUME_FROM_CHECKPOINT))

train-dpo-dry-run:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_dpo.py \
>   --config $(DPO_CONFIG) \
>   --dry-run

validate-dpo-inputs:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_dpo.py \
>   --config $(DPO_CONFIG) \
>   --validate-inputs \
>   --limit $(DPO_DATA_LIMIT)

export:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/export_model.py \
>   --config $(EXPORT_CONFIG)

export-dry-run:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/export_model.py \
>   --config $(EXPORT_CONFIG) \
>   --dry-run
