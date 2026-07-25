# ============================================================
# SLM Distillation
# ============================================================

.RECIPEPREFIX := >

PYTHON := python3
PYTHONPATH := .
DPO_CONFIG ?= configs/dpo.yaml
LOGIT_CONFIG ?= configs/logit_distill.yaml
EXPORT_CONFIG ?= configs/export.yaml
ARTIFACT_CONFIG ?= configs/artifacts.yaml

.PHONY: help install test test-unit verify-artifacts pack-artifacts unpack-artifacts push-artifacts pull-artifacts train-logit train-logit-dry-run train-dpo train-dpo-dry-run export export-dry-run

help:
> @echo ""
> @echo "SLM Distillation"
> @echo "================"
> @echo ""
> @echo "Setup:"
> @echo "  install                 Install Python dependencies"
> @echo ""
> @echo "Training:"
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

verify-artifacts:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/verify_artifacts.py \
>   --manifest artifacts/slm-125m-deepseek-distilled/manifest.json

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
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_dpo.py \
>   --config $(DPO_CONFIG)

train-dpo-dry-run:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_dpo.py \
>   --config $(DPO_CONFIG) \
>   --dry-run

export:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/export_model.py \
>   --config $(EXPORT_CONFIG)

export-dry-run:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/export_model.py \
>   --config $(EXPORT_CONFIG) \
>   --dry-run
