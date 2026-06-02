# ============================================================
# SLM Distillation — Makefile
# ============================================================

.RECIPEPREFIX := >

PYTHON := python3
PYTHONPATH := .
CONFIG ?= configs/response_distill.yaml
TEACHERS_CONFIG ?= configs/teachers.yaml
DPO_CONFIG ?= configs/dpo.yaml
LOGIT_CONFIG ?= configs/logit_distill.yaml
PREFERENCE_CONFIG ?= configs/preference.yaml
EXPORT_CONFIG ?= configs/export.yaml
ARTIFACT_CONFIG ?= configs/artifacts.yaml
LIMIT ?=
TARGET_TOKENS ?=
ESTIMATED_TOKENS_PER_RECORD ?= 256
ALLOW_REPEAT_PROMPTS ?=

.PHONY: help install test test-unit generate generate-dry-run validate dataset token-report preference artifact-handoff verify-artifacts pack-artifacts unpack-artifacts push-artifacts pull-artifacts train-logit train-logit-dry-run train-dpo train-dpo-dry-run export export-dry-run response-pipeline response-pipeline-dry-run clean-generated

help:
> @echo ""
> @echo "SLM Distillation"
> @echo "================"
> @echo ""
> @echo "Usage: make <target> [CONFIG=path] [TEACHERS_CONFIG=path] [LIMIT=N]"
> @echo ""
> @echo "Setup:"
> @echo "  install                    Install Python dependencies"
> @echo ""
> @echo "Pipeline:"
> @echo "  generate-dry-run            Generate local dry-run teacher responses"
> @echo "  generate                    Generate teacher responses through configured provider"
> @echo "  validate                    Validate raw teacher responses"
> @echo "  dataset                     Build response-distillation dataset"
> @echo "  response-pipeline-dry-run   Dry-run generate -> validate -> dataset"
> @echo "  response-pipeline           Generate -> validate -> dataset"
> @echo ""
> @echo "Tests:"
> @echo "  test                        Run full test suite"
> @echo "  test-unit                   Run unit tests"
> @echo ""
> @echo "Cleanup:"
> @echo "  clean-generated             Remove generated JSONL files and runs"
> @echo ""

install:
> $(PYTHON) -m pip install -r requirements.txt

test:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest

test-unit:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests

generate:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/generate_teacher_responses.py \
>   --config $(CONFIG) \
>   --teachers $(TEACHERS_CONFIG) \
>   $(if $(LIMIT),--limit $(LIMIT),) \
>   $(if $(TARGET_TOKENS),--target-tokens $(TARGET_TOKENS),) \
>   $(if $(ESTIMATED_TOKENS_PER_RECORD),--estimated-tokens-per-record $(ESTIMATED_TOKENS_PER_RECORD),) \
>   $(if $(ALLOW_REPEAT_PROMPTS),--allow-repeat-prompts,)

generate-dry-run:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/generate_teacher_responses.py \
>   --config $(CONFIG) \
>   --teachers $(TEACHERS_CONFIG) \
>   $(if $(LIMIT),--limit $(LIMIT),) \
>   $(if $(TARGET_TOKENS),--target-tokens $(TARGET_TOKENS),) \
>   $(if $(ESTIMATED_TOKENS_PER_RECORD),--estimated-tokens-per-record $(ESTIMATED_TOKENS_PER_RECORD),) \
>   $(if $(ALLOW_REPEAT_PROMPTS),--allow-repeat-prompts,) \
>   --dry-run

validate:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/validate_teacher_responses.py \
>   --config $(CONFIG)

dataset:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/build_dataset.py \
>   --config $(CONFIG)

token-report:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/report_token_counts.py

preference:
> PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/build_preference_dataset.py \
>   --config $(PREFERENCE_CONFIG)

artifact-handoff: pack-artifacts push-artifacts

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

response-pipeline: generate validate dataset

response-pipeline-dry-run: generate-dry-run validate dataset

clean-generated:
> rm -f data/raw_teacher/*.jsonl
> rm -f data/validated/*.jsonl
> rm -f data/rejected/*.jsonl
> rm -f data/distill/*.jsonl
> rm -f data/preference/*.jsonl
> rm -rf runs/