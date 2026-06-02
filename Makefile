.PHONY: install test test-unit generate generate-dry-run validate dataset response-pipeline response-pipeline-dry-run clean-generated

CONFIG ?= configs/response_distill.yaml
TEACHERS_CONFIG ?= configs/teachers.yaml
LIMIT ?=

install:
	python3 -m pip install -r requirements.txt

test:
	python3 -m pytest

test-unit:
	python3 -m pytest tests

generate:
	python3 scripts/generate_teacher_responses.py \
		--config $(CONFIG) \
		--teachers $(TEACHERS_CONFIG) \
		$(if $(LIMIT),--limit $(LIMIT),)

generate-dry-run:
	python3 scripts/generate_teacher_responses.py \
		--config $(CONFIG) \
		--teachers $(TEACHERS_CONFIG) \
		$(if $(LIMIT),--limit $(LIMIT),) \
		--dry-run

validate:
	python3 scripts/validate_teacher_responses.py \
		--config $(CONFIG)

dataset:
	python3 scripts/build_dataset.py \
		--config $(CONFIG)

response-pipeline: generate validate dataset

response-pipeline-dry-run: generate-dry-run validate dataset

clean-generated:
	rm -f data/raw_teacher/*.jsonl
	rm -f data/validated/*.jsonl
	rm -f data/rejected/*.jsonl
	rm -f data/distill/*.jsonl
	rm -rf runs/
