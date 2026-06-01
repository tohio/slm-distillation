.PHONY: install test test-unit

install:
	python -m pip install -r requirements.txt

test:
	pytest

test-unit:
	pytest tests
