import json
from pathlib import Path

import pytest

from distill.preference.build_preference_dataset import build_preference_dataset
from distill.preference.records import load_preference_records
from distill.utils.config import load_preference_config


def test_load_preference_config_reads_default_file() -> None:
    config = load_preference_config("configs/preference.yaml")

    assert config.source.validated_teacher_path == "data/validated/deepseek_v4_flash.jsonl"
    assert config.source.rejected_response_path is None
    assert config.data.output_path == "data/preference/dpo_pairs.jsonl"
    assert config.preference.default_rejected_response == "I do not know."
    assert config.preference.reject_identical_pairs is True


def test_build_preference_dataset_from_validated_teacher_records(tmp_path: Path) -> None:
    validated_path = tmp_path / "validated.jsonl"
    preference_path = tmp_path / "preference.jsonl"
    rejected_path = tmp_path / "rejected.jsonl"
    config_path = tmp_path / "preference.yaml"

    validated_path.write_text(
        json.dumps(
            {
                "prompt_id": "p1",
                "category": "instruction",
                "prompt": "Say hello.",
                "teacher": "teacher/model",
                "provider": "openrouter",
                "response": "hello",
                "input_tokens": 3,
                "output_tokens": 1,
                "metadata": {"source": "test"},
                "error": None,
                "validation": {"accepted": True, "reason": "accepted"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    config_path.write_text(
        f'''
source:
  validated_teacher_path: {validated_path}
  rejected_response_path: null

data:
  output_path: {preference_path}
  rejected_path: {rejected_path}

preference:
  default_rejected_response: "I do not know."
  require_non_empty_chosen: true
  require_non_empty_rejected: true
  reject_identical_pairs: true
''',
        encoding="utf-8",
    )

    result = build_preference_dataset(str(config_path))
    records = load_preference_records(preference_path)

    assert result.accepted == 1
    assert result.rejected == 0
    assert records[0].prompt == "Say hello."
    assert records[0].chosen == "hello"
    assert records[0].rejected == "I do not know."
    assert records[0].metadata["prompt_id"] == "p1"


def test_build_preference_dataset_rejects_identical_pairs(tmp_path: Path) -> None:
    validated_path = tmp_path / "validated.jsonl"
    preference_path = tmp_path / "preference.jsonl"
    rejected_path = tmp_path / "rejected.jsonl"
    config_path = tmp_path / "preference.yaml"

    validated_path.write_text(
        json.dumps(
            {
                "prompt_id": "p1",
                "category": "instruction",
                "prompt": "Say hello.",
                "teacher": "teacher/model",
                "provider": "openrouter",
                "response": "same",
                "input_tokens": None,
                "output_tokens": None,
                "metadata": {},
                "error": None,
                "validation": {"accepted": True, "reason": "accepted"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    config_path.write_text(
        f'''
source:
  validated_teacher_path: {validated_path}
  rejected_response_path: null

data:
  output_path: {preference_path}
  rejected_path: {rejected_path}

preference:
  default_rejected_response: "same"
  require_non_empty_chosen: true
  require_non_empty_rejected: true
  reject_identical_pairs: true
''',
        encoding="utf-8",
    )

    result = build_preference_dataset(str(config_path))
    rejected = [json.loads(line) for line in rejected_path.read_text().splitlines()]

    assert result.accepted == 0
    assert result.rejected == 1
    assert rejected[0]["reason"] == "identical_pair"


def test_load_preference_records_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_preference_records(tmp_path / "missing.jsonl")
