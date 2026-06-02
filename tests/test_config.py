from pathlib import Path

import pytest

from distill.utils.config import get_default_teacher, load_teachers_config, load_yaml


def test_load_yaml_reads_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("name: test\n", encoding="utf-8")

    assert load_yaml(path) == {"name": "test"}


def test_load_yaml_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "missing.yaml")


def test_load_teachers_config_reads_default_teacher() -> None:
    config = load_teachers_config("configs/teachers.yaml")
    teacher = get_default_teacher(config)

    assert config.default_teacher == "deepseek_v4_flash"
    assert teacher.provider == "openrouter"
    assert teacher.model == "deepseek/deepseek-v4-flash"
    assert teacher.mode == "response"
    assert teacher.distillation_allowed is True


def test_load_teachers_config_rejects_missing_default(tmp_path: Path) -> None:
    path = tmp_path / "teachers.yaml"
    path.write_text(
        """
default_teacher: missing

teachers:
  valid:
    provider: openrouter
    model: example/model
    mode: response
    purpose: test
    distillation_allowed: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="default_teacher"):
        load_teachers_config(path)


def test_load_response_distill_config_reads_default_file() -> None:
    from distill.utils.config import load_response_distill_config

    config = load_response_distill_config("configs/response_distill.yaml")

    assert config.teacher_name == "deepseek_v4_flash"
    assert config.student.name == "slm-student"
    assert config.distillation.mode == "response"
    assert config.distillation.target_tokens == 50000000
    assert config.distillation.max_retries == 2
    assert config.distillation.retry_delay_seconds == 2.0
    assert config.distillation.continue_on_error is True
    assert config.data.raw_teacher_path == "data/raw_teacher/deepseek_v4_flash.jsonl"
    assert config.output.run_dir == "runs/response_distill"


def test_load_response_distill_config_rejects_wrong_mode(tmp_path: Path) -> None:
    from distill.utils.config import load_response_distill_config

    path = tmp_path / "response_distill.yaml"
    path.write_text(
        """
teacher:
  name: deepseek_v4_flash

student:
  name: slm-student
  checkpoint_path: checkpoint
  tokenizer_path: tokenizer

distillation:
  mode: logit
  target_tokens: 100
  max_output_tokens: 32
  temperature: 0.2
  top_p: 0.9
  max_retries: 2
  retry_delay_seconds: 0.0
  continue_on_error: true

data:
  prompts_path: prompts.jsonl
  raw_teacher_path: raw.jsonl
  validated_path: valid.jsonl
  rejected_path: rejected.jsonl
  distill_dataset_path: distill.jsonl

validation:
  require_non_empty_output: true
  reject_refusals_when_not_expected: true
  reject_code_fences_for_function_body_tasks: true
  max_retries: 2

output:
  run_dir: runs/test
  checkpoint_dir: runs/test/checkpoints
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mode='response'"):
        load_response_distill_config(path)
