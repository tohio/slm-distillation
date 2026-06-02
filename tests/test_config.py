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

    config = load_response_distill_config("configs/response_distill_openrouter.yaml")

    assert config.teacher_name == "deepseek_v4_flash"
    assert config.student.name == "slm-student"
    assert config.distillation.mode == "response"
    assert config.distillation.target_tokens == 50000000
    assert config.distillation.max_retries == 2
    assert config.distillation.retry_delay_seconds == 2.0
    assert config.distillation.continue_on_error is True
    assert config.data.prompts_path is None
    assert config.data.prompts_paths == ["data/prompts/built_prompts.jsonl"]
    assert config.data.raw_teacher_path == "data/raw_teacher/deepseek_v4_flash.jsonl"
    assert config.output.model_name == "slm-125m-deepseek-distilled"
    assert config.output.source_model_name == "slm-125m"
    assert config.output.teacher_family == "deepseek"
    assert config.output.run_dir == "runs/slm-125m-deepseek-distilled"
    assert config.output.final_checkpoint_dir == (
        "runs/slm-125m-deepseek-distilled/dpo/checkpoints/final"
    )
    assert config.output.export_repo == "tohio/slm-125m-deepseek-distilled"
    assert config.model_card.source_checkpoint == "tohio/slm-125m-instruct"
    assert config.model_card.teacher_model == "deepseek/deepseek-v4-flash"
    assert config.model_card.teacher_provider == "openrouter"
    assert config.model_card.distillation_type == "response"
    assert config.model_card.dpo_applied is True


def test_load_response_distill_config_rejects_wrong_mode(tmp_path: Path) -> None:
    from distill.utils.config import load_response_distill_config

    path = tmp_path / "response_distill_openrouter.yaml"
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
  prompts_paths:
    - prompts.jsonl
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
  model_name: slm-test-teacher-distilled
  source_model_name: slm-test
  teacher_family: teacher
  run_dir: runs/slm-test-teacher-distilled
  checkpoint_dir: runs/slm-test-teacher-distilled/response_distill/checkpoints
  final_checkpoint_dir: runs/slm-test-teacher-distilled/dpo/checkpoints/final
  export_repo: tohio/slm-test-teacher-distilled

model_card:
  source_checkpoint: tohio/slm-test-instruct
  teacher_model: example/model
  teacher_provider: openrouter
  distillation_type: response
  dpo_applied: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mode='response'"):
        load_response_distill_config(path)


def test_load_response_distill_config_accepts_prompts_paths_only(tmp_path: Path) -> None:
    from distill.utils.config import load_response_distill_config

    path = tmp_path / "response_distill_openrouter.yaml"
    path.write_text(
        """
teacher:
  name: deepseek_v4_flash

student:
  name: slm-student
  checkpoint_path: checkpoint
  tokenizer_path: tokenizer

distillation:
  mode: response
  target_tokens: 100
  max_output_tokens: 32
  temperature: 0.2
  top_p: 0.9
  max_retries: 2
  retry_delay_seconds: 0.0
  continue_on_error: true

data:
  prompts_paths:
    - prompts-a.jsonl
    - prompts-b.jsonl
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
  model_name: slm-test-teacher-distilled
  source_model_name: slm-test
  teacher_family: teacher
  run_dir: runs/slm-test-teacher-distilled
  checkpoint_dir: runs/slm-test-teacher-distilled/response_distill/checkpoints
  final_checkpoint_dir: runs/slm-test-teacher-distilled/dpo/checkpoints/final
  export_repo: tohio/slm-test-teacher-distilled

model_card:
  source_checkpoint: tohio/slm-test-instruct
  teacher_model: example/model
  teacher_provider: openrouter
  distillation_type: response
  dpo_applied: true
""",
        encoding="utf-8",
    )

    config = load_response_distill_config(path)

    assert config.data.prompts_path is None
    assert config.data.prompts_paths == ["prompts-a.jsonl", "prompts-b.jsonl"]


def test_load_response_distill_config_rejects_conflicting_prompt_fields(
    tmp_path: Path,
) -> None:
    from distill.utils.config import load_response_distill_config

    path = tmp_path / "response_distill_openrouter.yaml"
    path.write_text(
        """
teacher:
  name: deepseek_v4_flash

student:
  name: slm-student
  checkpoint_path: checkpoint
  tokenizer_path: tokenizer

distillation:
  mode: response
  target_tokens: 100
  max_output_tokens: 32
  temperature: 0.2
  top_p: 0.9
  max_retries: 2
  retry_delay_seconds: 0.0
  continue_on_error: true

data:
  prompts_path: prompts-a.jsonl
  prompts_paths:
    - prompts-b.jsonl
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
  model_name: slm-test-teacher-distilled
  source_model_name: slm-test
  teacher_family: teacher
  run_dir: runs/slm-test-teacher-distilled
  checkpoint_dir: runs/slm-test-teacher-distilled/response_distill/checkpoints
  final_checkpoint_dir: runs/slm-test-teacher-distilled/dpo/checkpoints/final
  export_repo: tohio/slm-test-teacher-distilled

model_card:
  source_checkpoint: tohio/slm-test-instruct
  teacher_model: example/model
  teacher_provider: openrouter
  distillation_type: response
  dpo_applied: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="prompts_path"):
        load_response_distill_config(path)

def test_load_response_distill_config_reads_groq_model_naming() -> None:
    from distill.utils.config import load_response_distill_config

    config = load_response_distill_config("configs/response_distill_groq.yaml")

    assert config.teacher_name == "groq_llama_3_3_70b_versatile"
    assert config.output.model_name == "slm-125m-groq-distilled"
    assert config.output.source_model_name == "slm-125m"
    assert config.output.teacher_family == "groq"
    assert config.output.export_repo == "tohio/slm-125m-groq-distilled"
    assert config.model_card.teacher_model == "llama-3.3-70b-versatile"
    assert config.model_card.teacher_provider == "groq"
    assert config.model_card.distillation_type == "response"
    assert config.model_card.dpo_applied is True
