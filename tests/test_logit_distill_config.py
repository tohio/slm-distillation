import json
from pathlib import Path

import pytest

from distill.training.train_logit_distill import build_logit_distillation_plan
from distill.utils.config import load_logit_distill_config


def write_tokenizer(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": {
            "vocab": {
                "<pad>": 0,
                "hello": 1,
            }
        },
        "added_tokens": [],
    }
    (path / "tokenizer.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_logit_distill_config_reads_default_file() -> None:
    config = load_logit_distill_config("configs/logit_distill.yaml")

    assert config.teacher.provider == "local"
    assert config.distillation.mode == "logit"
    assert config.compatibility.require_same_tokenizer is True
    assert config.hardware.single_gpu_required is True
    assert config.hardware.allowed_gpu_classes == ["b300", "b200", "h200", "a100"]


def test_build_logit_distillation_plan_checks_tokenizer_compatibility(
    tmp_path: Path,
) -> None:
    teacher_tokenizer = tmp_path / "teacher_tokenizer"
    student_tokenizer = tmp_path / "student_tokenizer"
    write_tokenizer(teacher_tokenizer)
    write_tokenizer(student_tokenizer)

    config_path = tmp_path / "logit_distill.yaml"
    config_path.write_text(
        f'''
teacher:
  provider: local
  model_name: teacher
  checkpoint_path: teacher-checkpoint
  tokenizer_path: {teacher_tokenizer}

student:
  model_name: student
  checkpoint_path: student-checkpoint
  tokenizer_path: {student_tokenizer}

distillation:
  mode: logit
  temperature: 2.0
  alpha: 0.5
  max_length: 1024
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 0.000005
  num_train_epochs: 1
  bf16: true
  seed: 42

compatibility:
  require_same_tokenizer: true

hardware:
  single_gpu_required: true
  allowed_gpu_classes:
    - a100

output:
  run_dir: runs/test
  checkpoint_dir: runs/test/checkpoints
''',
        encoding="utf-8",
    )

    config = load_logit_distill_config(config_path)
    plan = build_logit_distillation_plan(config)

    assert plan.require_same_tokenizer is True
    assert plan.tokenizer_compatibility is not None
    assert plan.tokenizer_compatibility.compatible is True


def test_load_logit_distill_config_rejects_non_local_logit_provider(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "logit_distill.yaml"
    config_path.write_text(
        '''
teacher:
  provider: openrouter
  model_name: teacher
  checkpoint_path: teacher-checkpoint
  tokenizer_path: teacher-tokenizer

student:
  model_name: student
  checkpoint_path: student-checkpoint
  tokenizer_path: student-tokenizer

distillation:
  mode: logit
  temperature: 2.0
  alpha: 0.5
  max_length: 1024
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 0.000005
  num_train_epochs: 1
  bf16: true
  seed: 42

compatibility:
  require_same_tokenizer: true

hardware:
  single_gpu_required: true
  allowed_gpu_classes:
    - a100

output:
  run_dir: runs/test
  checkpoint_dir: runs/test/checkpoints
''',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="local provider"):
        load_logit_distill_config(config_path)
