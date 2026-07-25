from pathlib import Path

import pytest

from distill.training.train_logit_distill import (
    build_logit_distillation_plan,
)
from distill.utils.config import load_logit_distill_config


def test_load_logit_distill_config_reads_default_file() -> None:
    config = load_logit_distill_config("configs/logit_distill.yaml")

    assert config.teacher.provider == "local"
    assert (
        config.teacher.checkpoint_path
        == "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    )
    assert (
        config.student.checkpoint_path
        == "HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    assert config.data.dataset_id == "tohio/slm-synthetic-distillation-sft"
    assert config.formatting.mode == "chat"
    assert config.distillation.mode == "logit"
    assert config.distillation.temperature == 2.0
    assert config.distillation.alpha == 0.5
    assert config.compatibility.require_same_tokenizer is True
    assert config.hardware.single_gpu_required is True
    assert config.hardware.allowed_gpu_classes == [
        "b300",
        "b200",
        "h200",
        "a100",
    ]
    assert config.output.final_checkpoint_dir.endswith(
        "logit_distill/checkpoints/final"
    )


def test_build_logit_distillation_plan_is_side_effect_free() -> None:
    config = load_logit_distill_config("configs/logit_distill.yaml")

    plan = build_logit_distillation_plan(config)

    assert plan.teacher_checkpoint_path == config.teacher.checkpoint_path
    assert plan.student_checkpoint_path == config.student.checkpoint_path
    assert plan.dataset_id == config.data.dataset_id
    assert plan.temperature == 2.0
    assert plan.alpha == 0.5
    assert plan.require_same_tokenizer is True


def test_load_logit_distill_config_rejects_non_local_logit_provider(
    tmp_path: Path,
) -> None:
    config_text = Path("configs/logit_distill.yaml").read_text(
        encoding="utf-8"
    )
    config_path = tmp_path / "logit_distill.yaml"
    config_path.write_text(
        config_text.replace("provider: local", "provider: openrouter"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="local provider"):
        load_logit_distill_config(config_path)


def test_load_logit_distill_config_rejects_invalid_alpha(
    tmp_path: Path,
) -> None:
    config_text = Path("configs/logit_distill.yaml").read_text(
        encoding="utf-8"
    )
    config_path = tmp_path / "logit_distill.yaml"
    config_path.write_text(
        config_text.replace("alpha: 0.5", "alpha: 1.5"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="alpha"):
        load_logit_distill_config(config_path)
