from __future__ import annotations

from dataclasses import dataclass

from distill.utils.config import LogitDistillConfig, load_logit_distill_config
from distill.utils.tokenizer_compat import (
    TokenizerCompatibilityResult,
    assert_tokenizer_compatible,
)


@dataclass(frozen=True)
class LogitDistillationPlan:
    teacher_checkpoint_path: str
    teacher_tokenizer_path: str
    student_checkpoint_path: str
    student_tokenizer_path: str
    output_dir: str
    require_same_tokenizer: bool
    tokenizer_compatibility: TokenizerCompatibilityResult | None


def build_logit_distillation_plan(config: LogitDistillConfig) -> LogitDistillationPlan:
    compatibility = None

    if config.compatibility.require_same_tokenizer:
        compatibility = assert_tokenizer_compatible(
            teacher_tokenizer_path=config.teacher.tokenizer_path,
            student_tokenizer_path=config.student.tokenizer_path,
        )

    return LogitDistillationPlan(
        teacher_checkpoint_path=config.teacher.checkpoint_path,
        teacher_tokenizer_path=config.teacher.tokenizer_path,
        student_checkpoint_path=config.student.checkpoint_path,
        student_tokenizer_path=config.student.tokenizer_path,
        output_dir=config.output.checkpoint_dir,
        require_same_tokenizer=config.compatibility.require_same_tokenizer,
        tokenizer_compatibility=compatibility,
    )


def load_logit_distillation_plan(config_path: str) -> LogitDistillationPlan:
    return build_logit_distillation_plan(load_logit_distill_config(config_path))


def train_logit_distill(config_path: str) -> None:
    load_logit_distillation_plan(config_path)
    raise NotImplementedError(
        "Local logit distillation is configured with tokenizer compatibility checks. "
        "The trainer implementation will be added later."
    )
