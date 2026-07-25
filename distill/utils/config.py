from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ResponseModelConfig:
    model_name_or_path: str
    tokenizer_name_or_path: str
    revision: str | None


@dataclass(frozen=True)
class ResponseDataConfig:
    dataset_id: str
    dataset_config_name: str | None
    dataset_split: str
    id_field: str | None
    prompt_field: str
    response_field: str
    metadata_field: str | None


@dataclass(frozen=True)
class ResponseFormattingConfig:
    mode: str
    system_prompt: str | None


@dataclass(frozen=True)
class ResponseTrainingConfig:
    max_length: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: float
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    lr_scheduler_type: str
    logging_steps: int
    save_steps: int
    save_total_limit: int
    bf16: bool
    gradient_checkpointing: bool
    dataloader_num_workers: int
    seed: int
    max_steps: int | None
    max_train_samples: int | None


@dataclass(frozen=True)
class ResponseOutputConfig:
    model_name: str
    run_dir: str
    checkpoint_dir: str
    final_checkpoint_dir: str


@dataclass(frozen=True)
class ResponseDistillConfig:
    model: ResponseModelConfig
    data: ResponseDataConfig
    formatting: ResponseFormattingConfig
    training: ResponseTrainingConfig
    output: ResponseOutputConfig


@dataclass(frozen=True)
class DpoSourceConfig:
    model_name: str
    checkpoint_path: str
    tokenizer_path: str


@dataclass(frozen=True)
class DpoDataConfig:
    dataset_id: str
    dataset_split: str


@dataclass(frozen=True)
class DpoTrainingConfig:
    method: str
    beta: float
    max_length: int
    max_prompt_length: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: int
    warmup_ratio: float
    bf16: bool
    seed: int


@dataclass(frozen=True)
class DpoOutputConfig:
    model_name: str
    run_dir: str
    checkpoint_dir: str
    final_checkpoint_dir: str


@dataclass(frozen=True)
class DpoConfig:
    source: DpoSourceConfig
    data: DpoDataConfig
    training: DpoTrainingConfig
    output: DpoOutputConfig


@dataclass(frozen=True)
class ExportModelConfig:
    model_name: str
    checkpoint_path: str
    tokenizer_path: str
    export_repo: str


@dataclass(frozen=True)
class ExportModelCardConfig:
    output_path: str
    source_checkpoint: str
    teacher_model: str
    teacher_provider: str
    distillation_type: str
    dpo_applied: bool
    response_dataset: str
    preference_dataset: str
    eval_results_path: str


@dataclass(frozen=True)
class ExportSettings:
    push_to_hub: bool
    include_tokenizer: bool
    private: bool


@dataclass(frozen=True)
class ExportConfig:
    model: ExportModelConfig
    model_card: ExportModelCardConfig
    export: ExportSettings


@dataclass(frozen=True)
class LogitTeacherConfig:
    provider: str
    model_name: str
    checkpoint_path: str
    tokenizer_path: str


@dataclass(frozen=True)
class LogitStudentConfig:
    model_name: str
    checkpoint_path: str
    tokenizer_path: str


@dataclass(frozen=True)
class LogitDistillationSettings:
    mode: str
    temperature: float
    alpha: float
    max_length: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: int
    bf16: bool
    seed: int


@dataclass(frozen=True)
class LogitCompatibilityConfig:
    require_same_tokenizer: bool


@dataclass(frozen=True)
class LogitHardwareConfig:
    single_gpu_required: bool
    allowed_gpu_classes: list[str]


@dataclass(frozen=True)
class LogitOutputConfig:
    run_dir: str
    checkpoint_dir: str


@dataclass(frozen=True)
class LogitDistillConfig:
    teacher: LogitTeacherConfig
    student: LogitStudentConfig
    distillation: LogitDistillationSettings
    compatibility: LogitCompatibilityConfig
    hardware: LogitHardwareConfig
    output: LogitOutputConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    return data


def _require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"config requires '{key}' mapping")
    return value


def _require_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"config requires non-empty '{key}'")
    return value


def _require_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int):
        raise ValueError(f"config requires integer '{key}'")
    return value


def _require_float_or_int(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if not isinstance(value, (float, int)):
        raise ValueError(f"config requires numeric '{key}'")
    return float(value)


def _require_bool(data: dict[str, Any], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"config requires boolean '{key}'")
    return value


def _optional_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"config optional field '{key}' must be a non-empty string")
    return value


def _optional_int(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"config optional field '{key}' must be an integer")
    return value


def load_response_distill_config(path: str | Path) -> ResponseDistillConfig:
    data = load_yaml(path)
    model = _require_mapping(data, "model")
    data_cfg = _require_mapping(data, "data")
    formatting = _require_mapping(data, "formatting")
    training = _require_mapping(data, "training")
    output = _require_mapping(data, "output")

    formatting_mode = _require_str(formatting, "mode")
    if formatting_mode not in {"chat", "plain"}:
        raise ValueError("response formatting.mode must be 'chat' or 'plain'")

    training_config = ResponseTrainingConfig(
        max_length=_require_int(training, "max_length"),
        per_device_train_batch_size=_require_int(
            training, "per_device_train_batch_size"
        ),
        gradient_accumulation_steps=_require_int(
            training, "gradient_accumulation_steps"
        ),
        learning_rate=_require_float_or_int(training, "learning_rate"),
        num_train_epochs=_require_float_or_int(training, "num_train_epochs"),
        warmup_ratio=_require_float_or_int(training, "warmup_ratio"),
        weight_decay=_require_float_or_int(training, "weight_decay"),
        max_grad_norm=_require_float_or_int(training, "max_grad_norm"),
        lr_scheduler_type=_require_str(training, "lr_scheduler_type"),
        logging_steps=_require_int(training, "logging_steps"),
        save_steps=_require_int(training, "save_steps"),
        save_total_limit=_require_int(training, "save_total_limit"),
        bf16=_require_bool(training, "bf16"),
        gradient_checkpointing=_require_bool(
            training, "gradient_checkpointing"
        ),
        dataloader_num_workers=_require_int(training, "dataloader_num_workers"),
        seed=_require_int(training, "seed"),
        max_steps=_optional_int(training, "max_steps"),
        max_train_samples=_optional_int(training, "max_train_samples"),
    )
    positive_ints = {
        "max_length": training_config.max_length,
        "per_device_train_batch_size": (
            training_config.per_device_train_batch_size
        ),
        "gradient_accumulation_steps": (
            training_config.gradient_accumulation_steps
        ),
        "logging_steps": training_config.logging_steps,
        "save_steps": training_config.save_steps,
        "save_total_limit": training_config.save_total_limit,
    }
    for field, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"response training.{field} must be positive")
    if training_config.dataloader_num_workers < 0:
        raise ValueError(
            "response training.dataloader_num_workers must be non-negative"
        )
    if training_config.learning_rate <= 0:
        raise ValueError("response training.learning_rate must be positive")
    if training_config.num_train_epochs <= 0:
        raise ValueError("response training.num_train_epochs must be positive")
    if not 0 <= training_config.warmup_ratio < 1:
        raise ValueError(
            "response training.warmup_ratio must be greater than or equal to "
            "zero and less than one"
        )
    if training_config.weight_decay < 0:
        raise ValueError(
            "response training.weight_decay must be greater than or equal to zero"
        )
    if training_config.max_grad_norm <= 0:
        raise ValueError("response training.max_grad_norm must be positive")
    if (
        training_config.max_steps is not None
        and training_config.max_steps <= 0
    ):
        raise ValueError("response training.max_steps must be positive when set")
    if (
        training_config.max_train_samples is not None
        and training_config.max_train_samples <= 0
    ):
        raise ValueError(
            "response training.max_train_samples must be positive when set"
        )

    return ResponseDistillConfig(
        model=ResponseModelConfig(
            model_name_or_path=_require_str(model, "model_name_or_path"),
            tokenizer_name_or_path=_require_str(model, "tokenizer_name_or_path"),
            revision=_optional_str(model, "revision"),
        ),
        data=ResponseDataConfig(
            dataset_id=_require_str(data_cfg, "dataset_id"),
            dataset_config_name=_optional_str(data_cfg, "dataset_config_name"),
            dataset_split=_require_str(data_cfg, "dataset_split"),
            id_field=_optional_str(data_cfg, "id_field"),
            prompt_field=_require_str(data_cfg, "prompt_field"),
            response_field=_require_str(data_cfg, "response_field"),
            metadata_field=_optional_str(data_cfg, "metadata_field"),
        ),
        formatting=ResponseFormattingConfig(
            mode=formatting_mode,
            system_prompt=_optional_str(formatting, "system_prompt"),
        ),
        training=training_config,
        output=ResponseOutputConfig(
            model_name=_require_str(output, "model_name"),
            run_dir=_require_str(output, "run_dir"),
            checkpoint_dir=_require_str(output, "checkpoint_dir"),
            final_checkpoint_dir=_require_str(output, "final_checkpoint_dir"),
        ),
    )


def load_dpo_config(path: str | Path) -> DpoConfig:
    data = load_yaml(path)
    source = _require_mapping(data, "source")
    data_cfg = _require_mapping(data, "data")
    training = _require_mapping(data, "training")
    output = _require_mapping(data, "output")

    method = _require_str(training, "method")
    if method != "dpo":
        raise ValueError("DPO config requires training.method='dpo'")

    return DpoConfig(
        source=DpoSourceConfig(
            model_name=_require_str(source, "model_name"),
            checkpoint_path=_require_str(source, "checkpoint_path"),
            tokenizer_path=_require_str(source, "tokenizer_path"),
        ),
        data=DpoDataConfig(
            dataset_id=_require_str(data_cfg, "dataset_id"),
            dataset_split=_require_str(data_cfg, "dataset_split"),
        ),
        training=DpoTrainingConfig(
            method=method,
            beta=_require_float_or_int(training, "beta"),
            max_length=_require_int(training, "max_length"),
            max_prompt_length=_require_int(training, "max_prompt_length"),
            per_device_train_batch_size=_require_int(
                training, "per_device_train_batch_size"
            ),
            gradient_accumulation_steps=_require_int(
                training, "gradient_accumulation_steps"
            ),
            learning_rate=_require_float_or_int(training, "learning_rate"),
            num_train_epochs=_require_int(training, "num_train_epochs"),
            warmup_ratio=_require_float_or_int(training, "warmup_ratio"),
            bf16=_require_bool(training, "bf16"),
            seed=_require_int(training, "seed"),
        ),
        output=DpoOutputConfig(
            model_name=_require_str(output, "model_name"),
            run_dir=_require_str(output, "run_dir"),
            checkpoint_dir=_require_str(output, "checkpoint_dir"),
            final_checkpoint_dir=_require_str(output, "final_checkpoint_dir"),
        ),
    )


def load_export_config(path: str | Path) -> ExportConfig:
    data = load_yaml(path)
    model = _require_mapping(data, "model")
    model_card = _require_mapping(data, "model_card")
    export = _require_mapping(data, "export")

    return ExportConfig(
        model=ExportModelConfig(
            model_name=_require_str(model, "model_name"),
            checkpoint_path=_require_str(model, "checkpoint_path"),
            tokenizer_path=_require_str(model, "tokenizer_path"),
            export_repo=_require_str(model, "export_repo"),
        ),
        model_card=ExportModelCardConfig(
            output_path=_require_str(model_card, "output_path"),
            source_checkpoint=_require_str(model_card, "source_checkpoint"),
            teacher_model=_require_str(model_card, "teacher_model"),
            teacher_provider=_require_str(model_card, "teacher_provider"),
            distillation_type=_require_str(model_card, "distillation_type"),
            dpo_applied=_require_bool(model_card, "dpo_applied"),
            response_dataset=_require_str(model_card, "response_dataset"),
            preference_dataset=_require_str(model_card, "preference_dataset"),
            eval_results_path=_require_str(model_card, "eval_results_path"),
        ),
        export=ExportSettings(
            push_to_hub=_require_bool(export, "push_to_hub"),
            include_tokenizer=_require_bool(export, "include_tokenizer"),
            private=_require_bool(export, "private"),
        ),
    )


def load_logit_distill_config(path: str | Path) -> LogitDistillConfig:
    data = load_yaml(path)
    teacher = _require_mapping(data, "teacher")
    student = _require_mapping(data, "student")
    distillation = _require_mapping(data, "distillation")
    compatibility = _require_mapping(data, "compatibility")
    hardware = _require_mapping(data, "hardware")
    output = _require_mapping(data, "output")

    provider = _require_str(teacher, "provider")
    if provider != "local":
        raise ValueError("Logit distillation requires local provider")

    mode = _require_str(distillation, "mode")
    if mode != "logit":
        raise ValueError("logit_distill config requires distillation.mode='logit'")

    allowed_gpu_classes = hardware.get("allowed_gpu_classes")
    if not isinstance(allowed_gpu_classes, list) or not allowed_gpu_classes:
        raise ValueError("hardware.allowed_gpu_classes must be a non-empty list")

    normalized_gpu_classes: list[str] = []
    for index, value in enumerate(allowed_gpu_classes):
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"hardware.allowed_gpu_classes item {index} must be a non-empty string"
            )
        normalized_gpu_classes.append(value)

    return LogitDistillConfig(
        teacher=LogitTeacherConfig(
            provider=provider,
            model_name=_require_str(teacher, "model_name"),
            checkpoint_path=_require_str(teacher, "checkpoint_path"),
            tokenizer_path=_require_str(teacher, "tokenizer_path"),
        ),
        student=LogitStudentConfig(
            model_name=_require_str(student, "model_name"),
            checkpoint_path=_require_str(student, "checkpoint_path"),
            tokenizer_path=_require_str(student, "tokenizer_path"),
        ),
        distillation=LogitDistillationSettings(
            mode=mode,
            temperature=_require_float_or_int(distillation, "temperature"),
            alpha=_require_float_or_int(distillation, "alpha"),
            max_length=_require_int(distillation, "max_length"),
            per_device_train_batch_size=_require_int(
                distillation, "per_device_train_batch_size"
            ),
            gradient_accumulation_steps=_require_int(
                distillation, "gradient_accumulation_steps"
            ),
            learning_rate=_require_float_or_int(distillation, "learning_rate"),
            num_train_epochs=_require_int(distillation, "num_train_epochs"),
            bf16=_require_bool(distillation, "bf16"),
            seed=_require_int(distillation, "seed"),
        ),
        compatibility=LogitCompatibilityConfig(
            require_same_tokenizer=_require_bool(
                compatibility, "require_same_tokenizer"
            ),
        ),
        hardware=LogitHardwareConfig(
            single_gpu_required=_require_bool(hardware, "single_gpu_required"),
            allowed_gpu_classes=normalized_gpu_classes,
        ),
        output=LogitOutputConfig(
            run_dir=_require_str(output, "run_dir"),
            checkpoint_dir=_require_str(output, "checkpoint_dir"),
        ),
    )
