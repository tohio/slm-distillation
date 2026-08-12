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
    revision: str | None


@dataclass(frozen=True)
class DpoDataConfig:
    dataset_id: str
    dataset_config_name: str | None
    dataset_split: str
    id_field: str | None
    prompt_field: str
    chosen_field: str
    rejected_field: str
    metadata_field: str | None


@dataclass(frozen=True)
class DpoTrainingConfig:
    method: str
    beta: float
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
    dataset_num_proc: int
    loss_type: list[str]
    loss_weights: list[float]
    truncation_mode: str
    precompute_ref_log_probs: bool
    seed: int
    max_steps: int | None
    max_train_samples: int | None


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
    revision: str | None


@dataclass(frozen=True)
class LogitStudentConfig:
    model_name: str
    checkpoint_path: str
    tokenizer_path: str
    revision: str | None


@dataclass(frozen=True)
class LogitDataConfig:
    dataset_id: str
    dataset_config_name: str | None
    dataset_split: str
    id_field: str | None
    prompt_field: str
    response_field: str
    metadata_field: str | None


@dataclass(frozen=True)
class LogitDistillationSettings:
    mode: str
    temperature: float
    alpha: float
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
class LogitCompatibilityConfig:
    require_same_tokenizer: bool


@dataclass(frozen=True)
class LogitHardwareConfig:
    single_gpu_required: bool
    allowed_gpu_classes: list[str]


@dataclass(frozen=True)
class LogitOutputConfig:
    model_name: str
    run_dir: str
    checkpoint_dir: str
    final_checkpoint_dir: str


@dataclass(frozen=True)
class LogitDistillConfig:
    teacher: LogitTeacherConfig
    student: LogitStudentConfig
    data: LogitDataConfig
    formatting: ResponseFormattingConfig
    distillation: LogitDistillationSettings
    compatibility: LogitCompatibilityConfig
    hardware: LogitHardwareConfig
    output: LogitOutputConfig


@dataclass(frozen=True)
class EvalModelConfig:
    name: str
    model_name_or_path: str
    tokenizer_name_or_path: str
    revision: str | None


@dataclass(frozen=True)
class EvalDataConfig:
    dataset_id: str
    dataset_config_name: str | None
    dataset_split: str
    id_field: str | None
    prompt_field: str
    reference_field: str


@dataclass(frozen=True)
class EvalPreferenceDataConfig:
    dataset_id: str
    dataset_config_name: str | None
    dataset_split: str
    id_field: str | None
    prompt_field: str
    chosen_field: str
    rejected_field: str


@dataclass(frozen=True)
class EvalGenerationConfig:
    max_input_length: int
    max_new_tokens: int
    per_device_batch_size: int
    do_sample: bool
    temperature: float
    seed: int
    limit: int


@dataclass(frozen=True)
class EvalOutputConfig:
    results_path: str
    predictions_dir: str


@dataclass(frozen=True)
class EvalConfig:
    models: list[EvalModelConfig]
    data: EvalDataConfig
    preference_data: EvalPreferenceDataConfig
    formatting: ResponseFormattingConfig
    generation: EvalGenerationConfig
    output: EvalOutputConfig


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


def _require_str_or_str_list(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key)
    if isinstance(value, str) and value:
        return [value]
    if (
        isinstance(value, list)
        and value
        and all(isinstance(item, str) and item for item in value)
    ):
        return list(value)
    raise ValueError(f"config requires non-empty string or string list '{key}'")


def _optional_numeric_list(
    data: dict[str, Any],
    key: str,
    *,
    default_length: int,
) -> list[float]:
    value = data.get(key)
    if value is None:
        return [1.0] * default_length
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, (float, int)) for item in value)
    ):
        raise ValueError(f"config optional field '{key}' must be a numeric list")
    return [float(item) for item in value]


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

    loss_types = _require_str_or_str_list(training, "loss_type")
    loss_weights = _optional_numeric_list(
        training,
        "loss_weights",
        default_length=len(loss_types),
    )
    training_config = DpoTrainingConfig(
        method=method,
        beta=_require_float_or_int(training, "beta"),
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
        dataset_num_proc=_require_int(training, "dataset_num_proc"),
        loss_type=loss_types,
        loss_weights=loss_weights,
        truncation_mode=_require_str(training, "truncation_mode"),
        precompute_ref_log_probs=_require_bool(
            training, "precompute_ref_log_probs"
        ),
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
        "dataset_num_proc": training_config.dataset_num_proc,
    }
    for field, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"DPO training.{field} must be positive")
    if training_config.dataloader_num_workers < 0:
        raise ValueError(
            "DPO training.dataloader_num_workers must be non-negative"
        )
    if training_config.beta <= 0:
        raise ValueError("DPO training.beta must be positive")
    if training_config.learning_rate <= 0:
        raise ValueError("DPO training.learning_rate must be positive")
    if training_config.num_train_epochs <= 0:
        raise ValueError("DPO training.num_train_epochs must be positive")
    if not 0 <= training_config.warmup_ratio < 1:
        raise ValueError(
            "DPO training.warmup_ratio must be greater than or equal to zero "
            "and less than one"
        )
    if training_config.weight_decay < 0:
        raise ValueError(
            "DPO training.weight_decay must be greater than or equal to zero"
        )
    if training_config.max_grad_norm <= 0:
        raise ValueError("DPO training.max_grad_norm must be positive")
    if training_config.truncation_mode != "keep_start":
        raise ValueError(
            "DPO training.truncation_mode must be 'keep_start'"
        )
    supported_loss_types = {
        "sigmoid",
        "hinge",
        "ipo",
        "exo_pair",
        "nca_pair",
        "robust",
        "bco_pair",
        "sppo_hard",
        "aot",
        "aot_unpaired",
        "apo_zero",
        "apo_down",
        "discopop",
        "sft",
        "sigmoid_norm",
    }
    unsupported_loss_types = sorted(
        set(training_config.loss_type) - supported_loss_types
    )
    if unsupported_loss_types:
        raise ValueError(
            "DPO training.loss_type contains unsupported value(s): "
            + ", ".join(unsupported_loss_types)
        )
    if len(training_config.loss_weights) != len(training_config.loss_type):
        raise ValueError(
            "DPO training.loss_weights must match training.loss_type length"
        )
    if any(weight <= 0 for weight in training_config.loss_weights):
        raise ValueError("DPO training.loss_weights values must be positive")
    if (
        training_config.max_steps is not None
        and training_config.max_steps <= 0
    ):
        raise ValueError("DPO training.max_steps must be positive when set")
    if (
        training_config.max_train_samples is not None
        and training_config.max_train_samples <= 0
    ):
        raise ValueError(
            "DPO training.max_train_samples must be positive when set"
        )

    return DpoConfig(
        source=DpoSourceConfig(
            model_name=_require_str(source, "model_name"),
            checkpoint_path=_require_str(source, "checkpoint_path"),
            tokenizer_path=_require_str(source, "tokenizer_path"),
            revision=_optional_str(source, "revision"),
        ),
        data=DpoDataConfig(
            dataset_id=_require_str(data_cfg, "dataset_id"),
            dataset_config_name=_optional_str(data_cfg, "dataset_config_name"),
            dataset_split=_require_str(data_cfg, "dataset_split"),
            id_field=_optional_str(data_cfg, "id_field"),
            prompt_field=_require_str(data_cfg, "prompt_field"),
            chosen_field=_require_str(data_cfg, "chosen_field"),
            rejected_field=_require_str(data_cfg, "rejected_field"),
            metadata_field=_optional_str(data_cfg, "metadata_field"),
        ),
        training=training_config,
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
    data_cfg = _require_mapping(data, "data")
    formatting = _require_mapping(data, "formatting")
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

    formatting_mode = _require_str(formatting, "mode")
    if formatting_mode not in {"chat", "plain"}:
        raise ValueError("logit formatting.mode must be 'chat' or 'plain'")

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

    distillation_config = LogitDistillationSettings(
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
        num_train_epochs=_require_float_or_int(
            distillation, "num_train_epochs"
        ),
        warmup_ratio=_require_float_or_int(distillation, "warmup_ratio"),
        weight_decay=_require_float_or_int(distillation, "weight_decay"),
        max_grad_norm=_require_float_or_int(distillation, "max_grad_norm"),
        lr_scheduler_type=_require_str(distillation, "lr_scheduler_type"),
        logging_steps=_require_int(distillation, "logging_steps"),
        save_steps=_require_int(distillation, "save_steps"),
        save_total_limit=_require_int(distillation, "save_total_limit"),
        bf16=_require_bool(distillation, "bf16"),
        gradient_checkpointing=_require_bool(
            distillation, "gradient_checkpointing"
        ),
        dataloader_num_workers=_require_int(
            distillation, "dataloader_num_workers"
        ),
        seed=_require_int(distillation, "seed"),
        max_steps=_optional_int(distillation, "max_steps"),
        max_train_samples=_optional_int(distillation, "max_train_samples"),
    )
    positive_ints = {
        "max_length": distillation_config.max_length,
        "per_device_train_batch_size": (
            distillation_config.per_device_train_batch_size
        ),
        "gradient_accumulation_steps": (
            distillation_config.gradient_accumulation_steps
        ),
        "logging_steps": distillation_config.logging_steps,
        "save_steps": distillation_config.save_steps,
        "save_total_limit": distillation_config.save_total_limit,
    }
    for field, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"logit distillation.{field} must be positive")
    if distillation_config.dataloader_num_workers < 0:
        raise ValueError(
            "logit distillation.dataloader_num_workers must be non-negative"
        )
    if distillation_config.temperature <= 0:
        raise ValueError("logit distillation.temperature must be positive")
    if not 0 <= distillation_config.alpha <= 1:
        raise ValueError("logit distillation.alpha must be between zero and one")
    if distillation_config.learning_rate <= 0:
        raise ValueError("logit distillation.learning_rate must be positive")
    if distillation_config.num_train_epochs <= 0:
        raise ValueError("logit distillation.num_train_epochs must be positive")
    if not 0 <= distillation_config.warmup_ratio < 1:
        raise ValueError(
            "logit distillation.warmup_ratio must be greater than or equal to "
            "zero and less than one"
        )
    if distillation_config.weight_decay < 0:
        raise ValueError(
            "logit distillation.weight_decay must be greater than or equal to zero"
        )
    if distillation_config.max_grad_norm <= 0:
        raise ValueError("logit distillation.max_grad_norm must be positive")
    if (
        distillation_config.max_steps is not None
        and distillation_config.max_steps <= 0
    ):
        raise ValueError(
            "logit distillation.max_steps must be positive when set"
        )
    if (
        distillation_config.max_train_samples is not None
        and distillation_config.max_train_samples <= 0
    ):
        raise ValueError(
            "logit distillation.max_train_samples must be positive when set"
        )

    return LogitDistillConfig(
        teacher=LogitTeacherConfig(
            provider=provider,
            model_name=_require_str(teacher, "model_name"),
            checkpoint_path=_require_str(teacher, "checkpoint_path"),
            tokenizer_path=_require_str(teacher, "tokenizer_path"),
            revision=_optional_str(teacher, "revision"),
        ),
        student=LogitStudentConfig(
            model_name=_require_str(student, "model_name"),
            checkpoint_path=_require_str(student, "checkpoint_path"),
            tokenizer_path=_require_str(student, "tokenizer_path"),
            revision=_optional_str(student, "revision"),
        ),
        data=LogitDataConfig(
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
        distillation=distillation_config,
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
            model_name=_require_str(output, "model_name"),
            run_dir=_require_str(output, "run_dir"),
            checkpoint_dir=_require_str(output, "checkpoint_dir"),
            final_checkpoint_dir=_require_str(
                output, "final_checkpoint_dir"
            ),
        ),
    )


def load_eval_config(path: str | Path) -> EvalConfig:
    data = load_yaml(path)
    raw_models = data.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("eval config requires a non-empty 'models' list")

    models: list[EvalModelConfig] = []
    names: set[str] = set()
    for index, raw_model in enumerate(raw_models):
        if not isinstance(raw_model, dict):
            raise ValueError(f"eval models item {index} must be a mapping")
        model = EvalModelConfig(
            name=_require_str(raw_model, "name"),
            model_name_or_path=_require_str(
                raw_model, "model_name_or_path"
            ),
            tokenizer_name_or_path=_require_str(
                raw_model, "tokenizer_name_or_path"
            ),
            revision=_optional_str(raw_model, "revision"),
        )
        if model.name in names:
            raise ValueError(f"eval model name must be unique: {model.name}")
        names.add(model.name)
        models.append(model)

    data_cfg = _require_mapping(data, "data")
    preference_data = _require_mapping(data, "preference_data")
    formatting = _require_mapping(data, "formatting")
    generation = _require_mapping(data, "generation")
    output = _require_mapping(data, "output")

    formatting_mode = _require_str(formatting, "mode")
    if formatting_mode not in {"chat", "plain"}:
        raise ValueError("eval formatting.mode must be 'chat' or 'plain'")

    generation_config = EvalGenerationConfig(
        max_input_length=_require_int(generation, "max_input_length"),
        max_new_tokens=_require_int(generation, "max_new_tokens"),
        per_device_batch_size=_require_int(
            generation, "per_device_batch_size"
        ),
        do_sample=_require_bool(generation, "do_sample"),
        temperature=_require_float_or_int(generation, "temperature"),
        seed=_require_int(generation, "seed"),
        limit=_require_int(generation, "limit"),
    )
    positive_values = {
        "max_input_length": generation_config.max_input_length,
        "max_new_tokens": generation_config.max_new_tokens,
        "per_device_batch_size": generation_config.per_device_batch_size,
        "limit": generation_config.limit,
    }
    for field, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"eval generation.{field} must be positive")
    if generation_config.do_sample and generation_config.temperature <= 0:
        raise ValueError(
            "eval generation.temperature must be positive when sampling"
        )
    if not generation_config.do_sample and generation_config.temperature < 0:
        raise ValueError(
            "eval generation.temperature must be non-negative"
        )

    return EvalConfig(
        models=models,
        data=EvalDataConfig(
            dataset_id=_require_str(data_cfg, "dataset_id"),
            dataset_config_name=_optional_str(data_cfg, "dataset_config_name"),
            dataset_split=_require_str(data_cfg, "dataset_split"),
            id_field=_optional_str(data_cfg, "id_field"),
            prompt_field=_require_str(data_cfg, "prompt_field"),
            reference_field=_require_str(data_cfg, "reference_field"),
        ),
        preference_data=EvalPreferenceDataConfig(
            dataset_id=_require_str(preference_data, "dataset_id"),
            dataset_config_name=_optional_str(
                preference_data, "dataset_config_name"
            ),
            dataset_split=_require_str(preference_data, "dataset_split"),
            id_field=_optional_str(preference_data, "id_field"),
            prompt_field=_require_str(preference_data, "prompt_field"),
            chosen_field=_require_str(preference_data, "chosen_field"),
            rejected_field=_require_str(preference_data, "rejected_field"),
        ),
        formatting=ResponseFormattingConfig(
            mode=formatting_mode,
            system_prompt=_optional_str(formatting, "system_prompt"),
        ),
        generation=generation_config,
        output=EvalOutputConfig(
            results_path=_require_str(output, "results_path"),
            predictions_dir=_require_str(output, "predictions_dir"),
        ),
    )
