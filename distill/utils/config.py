from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TeacherConfig:
    name: str
    provider: str
    model: str
    mode: str
    purpose: str
    distillation_allowed: bool | str
    input_price_per_1m: float | None = None
    output_price_per_1m: float | None = None


@dataclass(frozen=True)
class TeachersConfig:
    default_teacher: str
    teachers: dict[str, TeacherConfig]


@dataclass(frozen=True)
class StudentConfig:
    name: str
    checkpoint_path: str
    tokenizer_path: str


@dataclass(frozen=True)
class ResponseDistillationSettings:
    mode: str
    target_tokens: int
    max_output_tokens: int
    temperature: float
    top_p: float
    max_retries: int
    retry_delay_seconds: float
    continue_on_error: bool


@dataclass(frozen=True)
class ResponseDataConfig:
    prompts_path: str | None
    prompts_paths: list[str]
    raw_teacher_path: str
    validated_path: str
    rejected_path: str
    distill_dataset_path: str


@dataclass(frozen=True)
class ValidationConfig:
    require_non_empty_output: bool
    reject_refusals_when_not_expected: bool
    reject_code_fences_for_function_body_tasks: bool
    max_retries: int


@dataclass(frozen=True)
class OutputConfig:
    model_name: str
    source_model_name: str
    teacher_family: str
    run_dir: str
    checkpoint_dir: str
    final_checkpoint_dir: str
    export_repo: str


@dataclass(frozen=True)
class ModelCardConfig:
    source_checkpoint: str
    teacher_model: str
    teacher_provider: str
    distillation_type: str
    dpo_applied: bool


@dataclass(frozen=True)
class ResponseDistillConfig:
    teacher_name: str
    student: StudentConfig
    distillation: ResponseDistillationSettings
    data: ResponseDataConfig
    validation: ValidationConfig
    output: OutputConfig
    model_card: ModelCardConfig


@dataclass(frozen=True)
class DpoSourceConfig:
    model_name: str
    checkpoint_path: str
    tokenizer_path: str


@dataclass(frozen=True)
class DpoDataConfig:
    preference_dataset_path: str
    rejected_path: str


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
class PreferenceSourceConfig:
    validated_teacher_path: str
    rejected_response_path: str | None


@dataclass(frozen=True)
class PreferenceDataConfig:
    output_path: str
    rejected_path: str


@dataclass(frozen=True)
class PreferenceBuildSettings:
    default_rejected_response: str
    require_non_empty_chosen: bool
    require_non_empty_rejected: bool
    reject_identical_pairs: bool


@dataclass(frozen=True)
class PreferenceConfig:
    source: PreferenceSourceConfig
    data: PreferenceDataConfig
    preference: PreferenceBuildSettings


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


def _optional_str_list(data: dict[str, Any], key: str) -> list[str] | None:
    value = data.get(key)

    if value is None:
        return None

    if not isinstance(value, list) or not value:
        raise ValueError(f"config optional field '{key}' must be a non-empty list")

    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise ValueError(
                f"config optional field '{key}' item {index} must be a non-empty string"
            )
        result.append(item)

    return result


def _prompt_paths_from_data_config(data_cfg: dict[str, Any]) -> tuple[str | None, list[str]]:
    prompts_path = _optional_str(data_cfg, "prompts_path")
    prompts_paths = _optional_str_list(data_cfg, "prompts_paths")

    if prompts_paths is None:
        if prompts_path is None:
            raise ValueError("config requires 'data.prompts_path' or 'data.prompts_paths'")
        return prompts_path, [prompts_path]

    if prompts_path is not None and prompts_path not in prompts_paths:
        raise ValueError(
            "config cannot define both 'data.prompts_path' and "
            "'data.prompts_paths' with different values"
        )

    return prompts_path, prompts_paths


def load_teachers_config(path: str | Path) -> TeachersConfig:
    data = load_yaml(path)

    default_teacher = data.get("default_teacher")
    teachers_raw = data.get("teachers")

    if not isinstance(default_teacher, str) or not default_teacher:
        raise ValueError("teachers config requires non-empty 'default_teacher'")

    if not isinstance(teachers_raw, dict) or not teachers_raw:
        raise ValueError("teachers config requires non-empty 'teachers' mapping")

    teachers: dict[str, TeacherConfig] = {}

    for name, raw in teachers_raw.items():
        if not isinstance(raw, dict):
            raise ValueError(f"teacher '{name}' must be a mapping")

        missing = [
            field
            for field in (
                "provider",
                "model",
                "mode",
                "purpose",
                "distillation_allowed",
            )
            if field not in raw
        ]

        if missing:
            raise ValueError(f"teacher '{name}' missing required fields: {missing}")

        input_price = raw.get("input_price_per_1m")
        output_price = raw.get("output_price_per_1m")

        if input_price is not None and not isinstance(input_price, (int, float)):
            raise ValueError(
                f"teacher '{name}' field 'input_price_per_1m' must be numeric"
            )

        if output_price is not None and not isinstance(output_price, (int, float)):
            raise ValueError(
                f"teacher '{name}' field 'output_price_per_1m' must be numeric"
            )

        teachers[name] = TeacherConfig(
            name=name,
            provider=str(raw["provider"]),
            model=str(raw["model"]),
            mode=str(raw["mode"]),
            purpose=str(raw["purpose"]),
            distillation_allowed=raw["distillation_allowed"],
            input_price_per_1m=(
                float(input_price) if input_price is not None else None
            ),
            output_price_per_1m=(
                float(output_price) if output_price is not None else None
            ),
        )

    if default_teacher not in teachers:
        raise ValueError(
            f"default_teacher '{default_teacher}' not found in teachers mapping"
        )

    return TeachersConfig(default_teacher=default_teacher, teachers=teachers)


def get_default_teacher(config: TeachersConfig) -> TeacherConfig:
    return config.teachers[config.default_teacher]


def load_response_distill_config(path: str | Path) -> ResponseDistillConfig:
    data = load_yaml(path)

    teacher = _require_mapping(data, "teacher")
    student = _require_mapping(data, "student")
    distillation = _require_mapping(data, "distillation")
    data_cfg = _require_mapping(data, "data")
    validation = _require_mapping(data, "validation")
    output = _require_mapping(data, "output")
    model_card = _require_mapping(data, "model_card")

    mode = _require_str(distillation, "mode")
    if mode != "response":
        raise ValueError("response distill config requires distillation.mode='response'")

    return ResponseDistillConfig(
        teacher_name=_require_str(teacher, "name"),
        student=StudentConfig(
            name=_require_str(student, "name"),
            checkpoint_path=_require_str(student, "checkpoint_path"),
            tokenizer_path=_require_str(student, "tokenizer_path"),
        ),
        distillation=ResponseDistillationSettings(
            mode=mode,
            target_tokens=_require_int(distillation, "target_tokens"),
            max_output_tokens=_require_int(distillation, "max_output_tokens"),
            temperature=_require_float_or_int(distillation, "temperature"),
            top_p=_require_float_or_int(distillation, "top_p"),
            max_retries=_require_int(distillation, "max_retries"),
            retry_delay_seconds=_require_float_or_int(
                distillation, "retry_delay_seconds"
            ),
            continue_on_error=_require_bool(distillation, "continue_on_error"),
        ),
        data=ResponseDataConfig(
            prompts_path=_prompt_paths_from_data_config(data_cfg)[0],
            prompts_paths=_prompt_paths_from_data_config(data_cfg)[1],
            raw_teacher_path=_require_str(data_cfg, "raw_teacher_path"),
            validated_path=_require_str(data_cfg, "validated_path"),
            rejected_path=_require_str(data_cfg, "rejected_path"),
            distill_dataset_path=_require_str(data_cfg, "distill_dataset_path"),
        ),
        validation=ValidationConfig(
            require_non_empty_output=_require_bool(
                validation, "require_non_empty_output"
            ),
            reject_refusals_when_not_expected=_require_bool(
                validation, "reject_refusals_when_not_expected"
            ),
            reject_code_fences_for_function_body_tasks=_require_bool(
                validation, "reject_code_fences_for_function_body_tasks"
            ),
            max_retries=_require_int(validation, "max_retries"),
        ),
        output=OutputConfig(
            model_name=_require_str(output, "model_name"),
            source_model_name=_require_str(output, "source_model_name"),
            teacher_family=_require_str(output, "teacher_family"),
            run_dir=_require_str(output, "run_dir"),
            checkpoint_dir=_require_str(output, "checkpoint_dir"),
            final_checkpoint_dir=_require_str(output, "final_checkpoint_dir"),
            export_repo=_require_str(output, "export_repo"),
        ),
        model_card=ModelCardConfig(
            source_checkpoint=_require_str(model_card, "source_checkpoint"),
            teacher_model=_require_str(model_card, "teacher_model"),
            teacher_provider=_require_str(model_card, "teacher_provider"),
            distillation_type=_require_str(model_card, "distillation_type"),
            dpo_applied=_require_bool(model_card, "dpo_applied"),
        ),
    )


def teacher_to_pricing(teacher: TeacherConfig):
    from distill.utils.pricing import ModelPricing

    if teacher.input_price_per_1m is None:
        raise ValueError(f"teacher '{teacher.name}' missing input_price_per_1m")

    if teacher.output_price_per_1m is None:
        raise ValueError(f"teacher '{teacher.name}' missing output_price_per_1m")

    return ModelPricing(
        provider=teacher.provider,
        model=teacher.model,
        input_price_per_1m=teacher.input_price_per_1m,
        output_price_per_1m=teacher.output_price_per_1m,
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
            preference_dataset_path=_require_str(data_cfg, "preference_dataset_path"),
            rejected_path=_require_str(data_cfg, "rejected_path"),
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

def load_preference_config(path: str | Path) -> PreferenceConfig:
    data = load_yaml(path)

    source = _require_mapping(data, "source")
    data_cfg = _require_mapping(data, "data")
    preference = _require_mapping(data, "preference")

    rejected_response_path = source.get("rejected_response_path")
    if rejected_response_path is not None and not isinstance(rejected_response_path, str):
        raise ValueError(
            "config optional field 'source.rejected_response_path' must be a string or null"
        )

    return PreferenceConfig(
        source=PreferenceSourceConfig(
            validated_teacher_path=_require_str(source, "validated_teacher_path"),
            rejected_response_path=rejected_response_path,
        ),
        data=PreferenceDataConfig(
            output_path=_require_str(data_cfg, "output_path"),
            rejected_path=_require_str(data_cfg, "rejected_path"),
        ),
        preference=PreferenceBuildSettings(
            default_rejected_response=_require_str(
                preference, "default_rejected_response"
            ),
            require_non_empty_chosen=_require_bool(
                preference, "require_non_empty_chosen"
            ),
            require_non_empty_rejected=_require_bool(
                preference, "require_non_empty_rejected"
            ),
            reject_identical_pairs=_require_bool(preference, "reject_identical_pairs"),
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
            preference_dataset=_require_str(model_card, "preference_dataset"),
            eval_results_path=_require_str(model_card, "eval_results_path"),
        ),
        export=ExportSettings(
            push_to_hub=_require_bool(export, "push_to_hub"),
            include_tokenizer=_require_bool(export, "include_tokenizer"),
            private=_require_bool(export, "private"),
        ),
    )

