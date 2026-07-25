from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from distill.data.response import ResponseDatasetSummary, inspect_response_dataset
from distill.models.resolve import ModelInfoLoader, ModelResolution, resolve_model_reference
from distill.utils.config import (
    ResponseDistillConfig,
    load_response_distill_config,
)


@dataclass(frozen=True)
class ResponseTrainingPlan:
    model_name_or_path: str
    tokenizer_name_or_path: str
    revision: str | None
    dataset_id: str
    dataset_config_name: str | None
    dataset_split: str
    prompt_field: str
    response_field: str
    output_dir: str
    final_checkpoint_dir: str


@dataclass(frozen=True)
class ResponseInputValidation:
    model: ModelResolution
    tokenizer: ModelResolution
    dataset: ResponseDatasetSummary


def build_response_training_plan(
    config: ResponseDistillConfig,
) -> ResponseTrainingPlan:
    return ResponseTrainingPlan(
        model_name_or_path=config.model.model_name_or_path,
        tokenizer_name_or_path=config.model.tokenizer_name_or_path,
        revision=config.model.revision,
        dataset_id=config.data.dataset_id,
        dataset_config_name=config.data.dataset_config_name,
        dataset_split=config.data.dataset_split,
        prompt_field=config.data.prompt_field,
        response_field=config.data.response_field,
        output_dir=config.output.checkpoint_dir,
        final_checkpoint_dir=config.output.final_checkpoint_dir,
    )


def load_response_training_plan(config_path: str) -> ResponseTrainingPlan:
    return build_response_training_plan(load_response_distill_config(config_path))


def validate_response_inputs(
    config_path: str,
    *,
    limit: int | None = None,
    model_info_loader: ModelInfoLoader | None = None,
    dataset_loader: Any = None,
) -> ResponseInputValidation:
    config = load_response_distill_config(config_path)
    model = resolve_model_reference(
        config.model.model_name_or_path,
        revision=config.model.revision,
        model_info_loader=model_info_loader,
    )

    if config.model.tokenizer_name_or_path == config.model.model_name_or_path:
        tokenizer = model
    else:
        tokenizer = resolve_model_reference(
            config.model.tokenizer_name_or_path,
            revision=config.model.revision,
            model_info_loader=model_info_loader,
        )

    dataset = inspect_response_dataset(
        config.data,
        limit=limit,
        loader=dataset_loader,
    )
    return ResponseInputValidation(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
    )


def train_response_distill(config_path: str) -> None:
    load_response_training_plan(config_path)
    raise NotImplementedError(
        "Response training inputs are configured and validated. "
        "The trainer implementation is not available yet."
    )
