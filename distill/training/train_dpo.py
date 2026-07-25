from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distill.data.preference import (
    PreferenceDatasetSummary,
    convert_preference_row,
    inspect_preference_dataset,
    load_preference_dataset,
    validate_preference_dataset,
)
from distill.models.resolve import (
    ModelInfoLoader,
    ModelResolution,
    resolve_model_reference,
)
from distill.utils.config import DpoConfig, load_dpo_config
from distill.utils.env import get_env_value


@dataclass(frozen=True)
class DpoTrainingPlan:
    source_checkpoint: str
    tokenizer_path: str
    revision: str | None
    dataset_id: str
    dataset_config_name: str | None
    dataset_split: str
    prompt_field: str
    chosen_field: str
    rejected_field: str
    output_dir: str
    final_checkpoint_dir: str
    beta: float
    loss_type: str
    max_length: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: float
    bf16: bool
    gradient_checkpointing: bool
    precompute_ref_log_probs: bool
    max_steps: int | None
    max_train_samples: int | None


@dataclass(frozen=True)
class DpoInputValidation:
    model: ModelResolution
    tokenizer: ModelResolution
    dataset: PreferenceDatasetSummary


@dataclass(frozen=True)
class DpoTrainingResult:
    final_checkpoint_dir: str
    train_samples: int
    global_step: int
    training_loss: float | None


def build_dpo_training_plan(config: DpoConfig) -> DpoTrainingPlan:
    return DpoTrainingPlan(
        source_checkpoint=config.source.checkpoint_path,
        tokenizer_path=config.source.tokenizer_path,
        revision=config.source.revision,
        dataset_id=config.data.dataset_id,
        dataset_config_name=config.data.dataset_config_name,
        dataset_split=config.data.dataset_split,
        prompt_field=config.data.prompt_field,
        chosen_field=config.data.chosen_field,
        rejected_field=config.data.rejected_field,
        output_dir=config.output.checkpoint_dir,
        final_checkpoint_dir=config.output.final_checkpoint_dir,
        beta=config.training.beta,
        loss_type=config.training.loss_type,
        max_length=config.training.max_length,
        per_device_train_batch_size=(
            config.training.per_device_train_batch_size
        ),
        gradient_accumulation_steps=(
            config.training.gradient_accumulation_steps
        ),
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.num_train_epochs,
        bf16=config.training.bf16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        precompute_ref_log_probs=config.training.precompute_ref_log_probs,
        max_steps=config.training.max_steps,
        max_train_samples=config.training.max_train_samples,
    )


def load_dpo_training_plan(config_path: str) -> DpoTrainingPlan:
    return build_dpo_training_plan(load_dpo_config(config_path))


def validate_dpo_inputs(
    config_path: str,
    *,
    limit: int | None = None,
    model_info_loader: ModelInfoLoader | None = None,
    dataset_loader: Any = None,
) -> DpoInputValidation:
    config = load_dpo_config(config_path)
    model = resolve_model_reference(
        config.source.checkpoint_path,
        revision=config.source.revision,
        model_info_loader=model_info_loader,
    )
    if config.source.tokenizer_path == config.source.checkpoint_path:
        tokenizer = model
    else:
        tokenizer = resolve_model_reference(
            config.source.tokenizer_path,
            revision=config.source.revision,
            model_info_loader=model_info_loader,
        )

    dataset = inspect_preference_dataset(
        config.data,
        limit=limit,
        loader=dataset_loader,
    )
    return DpoInputValidation(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
    )


def prepare_dpo_training_dataset(
    config: DpoConfig,
    *,
    dataset_loader: Any = None,
    max_train_samples: int | None = None,
) -> Any:
    dataset = load_preference_dataset(config.data, loader=dataset_loader)
    validate_preference_dataset(dataset, config.data)

    sample_limit = (
        config.training.max_train_samples
        if max_train_samples is None
        else max_train_samples
    )
    if sample_limit is not None:
        if sample_limit <= 0:
            raise ValueError("max_train_samples must be positive when set")
        dataset = dataset.select(range(min(sample_limit, len(dataset))))

    original_columns = list(dataset.column_names)

    def convert_row(row: Mapping[str, Any], row_index: int) -> dict[str, Any]:
        record = convert_preference_row(
            row,
            row_index=row_index,
            config=config.data,
        )
        return {
            "prompt": record.prompt,
            "chosen": record.chosen,
            "rejected": record.rejected,
        }

    return dataset.map(
        convert_row,
        with_indices=True,
        remove_columns=original_columns,
        desc="Preparing DPO preference dataset",
    )


def _configure_tokenizer(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None or tokenizer.eos_token is None:
            raise ValueError("DPO tokenizer requires a pad token or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


def train_dpo(
    config_path: str,
    *,
    max_steps: int | None = None,
    max_train_samples: int | None = None,
    resume_from_checkpoint: str | None = None,
    dataset_loader: Any = None,
) -> DpoTrainingResult:
    from transformers import AutoTokenizer
    from trl import DPOConfig as TrlDpoConfig
    from trl import DPOTrainer

    config = load_dpo_config(config_path)
    hf_token = get_env_value("HF_TOKEN", fallback_to_os=True)
    model_init_kwargs: dict[str, Any] = {}
    if config.source.revision is not None:
        model_init_kwargs["revision"] = config.source.revision
    if hf_token is not None:
        model_init_kwargs["token"] = hf_token

    tokenizer = AutoTokenizer.from_pretrained(
        config.source.tokenizer_path,
        **model_init_kwargs,
    )
    _configure_tokenizer(tokenizer)

    resolved_max_steps = (
        config.training.max_steps if max_steps is None else max_steps
    )
    if resolved_max_steps is not None and resolved_max_steps <= 0:
        raise ValueError("max_steps must be positive when set")

    training_args = TrlDpoConfig(
        output_dir=config.output.checkpoint_dir,
        do_train=True,
        per_device_train_batch_size=(
            config.training.per_device_train_batch_size
        ),
        gradient_accumulation_steps=(
            config.training.gradient_accumulation_steps
        ),
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.num_train_epochs,
        warmup_ratio=config.training.warmup_ratio,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        lr_scheduler_type=config.training.lr_scheduler_type,
        logging_strategy="steps",
        logging_steps=config.training.logging_steps,
        save_strategy="steps",
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        bf16=config.training.bf16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        dataloader_num_workers=config.training.dataloader_num_workers,
        dataset_num_proc=config.training.dataset_num_proc,
        seed=config.training.seed,
        max_steps=resolved_max_steps if resolved_max_steps is not None else -1,
        report_to=[],
        max_length=config.training.max_length,
        truncation_mode=config.training.truncation_mode,
        beta=config.training.beta,
        loss_type=[config.training.loss_type],
        precompute_ref_log_probs=(
            config.training.precompute_ref_log_probs
        ),
        model_init_kwargs=model_init_kwargs or None,
    )
    with training_args.main_process_first(desc="Prepare DPO dataset"):
        train_dataset = prepare_dpo_training_dataset(
            config,
            dataset_loader=dataset_loader,
            max_train_samples=max_train_samples,
        )

    trainer = DPOTrainer(
        model=config.source.checkpoint_path,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    original_use_cache = getattr(trainer.model.config, "use_cache", None)

    train_output = trainer.train(
        resume_from_checkpoint=resume_from_checkpoint,
    )
    final_checkpoint_dir = Path(config.output.final_checkpoint_dir)
    if original_use_cache is not None:
        trainer.model.config.use_cache = original_use_cache
    trainer.save_model(str(final_checkpoint_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_checkpoint_dir)
    trainer.save_state()
    trainer.log_metrics("train", train_output.metrics)
    trainer.save_metrics("train", train_output.metrics)

    raw_loss = train_output.metrics.get("train_loss")
    return DpoTrainingResult(
        final_checkpoint_dir=str(final_checkpoint_dir),
        train_samples=len(train_dataset),
        global_step=int(trainer.state.global_step),
        training_loss=float(raw_loss) if raw_loss is not None else None,
    )
