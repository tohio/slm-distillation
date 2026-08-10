from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distill.data.response import (
    CanonicalResponseRecord,
    ResponseDatasetSummary,
    convert_response_row,
    inspect_response_dataset,
    load_response_dataset,
    validate_response_dataset,
)
from distill.models.resolve import (
    ModelInfoLoader,
    ModelResolution,
    resolve_model_reference,
)
from distill.utils.config import (
    ResponseDistillConfig,
    ResponseFormattingConfig,
    load_response_distill_config,
)
from distill.utils.env import get_env_value


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
    formatting_mode: str
    max_length: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: float
    bf16: bool
    gradient_checkpointing: bool
    max_steps: int | None
    max_train_samples: int | None
    output_dir: str
    final_checkpoint_dir: str


@dataclass(frozen=True)
class ResponseInputValidation:
    model: ModelResolution
    tokenizer: ModelResolution
    dataset: ResponseDatasetSummary


@dataclass(frozen=True)
class ResponseTrainingResult:
    final_checkpoint_dir: str
    train_samples: int
    global_step: int
    training_loss: float | None


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
        formatting_mode=config.formatting.mode,
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
        max_steps=config.training.max_steps,
        max_train_samples=config.training.max_train_samples,
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


def _extract_input_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    if not isinstance(encoded, (list, tuple)):
        raise ValueError("Tokenizer did not return a sequence of input IDs")
    return [int(token_id) for token_id in encoded]


def _encode_record_parts(
    record: CanonicalResponseRecord,
    tokenizer: Any,
    formatting: ResponseFormattingConfig,
) -> tuple[list[int], list[int]]:
    if formatting.mode == "chat":
        messages: list[dict[str, str]] = []
        if formatting.system_prompt is not None:
            messages.append(
                {"role": "system", "content": formatting.system_prompt}
            )
        messages.append({"role": "user", "content": record.prompt})
        try:
            prompt_encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            full_encoded = tokenizer.apply_chat_template(
                [
                    *messages,
                    {"role": "assistant", "content": record.response},
                ],
                tokenize=True,
                add_generation_prompt=False,
            )
        except (AttributeError, ValueError, TypeError) as exc:
            raise ValueError(
                "formatting.mode='chat' requires a tokenizer with a usable "
                "chat template; use formatting.mode='plain' otherwise"
            ) from exc
        prompt_ids = _extract_input_ids(prompt_encoded)
        full_ids = _extract_input_ids(full_encoded)
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                "Tokenizer chat template does not produce a stable assistant "
                "response boundary"
            )
        return prompt_ids, full_ids[len(prompt_ids) :]

    prompt_encoded = tokenizer(
        f"{record.prompt.rstrip()}\n",
        add_special_tokens=True,
        truncation=False,
    )
    response_encoded = tokenizer(
        record.response,
        add_special_tokens=False,
        truncation=False,
    )
    return (
        _extract_input_ids(prompt_encoded),
        _extract_input_ids(response_encoded),
    )


def _truncate_prompt(prompt_ids: list[int], budget: int) -> list[int]:
    if len(prompt_ids) <= budget:
        return prompt_ids
    if budget <= 0:
        return []
    if budget == 1:
        return [prompt_ids[-1]]
    return [prompt_ids[0], *prompt_ids[-(budget - 1) :]]


def encode_response_record(
    record: CanonicalResponseRecord,
    *,
    tokenizer: Any,
    formatting: ResponseFormattingConfig,
    max_length: int,
) -> dict[str, list[int]]:
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("Response training tokenizer requires an EOS token")

    prompt_ids, response_ids = _encode_record_parts(
        record,
        tokenizer,
        formatting,
    )
    if not response_ids or response_ids[-1] != eos_token_id:
        response_ids.append(int(eos_token_id))

    if len(response_ids) >= max_length:
        response_ids = response_ids[:max_length]
        response_ids[-1] = int(eos_token_id)
        prompt_ids = []
    else:
        prompt_ids = _truncate_prompt(
            prompt_ids,
            max_length - len(response_ids),
        )

    input_ids = [*prompt_ids, *response_ids]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + response_ids,
    }


def prepare_response_training_dataset(
    config: ResponseDistillConfig,
    *,
    tokenizer: Any,
    dataset_loader: Any = None,
    max_train_samples: int | None = None,
) -> Any:
    dataset = load_response_dataset(config.data, loader=dataset_loader)
    validate_response_dataset(dataset, config.data)

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

    def encode_row(row: Mapping[str, Any], row_index: int) -> dict[str, list[int]]:
        record = convert_response_row(
            row,
            row_index=row_index,
            config=config.data,
        )
        return encode_response_record(
            record,
            tokenizer=tokenizer,
            formatting=config.formatting,
            max_length=config.training.max_length,
        )

    return dataset.map(
        encode_row,
        with_indices=True,
        remove_columns=original_columns,
        desc="Tokenizing response-distillation dataset",
    )


def _configure_tokenizer_padding(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is not None:
        return
    if tokenizer.eos_token_id is None or tokenizer.eos_token is None:
        raise ValueError(
            "Response training tokenizer requires a pad token or EOS token"
        )
    tokenizer.pad_token = tokenizer.eos_token


def train_response_distill(
    config_path: str,
    *,
    max_steps: int | None = None,
    max_train_samples: int | None = None,
    resume_from_checkpoint: str | None = None,
    dataset_loader: Any = None,
) -> ResponseTrainingResult:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    config = load_response_distill_config(config_path)
    hf_token = get_env_value("HF_TOKEN", fallback_to_os=True)
    common_model_kwargs: dict[str, Any] = {
        "revision": config.model.revision,
        "token": hf_token,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_name_or_path,
        **common_model_kwargs,
    )
    _configure_tokenizer_padding(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name_or_path,
        **common_model_kwargs,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    original_use_cache = getattr(model.config, "use_cache", None)
    if config.training.gradient_checkpointing:
        model.config.use_cache = False

    resolved_max_steps = (
        config.training.max_steps if max_steps is None else max_steps
    )
    if resolved_max_steps is not None and resolved_max_steps <= 0:
        raise ValueError("max_steps must be positive when set")

    training_args = TrainingArguments(
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
        warmup_steps=config.training.warmup_ratio,
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
        seed=config.training.seed,
        max_steps=resolved_max_steps if resolved_max_steps is not None else -1,
        report_to=[],
        remove_unused_columns=False,
    )
    with training_args.main_process_first(
        desc="Prepare response-distillation dataset"
    ):
        train_dataset = prepare_response_training_dataset(
            config,
            tokenizer=tokenizer,
            dataset_loader=dataset_loader,
            max_train_samples=max_train_samples,
        )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

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
    return ResponseTrainingResult(
        final_checkpoint_dir=str(final_checkpoint_dir),
        train_samples=len(train_dataset),
        global_step=int(trainer.state.global_step),
        training_loss=float(raw_loss) if raw_loss is not None else None,
    )
