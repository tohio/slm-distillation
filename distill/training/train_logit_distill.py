from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distill.data.response import (
    ResponseDatasetSummary,
    convert_response_row,
    inspect_response_dataset,
    load_response_dataset,
    validate_response_dataset,
)
from distill.models.resolve import (
    ModelInfoLoader,
    ModelResolution,
    build_model_load_kwargs,
    resolve_model_reference,
)
from distill.training.train_response_distill import encode_response_record
from distill.utils.config import (
    LogitDistillConfig,
    ResponseDataConfig,
    load_logit_distill_config,
)
from distill.utils.env import configure_wandb_environment, get_env_value
from distill.utils.hardware import validate_single_cuda_gpu
from distill.utils.logging import install_compact_logging
from distill.utils.tokenizer_compat import (
    TokenizerCompatibilityResult,
    TokenizerLoader,
    assert_tokenizer_references_compatible,
)


@dataclass(frozen=True)
class LogitDistillationPlan:
    teacher_checkpoint_path: str
    teacher_tokenizer_path: str
    teacher_revision: str | None
    student_checkpoint_path: str
    student_tokenizer_path: str
    student_revision: str | None
    dataset_id: str
    dataset_split: str
    output_dir: str
    final_checkpoint_dir: str
    temperature: float
    alpha: float
    max_length: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: float
    bf16: bool
    max_steps: int | None
    max_train_samples: int | None
    require_same_tokenizer: bool


@dataclass(frozen=True)
class LogitInputValidation:
    teacher: ModelResolution
    student: ModelResolution
    tokenizer_compatibility: TokenizerCompatibilityResult
    dataset: ResponseDatasetSummary


@dataclass(frozen=True)
class LogitTrainingResult:
    final_checkpoint_dir: str
    train_samples: int
    global_step: int
    training_loss: float | None


def compute_logit_distillation_loss(
    *,
    student_logits: Any,
    teacher_logits: Any,
    labels: Any,
    hard_loss: Any,
    temperature: float,
    alpha: float,
    return_components: bool = False,
) -> Any:
    import torch.nn.functional as functional

    shifted_student_logits = student_logits[:, :-1, :]
    shifted_teacher_logits = teacher_logits[:, :-1, :]
    supervised_mask = labels[:, 1:].ne(-100)
    if not bool(supervised_mask.any()):
        raise ValueError(
            "Logit distillation batch contains no supervised tokens"
        )

    student_log_probs = functional.log_softmax(
        shifted_student_logits / temperature,
        dim=-1,
    )
    teacher_probs = functional.softmax(
        shifted_teacher_logits / temperature,
        dim=-1,
    )
    token_kl = functional.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none",
    ).sum(dim=-1)
    soft_loss = (
        token_kl.masked_select(supervised_mask).mean()
        * temperature
        * temperature
    )
    loss = alpha * hard_loss + (1.0 - alpha) * soft_loss
    if return_components:
        return loss, hard_loss, soft_loss
    return loss


def _response_data_config(config: LogitDistillConfig) -> ResponseDataConfig:
    return ResponseDataConfig(
        dataset_id=config.data.dataset_id,
        dataset_config_name=config.data.dataset_config_name,
        dataset_split=config.data.dataset_split,
        id_field=config.data.id_field,
        prompt_field=config.data.prompt_field,
        response_field=config.data.response_field,
        metadata_field=config.data.metadata_field,
    )


def build_logit_distillation_plan(
    config: LogitDistillConfig,
) -> LogitDistillationPlan:
    return LogitDistillationPlan(
        teacher_checkpoint_path=config.teacher.checkpoint_path,
        teacher_tokenizer_path=config.teacher.tokenizer_path,
        teacher_revision=config.teacher.revision,
        student_checkpoint_path=config.student.checkpoint_path,
        student_tokenizer_path=config.student.tokenizer_path,
        student_revision=config.student.revision,
        dataset_id=config.data.dataset_id,
        dataset_split=config.data.dataset_split,
        output_dir=config.output.checkpoint_dir,
        final_checkpoint_dir=config.output.final_checkpoint_dir,
        temperature=config.distillation.temperature,
        alpha=config.distillation.alpha,
        max_length=config.distillation.max_length,
        per_device_train_batch_size=(
            config.distillation.per_device_train_batch_size
        ),
        gradient_accumulation_steps=(
            config.distillation.gradient_accumulation_steps
        ),
        learning_rate=config.distillation.learning_rate,
        num_train_epochs=config.distillation.num_train_epochs,
        bf16=config.distillation.bf16,
        max_steps=config.distillation.max_steps,
        max_train_samples=config.distillation.max_train_samples,
        require_same_tokenizer=config.compatibility.require_same_tokenizer,
    )


def load_logit_distillation_plan(config_path: str) -> LogitDistillationPlan:
    return build_logit_distillation_plan(load_logit_distill_config(config_path))


def validate_logit_inputs(
    config_path: str,
    *,
    limit: int | None = None,
    model_info_loader: ModelInfoLoader | None = None,
    tokenizer_loader: TokenizerLoader | None = None,
    dataset_loader: Any = None,
) -> LogitInputValidation:
    config = load_logit_distill_config(config_path)
    hf_token = get_env_value("HF_TOKEN", fallback_to_os=True)
    teacher = resolve_model_reference(
        config.teacher.checkpoint_path,
        revision=config.teacher.revision,
        model_info_loader=model_info_loader,
    )
    student = resolve_model_reference(
        config.student.checkpoint_path,
        revision=config.student.revision,
        model_info_loader=model_info_loader,
    )

    if not config.compatibility.require_same_tokenizer:
        raise ValueError(
            "Logit distillation requires tokenizer compatibility validation"
        )
    compatibility = assert_tokenizer_references_compatible(
        config.teacher.tokenizer_path,
        config.student.tokenizer_path,
        teacher_revision=config.teacher.revision,
        student_revision=config.student.revision,
        token=hf_token,
        tokenizer_loader=tokenizer_loader,
    )
    dataset = inspect_response_dataset(
        _response_data_config(config),
        limit=limit,
        loader=dataset_loader,
    )
    return LogitInputValidation(
        teacher=teacher,
        student=student,
        tokenizer_compatibility=compatibility,
        dataset=dataset,
    )


def prepare_logit_training_dataset(
    config: LogitDistillConfig,
    *,
    tokenizer: Any,
    dataset_loader: Any = None,
    max_train_samples: int | None = None,
) -> Any:
    data_config = _response_data_config(config)
    dataset = load_response_dataset(data_config, loader=dataset_loader)
    validate_response_dataset(dataset, data_config)

    sample_limit = (
        config.distillation.max_train_samples
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
            config=data_config,
        )
        return encode_response_record(
            record,
            tokenizer=tokenizer,
            formatting=config.formatting,
            max_length=config.distillation.max_length,
        )

    return dataset.map(
        encode_row,
        with_indices=True,
        remove_columns=original_columns,
        desc="Tokenizing logit-distillation dataset",
    )


def _validate_training_hardware(config: LogitDistillConfig, torch: Any) -> None:
    if not config.hardware.single_gpu_required:
        raise ValueError(
            "Logit distillation hardware.single_gpu_required must be true"
        )
    validate_single_cuda_gpu(
        torch,
        stage="Logit distillation",
        allowed_gpu_classes=config.hardware.allowed_gpu_classes,
    )


def train_logit_distill(
    config_path: str,
    *,
    max_steps: int | None = None,
    max_train_samples: int | None = None,
    resume_from_checkpoint: str | None = None,
    dataset_loader: Any = None,
) -> LogitTrainingResult:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    class LogitDistillationTrainer(Trainer):
        def __init__(
            self,
            *,
            teacher_model: Any,
            temperature: float,
            alpha: float,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.model_accepts_loss_kwargs = False
            self.teacher_model = teacher_model
            self.distillation_temperature = temperature
            self.distillation_alpha = alpha
            self._hard_loss_sum = None
            self._soft_loss_sum = None
            self._component_loss_count = 0

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            del num_items_in_batch
            labels = inputs["labels"]
            student_outputs = model(**inputs)
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=False,
                )
            loss, hard_loss, soft_loss = compute_logit_distillation_loss(
                student_logits=student_outputs.logits,
                teacher_logits=teacher_outputs.logits,
                labels=labels,
                hard_loss=student_outputs.loss,
                temperature=self.distillation_temperature,
                alpha=self.distillation_alpha,
                return_components=True,
            )
            detached_hard = hard_loss.detach()
            detached_soft = soft_loss.detach()
            self._hard_loss_sum = (
                detached_hard
                if self._hard_loss_sum is None
                else self._hard_loss_sum + detached_hard
            )
            self._soft_loss_sum = (
                detached_soft
                if self._soft_loss_sum is None
                else self._soft_loss_sum + detached_soft
            )
            self._component_loss_count += 1
            if return_outputs:
                return loss, student_outputs
            return loss

        def log(self, logs: dict[str, Any], *args: Any, **kwargs: Any) -> None:
            enriched = dict(logs)
            if "loss" in enriched and self._component_loss_count:
                enriched["loss/hard"] = float(
                    self._hard_loss_sum.item() / self._component_loss_count
                )
                enriched["loss/soft_kl"] = float(
                    self._soft_loss_sum.item() / self._component_loss_count
                )
                self._hard_loss_sum = None
                self._soft_loss_sum = None
                self._component_loss_count = 0
            super().log(enriched, *args, **kwargs)

    config = load_logit_distill_config(config_path)
    _validate_training_hardware(config, torch)
    validation = validate_logit_inputs(
        config_path,
        dataset_loader=dataset_loader,
    )
    if not validation.tokenizer_compatibility.compatible:
        raise ValueError("Teacher and student tokenizers are incompatible")

    hf_token = get_env_value("HF_TOKEN", fallback_to_os=True)

    def model_kwargs(
        revision: str | None,
        *,
        dtype: Any = None,
    ) -> dict[str, Any]:
        return build_model_load_kwargs(
            revision=revision,
            token=hf_token,
            dtype=dtype,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        config.student.tokenizer_path,
        **model_kwargs(config.student.revision),
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None or tokenizer.eos_token is None:
            raise ValueError(
                "Logit distillation tokenizer requires a pad token or EOS token"
            )
        tokenizer.pad_token = tokenizer.eos_token

    student = AutoModelForCausalLM.from_pretrained(
        config.student.checkpoint_path,
        **model_kwargs(config.student.revision),
    )
    teacher_kwargs = model_kwargs(
        config.teacher.revision,
        dtype=torch.bfloat16 if config.distillation.bf16 else None,
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        config.teacher.checkpoint_path,
        **teacher_kwargs,
    )
    teacher.requires_grad_(False)
    teacher.eval()
    teacher.config.use_cache = False

    resolved_max_steps = (
        config.distillation.max_steps if max_steps is None else max_steps
    )
    if resolved_max_steps is not None and resolved_max_steps <= 0:
        raise ValueError("max_steps must be positive when set")

    training_args = TrainingArguments(
        output_dir=config.output.checkpoint_dir,
        do_train=True,
        per_device_train_batch_size=(
            config.distillation.per_device_train_batch_size
        ),
        gradient_accumulation_steps=(
            config.distillation.gradient_accumulation_steps
        ),
        learning_rate=config.distillation.learning_rate,
        num_train_epochs=config.distillation.num_train_epochs,
        warmup_steps=config.distillation.warmup_ratio,
        weight_decay=config.distillation.weight_decay,
        max_grad_norm=config.distillation.max_grad_norm,
        lr_scheduler_type=config.distillation.lr_scheduler_type,
        logging_strategy="steps",
        logging_steps=config.distillation.logging_steps,
        save_strategy="steps",
        save_steps=config.distillation.save_steps,
        save_total_limit=config.distillation.save_total_limit,
        bf16=config.distillation.bf16,
        gradient_checkpointing=(
            config.distillation.gradient_checkpointing
        ),
        dataloader_num_workers=(
            config.distillation.dataloader_num_workers
        ),
        seed=config.distillation.seed,
        max_steps=resolved_max_steps if resolved_max_steps is not None else -1,
        report_to=configure_wandb_environment(
            run_name=config.output.model_name,
            stage="logit-distill",
        ),
        disable_tqdm=True,
        remove_unused_columns=False,
    )
    teacher.to(training_args.device)
    with training_args.main_process_first(
        desc="Prepare logit-distillation dataset"
    ):
        train_dataset = prepare_logit_training_dataset(
            config,
            tokenizer=tokenizer,
            dataset_loader=dataset_loader,
            max_train_samples=max_train_samples,
        )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=student,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    original_use_cache = getattr(student.config, "use_cache", None)
    student.config.use_cache = False
    trainer = LogitDistillationTrainer(
        teacher_model=teacher,
        temperature=config.distillation.temperature,
        alpha=config.distillation.alpha,
        model=student,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    install_compact_logging(trainer, stage="logit-distill")
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
    return LogitTrainingResult(
        final_checkpoint_dir=str(final_checkpoint_dir),
        train_samples=len(train_dataset),
        global_step=int(trainer.state.global_step),
        training_loss=float(raw_loss) if raw_loss is not None else None,
    )
