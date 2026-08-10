from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from distill.data.preference import (
    CanonicalPreferenceRecord,
    PreferenceDatasetSummary,
    convert_preference_row,
    load_preference_dataset,
    validate_preference_dataset,
)
from distill.eval.compare_outputs import EvaluationMetrics, score_predictions
from distill.models.resolve import (
    ModelInfoLoader,
    ModelResolution,
    build_model_load_kwargs,
    resolve_model_reference,
)
from distill.utils.config import (
    DpoDataConfig,
    EvalConfig,
    EvalModelConfig,
    load_eval_config,
)
from distill.utils.env import get_env_value


@dataclass(frozen=True)
class EvaluationPlan:
    model_names: list[str]
    model_references: list[str]
    dataset_id: str
    dataset_split: str
    preference_dataset_id: str
    preference_dataset_split: str
    limit: int
    results_path: str
    predictions_dir: str


@dataclass(frozen=True)
class EvalDatasetSummary:
    dataset_id: str
    dataset_split: str
    row_count: int
    inspected_rows: int
    column_names: list[str]


@dataclass(frozen=True)
class EvaluationInputValidation:
    models: list[ModelResolution]
    tokenizers: list[ModelResolution]
    dataset: EvalDatasetSummary
    preference_dataset: PreferenceDatasetSummary


@dataclass(frozen=True)
class ModelEvaluationResult:
    name: str
    model_name_or_path: str
    predictions_path: str
    metrics: EvaluationMetrics
    preference_pairs: int
    preference_accuracy: float


@dataclass(frozen=True)
class ModelComparison:
    baseline: str
    model: str
    exact_match_delta: float
    normalized_exact_match_delta: float
    token_f1_delta: float
    non_empty_rate_delta: float
    preference_accuracy_delta: float


@dataclass(frozen=True)
class EvaluationResult:
    dataset_id: str
    dataset_split: str
    examples: int
    preference_pairs: int
    models: list[ModelEvaluationResult]
    comparisons: list[ModelComparison]
    results_path: str


@dataclass(frozen=True)
class LoadedEvaluationModel:
    model: Any
    tokenizer: Any
    device: Any


def build_evaluation_plan(config: EvalConfig) -> EvaluationPlan:
    return EvaluationPlan(
        model_names=[model.name for model in config.models],
        model_references=[
            model.model_name_or_path for model in config.models
        ],
        dataset_id=config.data.dataset_id,
        dataset_split=config.data.dataset_split,
        preference_dataset_id=config.preference_data.dataset_id,
        preference_dataset_split=config.preference_data.dataset_split,
        limit=config.generation.limit,
        results_path=config.output.results_path,
        predictions_dir=config.output.predictions_dir,
    )


def load_evaluation_plan(config_path: str) -> EvaluationPlan:
    return build_evaluation_plan(load_eval_config(config_path))


def _load_eval_dataset(config: EvalConfig, loader: Any = None) -> Any:
    if loader is None:
        from datasets import load_dataset

        loader = load_dataset

    kwargs: dict[str, Any] = {
        "path": config.data.dataset_id,
        "split": config.data.dataset_split,
    }
    if config.data.dataset_config_name is not None:
        kwargs["name"] = config.data.dataset_config_name
    token = get_env_value("HF_TOKEN", fallback_to_os=True)
    if token is not None:
        kwargs["token"] = token
    return loader(**kwargs)


def _preference_config(config: EvalConfig) -> DpoDataConfig:
    data = config.preference_data
    return DpoDataConfig(
        dataset_id=data.dataset_id,
        dataset_config_name=data.dataset_config_name,
        dataset_split=data.dataset_split,
        id_field=data.id_field,
        prompt_field=data.prompt_field,
        chosen_field=data.chosen_field,
        rejected_field=data.rejected_field,
        metadata_field=None,
    )


def _validate_eval_dataset(
    dataset: Any,
    config: EvalConfig,
    *,
    limit: int,
) -> EvalDatasetSummary:
    required = {
        config.data.prompt_field,
        config.data.reference_field,
    }
    if config.data.id_field is not None:
        required.add(config.data.id_field)
    columns = list(dataset.column_names)
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(
            f"Evaluation dataset is missing configured column(s): "
            f"{', '.join(missing)}"
        )

    inspected = min(limit, len(dataset))
    if inspected <= 0:
        raise ValueError("Evaluation dataset must contain at least one row")
    unique_ids: set[str] = set()
    for row_index in range(inspected):
        row = dataset[row_index]
        for field in (
            config.data.prompt_field,
            config.data.reference_field,
        ):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Evaluation row {row_index} requires non-empty '{field}'"
                )
        if config.data.id_field is not None:
            record_id = row.get(config.data.id_field)
            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError(
                    f"Evaluation row {row_index} requires non-empty "
                    f"'{config.data.id_field}'"
                )
            if record_id in unique_ids:
                raise ValueError(
                    f"Evaluation dataset contains duplicate id: {record_id}"
                )
            unique_ids.add(record_id)

    return EvalDatasetSummary(
        dataset_id=config.data.dataset_id,
        dataset_split=config.data.dataset_split,
        row_count=len(dataset),
        inspected_rows=inspected,
        column_names=sorted(columns),
    )


def validate_evaluation_inputs(
    config_path: str,
    *,
    limit: int | None = None,
    model_info_loader: ModelInfoLoader | None = None,
    dataset_loader: Any = None,
) -> EvaluationInputValidation:
    config = load_eval_config(config_path)
    resolved_limit = config.generation.limit if limit is None else limit
    if resolved_limit <= 0:
        raise ValueError("Evaluation limit must be positive")

    models: list[ModelResolution] = []
    tokenizers: list[ModelResolution] = []
    for model_config in config.models:
        model = resolve_model_reference(
            model_config.model_name_or_path,
            revision=model_config.revision,
            model_info_loader=model_info_loader,
        )
        models.append(model)
        if (
            model_config.tokenizer_name_or_path
            == model_config.model_name_or_path
        ):
            tokenizers.append(model)
        else:
            tokenizers.append(
                resolve_model_reference(
                    model_config.tokenizer_name_or_path,
                    revision=model_config.revision,
                    model_info_loader=model_info_loader,
                )
            )

    dataset = _load_eval_dataset(config, dataset_loader)
    summary = _validate_eval_dataset(
        dataset,
        config,
        limit=resolved_limit,
    )
    preference_config = _preference_config(config)
    preference_dataset = load_preference_dataset(
        preference_config,
        loader=dataset_loader,
    )
    preference_summary = validate_preference_dataset(
        preference_dataset,
        preference_config,
        limit=resolved_limit,
    )
    return EvaluationInputValidation(
        models=models,
        tokenizers=tokenizers,
        dataset=summary,
        preference_dataset=preference_summary,
    )


def _format_prompt(
    prompt: str,
    tokenizer: Any,
    config: EvalConfig,
) -> str:
    if config.formatting.mode == "plain":
        return f"{prompt.rstrip()}\n"

    messages: list[dict[str, str]] = []
    if config.formatting.system_prompt is not None:
        messages.append(
            {"role": "system", "content": config.formatting.system_prompt}
        )
    messages.append({"role": "user", "content": prompt})
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except (AttributeError, ValueError, TypeError) as exc:
        raise ValueError(
            "eval formatting.mode='chat' requires a usable chat template"
        ) from exc


def _select_eval_rows(
    dataset: Any,
    config: EvalConfig,
    *,
    limit: int,
) -> tuple[list[str], list[str], list[str]]:
    selected = dataset.shuffle(seed=config.generation.seed).select(
        range(min(limit, len(dataset)))
    )
    ids: list[str] = []
    prompts: list[str] = []
    references: list[str] = []
    for row_index, row in enumerate(selected):
        if config.data.id_field is None:
            record_id = f"{config.data.dataset_split}-{row_index:08d}"
        else:
            record_id = str(row[config.data.id_field])
        ids.append(record_id)
        prompts.append(str(row[config.data.prompt_field]))
        references.append(str(row[config.data.reference_field]))
    return ids, prompts, references


def _load_evaluation_model(
    model_config: EvalModelConfig,
) -> LoadedEvaluationModel:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = get_env_value("HF_TOKEN", fallback_to_os=True)
    common_kwargs = build_model_load_kwargs(
        revision=model_config.revision,
        token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.tokenizer_name_or_path,
        **common_kwargs,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None or tokenizer.eos_token is None:
            raise ValueError("Evaluation tokenizer requires a pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = build_model_load_kwargs(
        revision=model_config.revision,
        token=hf_token,
        dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        **model_kwargs,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return LoadedEvaluationModel(
        model=model,
        tokenizer=tokenizer,
        device=device,
    )


def _generate_predictions(
    loaded: LoadedEvaluationModel,
    config: EvalConfig,
    prompts: list[str],
) -> list[str]:
    import torch
    from transformers import set_seed

    model = loaded.model
    tokenizer = loaded.tokenizer
    device = loaded.device
    set_seed(config.generation.seed)

    formatted_prompts = [
        _format_prompt(prompt, tokenizer, config) for prompt in prompts
    ]
    predictions: list[str] = []
    batch_size = config.generation.per_device_batch_size
    for start in range(0, len(formatted_prompts), batch_size):
        batch = formatted_prompts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=config.generation.max_input_length,
            return_tensors="pt",
        )
        encoded = {
            key: value.to(device) for key, value in encoded.items()
        }
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": config.generation.max_new_tokens,
            "do_sample": config.generation.do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if config.generation.do_sample:
            generation_kwargs["temperature"] = config.generation.temperature
        with torch.inference_mode():
            generated = model.generate(**encoded, **generation_kwargs)
        prompt_width = encoded["input_ids"].shape[1]
        new_tokens = generated[:, prompt_width:]
        predictions.extend(
            tokenizer.batch_decode(
                new_tokens,
                skip_special_tokens=True,
            )
        )

    return predictions


def _select_preference_records(
    dataset: Any,
    config: EvalConfig,
    *,
    limit: int,
) -> list[CanonicalPreferenceRecord]:
    preference_config = _preference_config(config)
    selected = dataset.shuffle(seed=config.generation.seed).select(
        range(min(limit, len(dataset)))
    )
    return [
        convert_preference_row(
            row,
            row_index=row_index,
            config=preference_config,
        )
        for row_index, row in enumerate(selected)
    ]


def _preference_texts(
    record: CanonicalPreferenceRecord,
    tokenizer: Any,
    config: EvalConfig,
) -> tuple[str, str, str]:
    if isinstance(record.prompt, str):
        if not isinstance(record.chosen, str) or not isinstance(
            record.rejected, str
        ):
            raise ValueError("Preference record formats must match")
        prompt_text = _format_prompt(record.prompt, tokenizer, config)
        if config.formatting.mode == "plain":
            return (
                prompt_text,
                f"{prompt_text}{record.chosen}",
                f"{prompt_text}{record.rejected}",
            )
        messages: list[dict[str, str]] = []
        if config.formatting.system_prompt is not None:
            messages.append(
                {
                    "role": "system",
                    "content": config.formatting.system_prompt,
                }
            )
        messages.append({"role": "user", "content": record.prompt})
        chosen_messages = messages + [
            {"role": "assistant", "content": record.chosen}
        ]
        rejected_messages = messages + [
            {"role": "assistant", "content": record.rejected}
        ]
    else:
        if not isinstance(record.chosen, list) or not isinstance(
            record.rejected, list
        ):
            raise ValueError("Preference record formats must match")
        messages = list(record.prompt)
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except (AttributeError, ValueError, TypeError) as exc:
            raise ValueError(
                "Conversational preference evaluation requires a chat template"
            ) from exc
        chosen_messages = messages + list(record.chosen)
        rejected_messages = messages + list(record.rejected)

    try:
        chosen_text = tokenizer.apply_chat_template(
            chosen_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        rejected_text = tokenizer.apply_chat_template(
            rejected_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except (AttributeError, ValueError, TypeError) as exc:
        raise ValueError(
            "Preference evaluation requires a usable chat template"
        ) from exc
    return prompt_text, chosen_text, rejected_text


def _conditional_log_probability(
    model: Any,
    tokenizer: Any,
    *,
    prompt_text: str,
    full_text: str,
    max_length: int,
    device: Any,
) -> float:
    import torch

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )["input_ids"]
    full_ids = tokenizer(
        full_text,
        add_special_tokens=False,
    )["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Tokenizer chat template did not preserve the prompt prefix"
        )
    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
    completion_start = len(prompt_ids)
    if completion_start >= len(full_ids):
        raise ValueError(
            "Preference example has no scoreable completion tokens within "
            "the evaluation length limit"
        )

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits
    token_log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]
    gathered = token_log_probs.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)
    first_target = max(completion_start - 1, 0)
    return float(gathered[:, first_target:].sum().item())


def _score_preferences(
    loaded: LoadedEvaluationModel,
    config: EvalConfig,
    records: list[CanonicalPreferenceRecord],
) -> float:
    model = loaded.model
    tokenizer = loaded.tokenizer
    device = loaded.device

    max_length = (
        config.generation.max_input_length
        + config.generation.max_new_tokens
    )
    wins = 0.0
    for record in records:
        prompt_text, chosen_text, rejected_text = _preference_texts(
            record,
            tokenizer,
            config,
        )
        chosen_score = _conditional_log_probability(
            model,
            tokenizer,
            prompt_text=prompt_text,
            full_text=chosen_text,
            max_length=max_length,
            device=device,
        )
        rejected_score = _conditional_log_probability(
            model,
            tokenizer,
            prompt_text=prompt_text,
            full_text=rejected_text,
            max_length=max_length,
            device=device,
        )
        if chosen_score > rejected_score:
            wins += 1.0
        elif chosen_score == rejected_score:
            wins += 0.5

    return wins / len(records)


def _write_predictions(
    path: Path,
    *,
    ids: list[str],
    prompts: list[str],
    references: list[str],
    predictions: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for values in zip(
            ids,
            prompts,
            references,
            predictions,
            strict=True,
        ):
            record_id, prompt, reference, prediction = values
            handle.write(
                json.dumps(
                    {
                        "id": record_id,
                        "prompt": prompt,
                        "reference": reference,
                        "prediction": prediction,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _build_comparisons(
    results: list[ModelEvaluationResult],
) -> list[ModelComparison]:
    if len(results) < 2:
        return []
    baseline = results[0]
    comparisons: list[ModelComparison] = []
    for result in results[1:]:
        comparisons.append(
            ModelComparison(
                baseline=baseline.name,
                model=result.name,
                exact_match_delta=(
                    result.metrics.exact_match
                    - baseline.metrics.exact_match
                ),
                normalized_exact_match_delta=(
                    result.metrics.normalized_exact_match
                    - baseline.metrics.normalized_exact_match
                ),
                token_f1_delta=(
                    result.metrics.token_f1 - baseline.metrics.token_f1
                ),
                non_empty_rate_delta=(
                    result.metrics.non_empty_rate
                    - baseline.metrics.non_empty_rate
                ),
                preference_accuracy_delta=(
                    result.preference_accuracy
                    - baseline.preference_accuracy
                ),
            )
        )
    return comparisons


def run_evaluation(
    config_path: str,
    *,
    limit: int | None = None,
    dataset_loader: Any = None,
) -> EvaluationResult:
    config = load_eval_config(config_path)
    resolved_limit = config.generation.limit if limit is None else limit
    if resolved_limit <= 0:
        raise ValueError("Evaluation limit must be positive")

    dataset = _load_eval_dataset(config, dataset_loader)
    _validate_eval_dataset(dataset, config, limit=resolved_limit)
    ids, prompts, references = _select_eval_rows(
        dataset,
        config,
        limit=resolved_limit,
    )
    preference_config = _preference_config(config)
    preference_dataset = load_preference_dataset(
        preference_config,
        loader=dataset_loader,
    )
    validate_preference_dataset(
        preference_dataset,
        preference_config,
        limit=resolved_limit,
    )
    preference_records = _select_preference_records(
        preference_dataset,
        config,
        limit=resolved_limit,
    )
    if not preference_records:
        raise ValueError(
            "Preference evaluation dataset must contain at least one row"
        )

    results: list[ModelEvaluationResult] = []
    predictions_dir = Path(config.output.predictions_dir)
    for model_config in config.models:
        loaded = _load_evaluation_model(model_config)
        try:
            predictions = _generate_predictions(
                loaded,
                config,
                prompts,
            )
            preference_accuracy = _score_preferences(
                loaded,
                config,
                preference_records,
            )
        finally:
            del loaded
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        predictions_path = predictions_dir / f"{model_config.name}.jsonl"
        _write_predictions(
            predictions_path,
            ids=ids,
            prompts=prompts,
            references=references,
            predictions=predictions,
        )
        results.append(
            ModelEvaluationResult(
                name=model_config.name,
                model_name_or_path=model_config.model_name_or_path,
                predictions_path=str(predictions_path),
                metrics=score_predictions(predictions, references),
                preference_pairs=len(preference_records),
                preference_accuracy=preference_accuracy,
            )
        )

    evaluation = EvaluationResult(
        dataset_id=config.data.dataset_id,
        dataset_split=config.data.dataset_split,
        examples=len(prompts),
        preference_pairs=len(preference_records),
        models=results,
        comparisons=_build_comparisons(results),
        results_path=config.output.results_path,
    )
    results_path = Path(config.output.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(asdict(evaluation), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evaluation
