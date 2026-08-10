from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from distill.utils.config import DpoDataConfig
from distill.utils.env import get_env_value


PreferenceValue = str | list[dict[str, str]]
DatasetLoader = Callable[..., Any]


@dataclass(frozen=True)
class CanonicalPreferenceRecord:
    id: str
    prompt: PreferenceValue
    chosen: PreferenceValue
    rejected: PreferenceValue
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PreferenceDatasetSummary:
    dataset_id: str
    dataset_config_name: str | None
    dataset_split: str
    row_count: int
    inspected_rows: int
    column_names: list[str]
    unique_ids: int
    format: str


def _required_columns(config: DpoDataConfig) -> set[str]:
    columns = {
        config.prompt_field,
        config.chosen_field,
        config.rejected_field,
    }
    if config.id_field is not None:
        columns.add(config.id_field)
    if config.metadata_field is not None:
        columns.add(config.metadata_field)
    return columns


def _validate_columns(column_names: list[str], config: DpoDataConfig) -> None:
    missing = sorted(_required_columns(config) - set(column_names))
    if missing:
        raise ValueError(
            f"Preference dataset is missing configured column(s): "
            f"{', '.join(missing)}"
        )


def _require_non_empty_text(
    row: Mapping[str, Any],
    field: str,
    *,
    row_index: int,
) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Preference dataset row {row_index} requires non-empty string "
            f"'{field}'"
        )
    return value


def _normalize_preference_value(
    value: Any,
    *,
    field: str,
    row_index: int,
) -> tuple[PreferenceValue, str]:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(
                f"Preference dataset row {row_index} requires non-empty "
                f"'{field}'"
            )
        return value, "standard"

    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise ValueError(
            f"Preference dataset row {row_index} field '{field}' must be "
            "non-empty text or a non-empty message list"
        )
    if not value:
        raise ValueError(
            f"Preference dataset row {row_index} requires non-empty message "
            f"list '{field}'"
        )

    messages: list[dict[str, str]] = []
    for message_index, message in enumerate(value):
        if not isinstance(message, Mapping):
            raise ValueError(
                f"Preference dataset row {row_index} field '{field}' message "
                f"{message_index} must be a mapping"
            )
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(
                f"Preference dataset row {row_index} field '{field}' message "
                f"{message_index} requires a non-empty role"
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"Preference dataset row {row_index} field '{field}' message "
                f"{message_index} requires non-empty text content"
            )
        messages.append({"role": role, "content": content})

    return messages, "conversational"


def convert_preference_row(
    row: Mapping[str, Any],
    *,
    row_index: int,
    config: DpoDataConfig,
) -> CanonicalPreferenceRecord:
    if config.id_field is None:
        record_id = f"{config.dataset_split}-{row_index:08d}"
    else:
        record_id = _require_non_empty_text(
            row,
            config.id_field,
            row_index=row_index,
        )

    prompt, prompt_format = _normalize_preference_value(
        row.get(config.prompt_field),
        field=config.prompt_field,
        row_index=row_index,
    )
    chosen, chosen_format = _normalize_preference_value(
        row.get(config.chosen_field),
        field=config.chosen_field,
        row_index=row_index,
    )
    rejected, rejected_format = _normalize_preference_value(
        row.get(config.rejected_field),
        field=config.rejected_field,
        row_index=row_index,
    )
    if len({prompt_format, chosen_format, rejected_format}) != 1:
        raise ValueError(
            f"Preference dataset row {row_index} mixes standard and "
            "conversational fields"
        )
    if chosen == rejected:
        raise ValueError(
            f"Preference dataset row {row_index} has identical chosen and "
            "rejected responses"
        )

    metadata: dict[str, Any] = {}
    if config.metadata_field is not None:
        raw_metadata = row.get(config.metadata_field)
        if not isinstance(raw_metadata, Mapping):
            raise ValueError(
                f"Preference dataset row {row_index} requires mapping "
                f"'{config.metadata_field}'"
            )
        metadata = dict(raw_metadata)

    return CanonicalPreferenceRecord(
        id=record_id,
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        metadata=metadata,
    )


def load_preference_dataset(
    config: DpoDataConfig,
    *,
    loader: DatasetLoader | None = None,
) -> Any:
    if loader is None:
        from datasets import load_dataset

        loader = load_dataset

    kwargs: dict[str, Any] = {
        "path": config.dataset_id,
        "split": config.dataset_split,
    }
    if config.dataset_config_name is not None:
        kwargs["name"] = config.dataset_config_name
    token = get_env_value("HF_TOKEN", fallback_to_os=True)
    if token is not None:
        kwargs["token"] = token

    return loader(**kwargs)


def validate_preference_dataset(
    dataset: Any,
    config: DpoDataConfig,
    *,
    limit: int | None = None,
) -> PreferenceDatasetSummary:
    if limit is not None and limit <= 0:
        raise ValueError("Preference dataset inspection limit must be positive")

    column_names = list(dataset.column_names)
    _validate_columns(column_names, config)

    row_count = len(dataset)
    inspected_rows = row_count if limit is None else min(limit, row_count)
    unique_ids: set[str] = set()
    dataset_format: str | None = None

    for row_index in range(inspected_rows):
        record = convert_preference_row(
            dataset[row_index],
            row_index=row_index,
            config=config,
        )
        if record.id in unique_ids:
            raise ValueError(
                "Preference dataset contains duplicate id in inspected rows: "
                f"{record.id}"
            )
        unique_ids.add(record.id)

        row_format = (
            "conversational" if isinstance(record.prompt, list) else "standard"
        )
        if dataset_format is None:
            dataset_format = row_format
        elif dataset_format != row_format:
            raise ValueError(
                "Preference dataset mixes standard and conversational rows"
            )

    return PreferenceDatasetSummary(
        dataset_id=config.dataset_id,
        dataset_config_name=config.dataset_config_name,
        dataset_split=config.dataset_split,
        row_count=row_count,
        inspected_rows=inspected_rows,
        column_names=sorted(column_names),
        unique_ids=len(unique_ids),
        format=dataset_format or "empty",
    )


def inspect_preference_dataset(
    config: DpoDataConfig,
    *,
    limit: int | None = None,
    loader: DatasetLoader | None = None,
) -> PreferenceDatasetSummary:
    dataset = load_preference_dataset(config, loader=loader)
    return validate_preference_dataset(dataset, config, limit=limit)
