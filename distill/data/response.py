from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from distill.utils.config import ResponseDataConfig
from distill.utils.env import get_env_value


@dataclass(frozen=True)
class CanonicalResponseRecord:
    id: str
    prompt: str
    response: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ResponseDatasetSummary:
    dataset_id: str
    dataset_config_name: str | None
    dataset_split: str
    row_count: int
    inspected_rows: int
    column_names: list[str]
    unique_ids: int


DatasetLoader = Callable[..., Any]


def _required_columns(config: ResponseDataConfig) -> set[str]:
    columns = {config.prompt_field, config.response_field}
    if config.id_field is not None:
        columns.add(config.id_field)
    if config.metadata_field is not None:
        columns.add(config.metadata_field)
    return columns


def _validate_columns(column_names: list[str], config: ResponseDataConfig) -> None:
    missing = sorted(_required_columns(config) - set(column_names))
    if missing:
        raise ValueError(
            f"Response dataset is missing configured column(s): {', '.join(missing)}"
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
            f"Response dataset row {row_index} requires non-empty string '{field}'"
        )
    return value


def convert_response_row(
    row: Mapping[str, Any],
    *,
    row_index: int,
    config: ResponseDataConfig,
) -> CanonicalResponseRecord:
    prompt = _require_non_empty_text(
        row,
        config.prompt_field,
        row_index=row_index,
    )
    response = _require_non_empty_text(
        row,
        config.response_field,
        row_index=row_index,
    )

    if config.id_field is None:
        record_id = f"{config.dataset_split}-{row_index:08d}"
    else:
        record_id = _require_non_empty_text(
            row,
            config.id_field,
            row_index=row_index,
        )

    metadata: dict[str, Any] = {}
    if config.metadata_field is not None:
        raw_metadata = row.get(config.metadata_field)
        if not isinstance(raw_metadata, Mapping):
            raise ValueError(
                f"Response dataset row {row_index} requires mapping "
                f"'{config.metadata_field}'"
            )
        metadata = dict(raw_metadata)

    return CanonicalResponseRecord(
        id=record_id,
        prompt=prompt,
        response=response,
        metadata=metadata,
    )


def load_response_dataset(
    config: ResponseDataConfig,
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


def validate_response_dataset(
    dataset: Any,
    config: ResponseDataConfig,
    *,
    limit: int | None = None,
) -> ResponseDatasetSummary:
    if limit is not None and limit <= 0:
        raise ValueError("Response dataset inspection limit must be positive")

    column_names = list(dataset.column_names)
    _validate_columns(column_names, config)

    row_count = len(dataset)
    inspected_rows = row_count if limit is None else min(limit, row_count)
    unique_ids: set[str] = set()

    for row_index in range(inspected_rows):
        record = convert_response_row(
            dataset[row_index],
            row_index=row_index,
            config=config,
        )
        if record.id in unique_ids:
            raise ValueError(
                f"Response dataset contains duplicate id in inspected rows: {record.id}"
            )
        unique_ids.add(record.id)

    return ResponseDatasetSummary(
        dataset_id=config.dataset_id,
        dataset_config_name=config.dataset_config_name,
        dataset_split=config.dataset_split,
        row_count=row_count,
        inspected_rows=inspected_rows,
        column_names=sorted(column_names),
        unique_ids=len(unique_ids),
    )


def inspect_response_dataset(
    config: ResponseDataConfig,
    *,
    limit: int | None = None,
    loader: DatasetLoader | None = None,
) -> ResponseDatasetSummary:
    dataset = load_response_dataset(config, loader=loader)
    return validate_response_dataset(dataset, config, limit=limit)
