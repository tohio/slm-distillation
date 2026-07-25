from __future__ import annotations

from typing import Any

import pytest

from distill.data.response import convert_response_row, inspect_response_dataset
from distill.utils.config import ResponseDataConfig


class FakeDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.column_names = list(rows[0]) if rows else []

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def response_config(**overrides: Any) -> ResponseDataConfig:
    values = {
        "dataset_id": "example/response",
        "dataset_config_name": None,
        "dataset_split": "train",
        "id_field": "id",
        "prompt_field": "prompt",
        "response_field": "response",
        "metadata_field": "metadata",
    }
    values.update(overrides)
    return ResponseDataConfig(**values)


def test_convert_response_row_uses_configured_fields() -> None:
    config = response_config(
        id_field=None,
        prompt_field="instruction",
        response_field="answer",
        metadata_field=None,
    )
    record = convert_response_row(
        {"instruction": "Add two numbers.", "answer": "Use +."},
        row_index=3,
        config=config,
    )

    assert record.id == "train-00000003"
    assert record.prompt == "Add two numbers."
    assert record.response == "Use +."
    assert record.metadata == {}


def test_inspect_response_dataset_validates_and_summarizes_rows() -> None:
    dataset = FakeDataset(
        [
            {
                "id": "one",
                "prompt": "Question one",
                "response": "Answer one",
                "metadata": {"category": "test"},
            },
            {
                "id": "two",
                "prompt": "Question two",
                "response": "Answer two",
                "metadata": {"category": "test"},
            },
        ]
    )

    summary = inspect_response_dataset(
        response_config(),
        limit=1,
        loader=lambda **_: dataset,
    )

    assert summary.row_count == 2
    assert summary.inspected_rows == 1
    assert summary.unique_ids == 1
    assert summary.column_names == ["id", "metadata", "prompt", "response"]


def test_inspect_response_dataset_rejects_missing_configured_column() -> None:
    dataset = FakeDataset(
        [{"id": "one", "prompt": "Question", "response": "Answer"}]
    )

    with pytest.raises(ValueError, match="metadata"):
        inspect_response_dataset(
            response_config(),
            loader=lambda **_: dataset,
        )


def test_inspect_response_dataset_rejects_duplicate_ids() -> None:
    dataset = FakeDataset(
        [
            {
                "id": "duplicate",
                "prompt": "Question one",
                "response": "Answer one",
                "metadata": {},
            },
            {
                "id": "duplicate",
                "prompt": "Question two",
                "response": "Answer two",
                "metadata": {},
            },
        ]
    )

    with pytest.raises(ValueError, match="duplicate id"):
        inspect_response_dataset(
            response_config(),
            loader=lambda **_: dataset,
        )
