from __future__ import annotations

from typing import Any

import pytest

from distill.data.preference import (
    convert_preference_row,
    load_preference_dataset,
    validate_preference_dataset,
)
from distill.utils.config import DpoDataConfig


class FakeDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.column_names = list(rows[0]) if rows else []

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def preference_config(**overrides: Any) -> DpoDataConfig:
    values = {
        "dataset_id": "example/preference",
        "dataset_config_name": None,
        "dataset_split": "train",
        "id_field": "id",
        "prompt_field": "prompt",
        "chosen_field": "chosen",
        "rejected_field": "rejected",
        "metadata_field": "metadata",
    }
    values.update(overrides)
    return DpoDataConfig(**values)


def conversational_row(
    *,
    record_id: str = "one",
    chosen: str = "Correct",
    rejected: str = "Incorrect",
) -> dict[str, Any]:
    return {
        "id": record_id,
        "prompt": [{"role": "user", "content": "Question"}],
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
        "metadata": {"category": "test"},
    }


def test_convert_preference_row_supports_configured_standard_fields() -> None:
    config = preference_config(
        id_field=None,
        prompt_field="instruction",
        chosen_field="accepted",
        rejected_field="declined",
        metadata_field=None,
    )
    record = convert_preference_row(
        {
            "instruction": "Question",
            "accepted": "Correct",
            "declined": "Incorrect",
        },
        row_index=3,
        config=config,
    )

    assert record.id == "train-00000003"
    assert record.prompt == "Question"
    assert record.chosen == "Correct"
    assert record.rejected == "Incorrect"
    assert record.metadata == {}


def test_validate_preference_dataset_accepts_conversational_rows() -> None:
    dataset = FakeDataset(
        [
            conversational_row(record_id="one"),
            conversational_row(record_id="two"),
        ]
    )

    summary = validate_preference_dataset(
        dataset,
        preference_config(),
        limit=1,
    )

    assert summary.row_count == 2
    assert summary.inspected_rows == 1
    assert summary.unique_ids == 1
    assert summary.format == "conversational"
    assert summary.column_names == [
        "chosen",
        "id",
        "metadata",
        "prompt",
        "rejected",
    ]


def test_load_preference_dataset_passes_hf_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = FakeDataset([conversational_row()])
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        "distill.data.preference.get_env_value",
        lambda *args, **kwargs: "test-token",
    )

    def load_dataset(**kwargs: Any) -> FakeDataset:
        calls.append(kwargs)
        return dataset

    load_preference_dataset(preference_config(), loader=load_dataset)

    assert calls[0]["token"] == "test-token"


def test_validate_preference_dataset_rejects_identical_pair() -> None:
    dataset = FakeDataset(
        [conversational_row(chosen="Same", rejected="Same")]
    )

    with pytest.raises(ValueError, match="identical chosen and rejected"):
        validate_preference_dataset(dataset, preference_config())


def test_validate_preference_dataset_rejects_duplicate_ids() -> None:
    dataset = FakeDataset(
        [
            conversational_row(record_id="duplicate"),
            conversational_row(record_id="duplicate"),
        ]
    )

    with pytest.raises(ValueError, match="duplicate id"):
        validate_preference_dataset(dataset, preference_config())


def test_convert_preference_row_rejects_mixed_formats() -> None:
    row = conversational_row()
    row["prompt"] = "Question"

    with pytest.raises(ValueError, match="mixes standard and conversational"):
        convert_preference_row(
            row,
            row_index=0,
            config=preference_config(),
        )
