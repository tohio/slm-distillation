from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from distill.eval.compare_outputs import normalize_text, score_predictions
from distill.eval.run_eval import (
    _load_eval_dataset,
    build_evaluation_plan,
    run_evaluation,
)
from distill.utils.config import load_eval_config


class FakeDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.column_names = list(rows[0]) if rows else []

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]

    def __iter__(self):
        return iter(self.rows)

    def shuffle(self, *, seed: int):
        assert isinstance(seed, int)
        return self

    def select(self, indices):
        return FakeDataset([self.rows[index] for index in indices])


def test_load_eval_config_and_build_plan() -> None:
    config = load_eval_config("configs/eval.yaml")
    plan = build_evaluation_plan(config)

    assert plan.model_names == ["base", "response_distilled", "response_dpo"]
    assert plan.dataset_id == "tohio/slm-synthetic-distillation-sft"
    assert (
        plan.preference_dataset_id
        == "tohio/slm-synthetic-distillation-dpo"
    )
    assert plan.limit == 200
    assert plan.results_path.endswith("eval/results.json")


def test_logit_eval_config_uses_logit_branch_checkpoints() -> None:
    config = load_eval_config("configs/eval_logit.yaml")

    assert [model.name for model in config.models] == [
        "base",
        "logit_distilled",
        "logit_dpo",
    ]
    assert config.models[-1].model_name_or_path == (
        "runs/smollm2-135m-logit-distilled/dpo/checkpoints/final"
    )


def test_load_eval_dataset_passes_hf_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_eval_config("configs/eval.yaml")
    dataset = FakeDataset(
        [{"id": "one", "prompt": "Question", "response": "Answer"}]
    )
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        "distill.eval.run_eval.get_env_value",
        lambda *args, **kwargs: "test-token",
    )

    def load_dataset(**kwargs: Any) -> FakeDataset:
        calls.append(kwargs)
        return dataset

    _load_eval_dataset(config, loader=load_dataset)

    assert calls[0]["token"] == "test-token"


def test_score_predictions_reports_exact_and_normalized_metrics() -> None:
    metrics = score_predictions(
        ["  FOUR ", "blue sky"],
        ["four", "blue"],
    )

    assert normalize_text("  FOUR\n") == "four"
    assert metrics.exact_match == 0.0
    assert metrics.normalized_exact_match == 0.5
    assert metrics.token_f1 == pytest.approx((1.0 + 2 / 3) / 2)
    assert metrics.non_empty_rate == 1.0


def test_run_evaluation_writes_predictions_and_comparisons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_text = Path("configs/eval.yaml").read_text(encoding="utf-8")
    results_path = tmp_path / "results.json"
    predictions_dir = tmp_path / "predictions"
    config_text = config_text.replace(
        "runs/smollm2-135m-deepseek-distilled/eval/results.json",
        str(results_path),
    ).replace(
        "runs/smollm2-135m-deepseek-distilled/eval/predictions",
        str(predictions_dir),
    )
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    dataset = FakeDataset(
        [
            {"id": "one", "prompt": "2 + 2", "response": "4"},
            {"id": "two", "prompt": "Sky color", "response": "Blue"},
        ]
    )
    preference_dataset = FakeDataset(
        [
            {
                "id": "preference-one",
                "prompt": "2 + 2",
                "chosen": "4",
                "rejected": "5",
            },
            {
                "id": "preference-two",
                "prompt": "Sky color",
                "chosen": "Blue",
                "rejected": "Green",
            },
        ]
    )
    generated = {
        "base": ["5", "Green"],
        "response_distilled": ["4", "Blue"],
        "response_dpo": ["4", "Blue"],
    }

    monkeypatch.setattr(
        "distill.eval.run_eval._generate_predictions",
        lambda model_config, config, prompts: generated[model_config.name],
    )
    preference_accuracy = {
        "base": 0.5,
        "response_distilled": 1.0,
        "response_dpo": 1.0,
    }
    monkeypatch.setattr(
        "distill.eval.run_eval._score_preferences",
        lambda model_config, config, records: preference_accuracy[
            model_config.name
        ],
    )

    def load_dataset(**kwargs):
        if kwargs["path"].endswith("-dpo"):
            return preference_dataset
        return dataset

    result = run_evaluation(
        str(config_path),
        limit=2,
        dataset_loader=load_dataset,
    )

    assert result.examples == 2
    assert result.models[0].metrics.normalized_exact_match == 0.0
    assert result.models[1].metrics.normalized_exact_match == 1.0
    assert result.models[1].preference_accuracy == 1.0
    assert result.comparisons[0].normalized_exact_match_delta == 1.0
    assert result.comparisons[0].preference_accuracy_delta == 0.5
    assert results_path.exists()
    assert len(list(predictions_dir.glob("*.jsonl"))) == 3
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["models"][2]["name"] == "response_dpo"
