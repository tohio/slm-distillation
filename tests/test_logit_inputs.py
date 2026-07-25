from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from distill.training.train_logit_distill import (
    compute_logit_distillation_loss,
    validate_logit_inputs,
)


class FakeDataset:
    column_names = ["id", "prompt", "response", "metadata"]

    def __init__(self) -> None:
        self.rows = [
            {
                "id": "one",
                "prompt": "Question",
                "response": "Answer",
                "metadata": {},
            }
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class FakeTokenizer:
    special_tokens_map = {"eos_token": "<eos>"}

    def get_vocab(self):
        return {"<eos>": 0, "hello": 1}


def test_validate_logit_inputs_resolves_models_tokenizers_and_dataset() -> None:
    model_calls = []
    tokenizer_calls = []

    def model_loader(**kwargs):
        model_calls.append(kwargs)
        return SimpleNamespace(id=kwargs["repo_id"], sha="abc123")

    def tokenizer_loader(reference, **kwargs):
        tokenizer_calls.append((reference, kwargs))
        return FakeTokenizer()

    result = validate_logit_inputs(
        "configs/logit_distill.yaml",
        limit=1,
        model_info_loader=model_loader,
        tokenizer_loader=tokenizer_loader,
        dataset_loader=lambda **_: FakeDataset(),
    )

    assert result.teacher.resolved_id == (
        "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    )
    assert result.student.resolved_id == (
        "HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    assert result.tokenizer_compatibility.compatible is True
    assert result.dataset.inspected_rows == 1
    assert len(model_calls) == 2
    assert len(tokenizer_calls) == 2


def test_logit_loss_uses_only_supervised_response_tokens() -> None:
    student_logits = torch.tensor(
        [[[4.0, 0.0], [0.0, 4.0], [4.0, 0.0]]]
    )
    teacher_logits = torch.tensor(
        [[[0.0, 4.0], [0.0, 4.0], [0.0, 4.0]]]
    )
    labels = torch.tensor([[-100, -100, 1]])
    hard_loss = torch.tensor(2.0)

    loss = compute_logit_distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        labels=labels,
        hard_loss=hard_loss,
        temperature=1.0,
        alpha=0.5,
    )

    assert loss.item() == pytest.approx(1.0, abs=0.05)


def test_logit_loss_rejects_batch_without_response_tokens() -> None:
    logits = torch.zeros((1, 2, 3))

    with pytest.raises(ValueError, match="no supervised tokens"):
        compute_logit_distillation_loss(
            student_logits=logits,
            teacher_logits=logits,
            labels=torch.full((1, 2), -100),
            hard_loss=torch.tensor(0.0),
            temperature=2.0,
            alpha=0.5,
        )
