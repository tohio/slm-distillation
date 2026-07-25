from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationMetrics:
    examples: int
    exact_match: float
    normalized_exact_match: float
    token_f1: float
    non_empty_rate: float
    average_output_characters: float


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    return re.sub(r"\s+", " ", normalized)


def _token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_text(prediction).split()
    reference_tokens = normalize_text(reference).split()
    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0

    overlap = sum(
        (Counter(prediction_tokens) & Counter(reference_tokens)).values()
    )
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def score_predictions(
    predictions: list[str],
    references: list[str],
) -> EvaluationMetrics:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have equal length")
    if not predictions:
        raise ValueError("Evaluation requires at least one prediction")

    count = len(predictions)
    exact = sum(
        prediction.strip() == reference.strip()
        for prediction, reference in zip(predictions, references, strict=True)
    )
    normalized_exact = sum(
        normalize_text(prediction) == normalize_text(reference)
        for prediction, reference in zip(predictions, references, strict=True)
    )
    f1 = sum(
        _token_f1(prediction, reference)
        for prediction, reference in zip(predictions, references, strict=True)
    )
    non_empty = sum(bool(prediction.strip()) for prediction in predictions)
    characters = sum(len(prediction) for prediction in predictions)

    return EvaluationMetrics(
        examples=count,
        exact_match=exact / count,
        normalized_exact_match=normalized_exact / count,
        token_f1=f1 / count,
        non_empty_rate=non_empty / count,
        average_output_characters=characters / count,
    )
