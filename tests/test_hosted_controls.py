from __future__ import annotations

import pytest

from distill.generation.hosted_controls import (
    HostedGenerationControls,
    RetryableProviderExhaustedError,
    hosted_controls_from_mapping,
    is_retryable_provider_error,
)


def test_hosted_controls_from_mapping_reads_values() -> None:
    controls = hosted_controls_from_mapping(
        {
            "concurrency": 4,
            "max_requeues": 2,
            "retry_backoff_max_seconds": 12.5,
            "adaptive_initial_in_flight": 2,
        }
    )

    assert controls.concurrency == 4
    assert controls.max_requeues == 2
    assert controls.retry_backoff_max_seconds == 12.5
    assert controls.adaptive_initial_in_flight == 2


def test_retryable_provider_error_detection() -> None:
    assert is_retryable_provider_error(RuntimeError("error code: 429"))
    assert is_retryable_provider_error(RuntimeError("connection reset"))
    assert not is_retryable_provider_error(RuntimeError("invalid request"))


def test_retryable_provider_exhausted_error_has_telemetry() -> None:
    error = RetryableProviderExhaustedError(
        "failed",
        telemetry={"retry_count": 3},
    )

    assert error.telemetry["retry_count"] == 3
