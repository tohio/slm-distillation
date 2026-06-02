from __future__ import annotations

import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from time import monotonic
from typing import Any


class RetryableProviderExhaustedError(RuntimeError):
    def __init__(self, message: str, *, telemetry: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.telemetry = telemetry or {}


@dataclass(frozen=True)
class HostedGenerationControls:
    concurrency: int = 1
    max_requeues: int = 3
    exhausted_retryable_requeue_delay_seconds: float = 60.0
    max_request_retries: int = 3
    max_retryable_request_attempts: int = 20
    retry_max_elapsed_seconds: float = 1800.0
    retry_sleep_seconds: float = 0.5
    retry_backoff_initial_seconds: float = 1.0
    retry_backoff_max_seconds: float = 30.0
    retry_backoff_multiplier: float = 2.0
    retry_jitter_ratio: float = 0.30
    adaptive_concurrency_enabled: bool = True
    adaptive_initial_in_flight: int = 1
    adaptive_minimum_in_flight: int = 1
    adaptive_rate_limit_decrease_factor: float = 0.50
    adaptive_cooldown_initial_seconds: float = 5.0
    adaptive_cooldown_max_seconds: float = 60.0
    adaptive_cooldown_multiplier: float = 2.0


class AdaptiveRequestController:
    def __init__(self, controls: HostedGenerationControls) -> None:
        self.enabled = controls.adaptive_concurrency_enabled
        self.maximum_in_flight = max(1, int(controls.concurrency))
        self.minimum_in_flight = min(
            self.maximum_in_flight,
            max(1, int(controls.adaptive_minimum_in_flight)),
        )
        self.current_limit = min(
            self.maximum_in_flight,
            max(self.minimum_in_flight, int(controls.adaptive_initial_in_flight)),
        )
        self.rate_limit_decrease_factor = min(
            1.0,
            max(0.01, float(controls.adaptive_rate_limit_decrease_factor)),
        )
        self.cooldown_initial_seconds = max(0.0, float(controls.adaptive_cooldown_initial_seconds))
        self.cooldown_max_seconds = max(
            self.cooldown_initial_seconds,
            float(controls.adaptive_cooldown_max_seconds),
        )
        self.cooldown_multiplier = max(1.0, float(controls.adaptive_cooldown_multiplier))
        self._condition = threading.Condition()
        self._active_requests = 0
        self._cooldown_until = 0.0
        self._next_cooldown_seconds = self.cooldown_initial_seconds
        self._generation = 0
        self._peak_limit = self.current_limit
        self._minimum_observed_limit = self.current_limit
        self._rate_limit_events: deque[float] = deque()

    def acquire(self) -> tuple[float, int]:
        started = monotonic()
        if not self.enabled:
            return 0.0, self._generation

        with self._condition:
            while True:
                now = monotonic()
                cooldown_remaining = self._cooldown_until - now
                if cooldown_remaining > 0:
                    self._condition.wait(timeout=cooldown_remaining)
                    continue

                if self._active_requests < self.current_limit:
                    self._active_requests += 1
                    return max(0.0, monotonic() - started), self._generation

                self._condition.wait()

    def release(self) -> None:
        if not self.enabled:
            return

        with self._condition:
            if self._active_requests <= 0:
                raise RuntimeError("adaptive controller released without acquired slot")
            self._active_requests -= 1
            self._condition.notify_all()

    def record_success(self, generation: int | None = None) -> None:
        if not self.enabled:
            return

        with self._condition:
            if generation is not None and generation != self._generation:
                return
            if monotonic() < self._cooldown_until:
                return
            if self.current_limit < self.maximum_in_flight:
                self.current_limit += 1
                self._peak_limit = max(self._peak_limit, self.current_limit)
                self._condition.notify_all()

    def record_rate_limit(self, generation: int | None = None) -> float:
        if not self.enabled:
            return 0.0

        with self._condition:
            if generation is not None and generation != self._generation:
                return 0.0

            previous = self.current_limit
            reduced = max(
                self.minimum_in_flight,
                int(previous * self.rate_limit_decrease_factor),
            )
            if reduced >= previous and previous > self.minimum_in_flight:
                reduced = previous - 1

            self.current_limit = reduced
            self._minimum_observed_limit = min(self._minimum_observed_limit, reduced)
            self._generation += 1

            cooldown = min(self._next_cooldown_seconds, self.cooldown_max_seconds)
            self._cooldown_until = monotonic() + cooldown
            self._next_cooldown_seconds = min(
                cooldown * self.cooldown_multiplier,
                self.cooldown_max_seconds,
            )
            self._rate_limit_events.append(monotonic())
            self._condition.notify_all()
            return cooldown

    def snapshot(self) -> dict[str, int]:
        return {
            "adaptive_current_in_flight_limit": self.current_limit,
            "adaptive_peak_in_flight_limit": self._peak_limit,
            "adaptive_min_in_flight_limit": self._minimum_observed_limit,
        }


def hosted_controls_from_mapping(data: dict[str, Any] | None) -> HostedGenerationControls:
    raw = data or {}

    def get_int(key: str, default: int) -> int:
        return int(raw.get(key, default))

    def get_float(key: str, default: float) -> float:
        return float(raw.get(key, default))

    def get_bool(key: str, default: bool) -> bool:
        return bool(raw.get(key, default))

    return HostedGenerationControls(
        concurrency=get_int("concurrency", 1),
        max_requeues=get_int("max_requeues", 3),
        exhausted_retryable_requeue_delay_seconds=get_float(
            "exhausted_retryable_requeue_delay_seconds",
            60.0,
        ),
        max_request_retries=get_int("max_request_retries", 3),
        max_retryable_request_attempts=get_int("max_retryable_request_attempts", 20),
        retry_max_elapsed_seconds=get_float("retry_max_elapsed_seconds", 1800.0),
        retry_sleep_seconds=get_float("retry_sleep_seconds", 0.5),
        retry_backoff_initial_seconds=get_float("retry_backoff_initial_seconds", 1.0),
        retry_backoff_max_seconds=get_float("retry_backoff_max_seconds", 30.0),
        retry_backoff_multiplier=get_float("retry_backoff_multiplier", 2.0),
        retry_jitter_ratio=get_float("retry_jitter_ratio", 0.30),
        adaptive_concurrency_enabled=get_bool("adaptive_concurrency_enabled", True),
        adaptive_initial_in_flight=get_int("adaptive_initial_in_flight", 1),
        adaptive_minimum_in_flight=get_int("adaptive_minimum_in_flight", 1),
        adaptive_rate_limit_decrease_factor=get_float(
            "adaptive_rate_limit_decrease_factor",
            0.50,
        ),
        adaptive_cooldown_initial_seconds=get_float(
            "adaptive_cooldown_initial_seconds",
            5.0,
        ),
        adaptive_cooldown_max_seconds=get_float("adaptive_cooldown_max_seconds", 60.0),
        adaptive_cooldown_multiplier=get_float("adaptive_cooldown_multiplier", 2.0),
    )


def is_capacity_or_rate_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "capacity_exceeded",
            "rate_limit",
            "rate limit",
            "too many requests",
            "error code: 429",
            "error code: 498",
            " 429 ",
            " 498 ",
        )
    )


def is_transient_transport_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "timeout",
            "timed out",
            "temporarily unavailable",
            "connection error",
            "connection reset",
            "remoteprotocolerror",
            "peer closed connection",
            "incomplete chunked read",
            "error code: 500",
            "error code: 502",
            "error code: 503",
            "error code: 504",
            " 500 ",
            " 502 ",
            " 503 ",
            " 504 ",
        )
    )


def is_retryable_provider_error(exc: Exception) -> bool:
    return is_capacity_or_rate_error(exc) or is_transient_transport_error(exc)


def retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        return None

    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(value))
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def can_retry_provider_call(
    *,
    attempt: int,
    started: float,
    exc: Exception,
    controls: HostedGenerationControls,
) -> bool:
    if is_retryable_provider_error(exc):
        return (
            attempt < controls.max_retryable_request_attempts
            and monotonic() - started < controls.retry_max_elapsed_seconds
        )

    return attempt < controls.max_request_retries


def retry_delay_seconds(
    *,
    attempt: int,
    started: float,
    exc: Exception,
    controls: HostedGenerationControls,
) -> float:
    if is_retryable_provider_error(exc):
        retry_after = retry_after_seconds(exc)
        if retry_after is not None:
            delay = retry_after
        else:
            base_delay = min(
                controls.retry_backoff_max_seconds,
                controls.retry_backoff_initial_seconds
                * (controls.retry_backoff_multiplier ** max(0, attempt - 1)),
            )
            delay = base_delay + base_delay * max(0.0, controls.retry_jitter_ratio) * random.random()

        remaining = max(0.0, controls.retry_max_elapsed_seconds - (monotonic() - started))
        return min(delay, remaining)

    return controls.retry_sleep_seconds * attempt


def generate_with_hosted_retries(
    *,
    provider: Any,
    request: Any,
    controls: HostedGenerationControls,
    controller: AdaptiveRequestController,
) -> Any:
    started: float | None = None
    attempt = 0
    retry_sleep_total = 0.0
    last_error: Exception | None = None

    while True:
        attempt += 1
        admission_wait, generation = controller.acquire()
        if started is None:
            started = monotonic()

        try:
            response = provider.generate(request)
        except Exception as exc:
            last_error = exc
            if is_capacity_or_rate_error(exc):
                controller.record_rate_limit(generation)
        else:
            controller.record_success(generation)
            return response
        finally:
            controller.release()

        assert started is not None
        if not can_retry_provider_call(
            attempt=attempt,
            started=started,
            exc=last_error,
            controls=controls,
        ):
            break

        delay = retry_delay_seconds(
            attempt=attempt,
            started=started,
            exc=last_error,
            controls=controls,
        )
        retry_sleep_total += delay
        if delay > 0:
            time.sleep(delay)

    telemetry = {
        "retry_count": max(0, attempt - 1),
        "retry_sleep_seconds": round(retry_sleep_total, 3),
        "adaptive_admission_wait_seconds": round(admission_wait, 3),
        **controller.snapshot(),
    }
    raise RetryableProviderExhaustedError(
        f"Hosted provider request failed after {attempt} attempts: {last_error}",
        telemetry=telemetry,
    ) from last_error
