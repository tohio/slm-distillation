from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    provider: str
    model: str
    input_price_per_1m: float
    output_price_per_1m: float


@dataclass(frozen=True)
class CostEstimate:
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float


def estimate_cost(
    pricing: ModelPricing,
    input_tokens: int,
    output_tokens: int,
) -> CostEstimate:
    if input_tokens < 0:
        raise ValueError("input_tokens must be >= 0")

    if output_tokens < 0:
        raise ValueError("output_tokens must be >= 0")

    input_cost = (input_tokens / 1_000_000) * pricing.input_price_per_1m
    output_cost = (output_tokens / 1_000_000) * pricing.output_price_per_1m

    return CostEstimate(
        provider=pricing.provider,
        model=pricing.model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        total_cost_usd=input_cost + output_cost,
    )


def cost_per_accepted_sample(total_cost_usd: float, accepted_samples: int) -> float:
    if accepted_samples <= 0:
        raise ValueError("accepted_samples must be > 0")

    return total_cost_usd / accepted_samples
