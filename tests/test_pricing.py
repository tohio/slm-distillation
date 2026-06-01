import pytest

from distill.utils.pricing import (
    ModelPricing,
    cost_per_accepted_sample,
    estimate_cost,
)


def test_estimate_cost() -> None:
    pricing = ModelPricing(
        provider="openrouter",
        model="deepseek/deepseek-v4-flash",
        input_price_per_1m=0.0983,
        output_price_per_1m=0.1966,
    )

    estimate = estimate_cost(
        pricing=pricing,
        input_tokens=1_000_000,
        output_tokens=2_000_000,
    )

    assert estimate.provider == "openrouter"
    assert estimate.model == "deepseek/deepseek-v4-flash"
    assert estimate.input_cost_usd == pytest.approx(0.0983)
    assert estimate.output_cost_usd == pytest.approx(0.3932)
    assert estimate.total_cost_usd == pytest.approx(0.4915)


def test_estimate_cost_rejects_negative_tokens() -> None:
    pricing = ModelPricing(
        provider="openrouter",
        model="model",
        input_price_per_1m=1.0,
        output_price_per_1m=1.0,
    )

    with pytest.raises(ValueError, match="input_tokens"):
        estimate_cost(pricing, input_tokens=-1, output_tokens=0)

    with pytest.raises(ValueError, match="output_tokens"):
        estimate_cost(pricing, input_tokens=0, output_tokens=-1)


def test_cost_per_accepted_sample() -> None:
    assert cost_per_accepted_sample(10.0, 100) == pytest.approx(0.1)


def test_cost_per_accepted_sample_rejects_zero() -> None:
    with pytest.raises(ValueError, match="accepted_samples"):
        cost_per_accepted_sample(10.0, 0)
