"""The quality a development bet buys, across the whole range it can be made in.

The response was linear up to a hard ceiling of 0.10, so the simulator held that
spend beyond roughly 25k a week bought no quality whatsoever. Every large
development plan therefore forecast as pure cash burn and was refused on solvency
grounds it had not earned — the model could not discover a strategy the world
actually rewards, because our own forecast denied it existed.
"""

from __future__ import annotations

import pytest
from lithops.simulator.components import BaselineQualityDynamics
from lithops.simulator.models import SimulationAction, SimulationState


def state() -> SimulationState:
    return SimulationState(
        cash=1_000_000.0,
        revenue_weekly=250.0,
        customers=10.0,
        churn_rate=0.05,
        price_per_customer_weekly=25.0,
        weekly_acquisition=2.0,
        marketing_spend=1_000.0,
        development_spend=1_000.0,
        product_quality=0.30,
        capacity=50_000.0,
        reputation=0.5,
    )


def action() -> SimulationAction:
    return SimulationAction(
        name="candidate",
        price_per_customer_weekly=25.0,
        marketing_spend=1_000.0,
        development_spend=1_000.0,
    )


def improvement_at(spend: float) -> float:
    transition = BaselineQualityDynamics().transition(
        state=state(),
        action=action(),
        development_spend=spend,
        quality_lag_weeks=1,
    )
    return sum(effect.improvement for effect in transition.pending_effects)


class TestDevelopmentQualityResponse:
    def test_the_response_keeps_growing_across_the_whole_spend_range(self) -> None:
        spends = [1_000, 10_000, 50_000, 250_000, 1_000_000, 5_000_000]
        gains = [improvement_at(spend) for spend in spends]

        assert all(
            later > earlier for earlier, later in zip(gains, gains[1:], strict=False)
        ), f"a bigger bet must buy more quality, got {list(zip(spends, gains, strict=True))}"

    def test_no_spend_level_is_declared_worthless(self) -> None:
        # The ceiling made these two identical, which is what priced every large
        # development plan out of consideration.
        assert improvement_at(1_000_000) > improvement_at(250_000) * 1.05

    def test_the_shape_is_learned_not_assumed(self) -> None:
        # An exponent of 1.0 is constant returns; anything below it diminishes.
        # Both are reachable, because our own history could not tell them apart.
        def gain(exponent: float, spend: float) -> float:
            transition = BaselineQualityDynamics().transition(
                state=state(),
                action=action(),
                development_spend=spend,
                quality_lag_weeks=1,
                development_quality_response=0.03,
                development_quality_exponent=exponent,
            )
            return sum(effect.improvement for effect in transition.pending_effects)

        # Constant returns: ten times the spend buys ten times the quality.
        assert gain(1.0, 100_000.0) == pytest.approx(gain(1.0, 10_000.0) * 10.0)
        # Diminishing: the same tenfold bet buys less than tenfold.
        assert gain(0.55, 100_000.0) < gain(0.55, 10_000.0) * 10.0
        assert gain(0.55, 100_000.0) > gain(0.55, 10_000.0)

    def test_returns_diminish_rather_than_stop(self) -> None:
        first = improvement_at(20_000) - improvement_at(10_000)
        second = improvement_at(40_000) - improvement_at(30_000)
        assert second < first, "returns should flatten"
        assert second > 0.0, "but never reach zero"

    def test_a_learned_response_scales_the_whole_curve(self) -> None:
        modest = BaselineQualityDynamics().transition(
            state=state(),
            action=action(),
            development_spend=40_000.0,
            quality_lag_weeks=1,
            development_quality_response=0.03,
            development_quality_exponent=0.55,
        )
        strong = BaselineQualityDynamics().transition(
            state=state(),
            action=action(),
            development_spend=40_000.0,
            quality_lag_weeks=1,
            development_quality_response=0.30,
            development_quality_exponent=0.55,
        )
        modest_gain = sum(e.improvement for e in modest.pending_effects)
        strong_gain = sum(e.improvement for e in strong.pending_effects)
        assert strong_gain == pytest.approx(modest_gain * 10.0, rel=1e-6)

    def test_zero_spend_buys_nothing(self) -> None:
        assert improvement_at(0.0) == 0.0
