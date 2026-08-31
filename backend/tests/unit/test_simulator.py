from datetime import UTC, datetime
from random import Random
from uuid import UUID

import pytest
from lithops.domain.models import ObservationSnapshot
from lithops.domain.world_model import WorldModelParameterName
from lithops.simulator import (
    SimulationAction,
    SimulationState,
    TargetedAdAllocation,
    TargetedDevelopmentAllocation,
    simulate,
)
from lithops.simulator.models import ProcessNoise, WeeklyShock
from lithops.simulator.parameter_sampling import sample_parameters
from lithops.simulator.state_transition import advance_simulation_week
from lithops.world_model import bootstrap_world_model

RUN_ID = UUID("22222222-2222-2222-2222-222222222222")


def world_model():
    return bootstrap_world_model(
        RUN_ID,
        ObservationSnapshot(
            day=0,
            cash=500_000,
            observed_at=datetime(2026, 8, 25, tzinfo=UTC),
        ),
    )


def state() -> SimulationState:
    return SimulationState(
        cash=500_000,
        revenue_weekly=50_000,
        customers=1_000,
        churn_rate=0.04,
        price_per_customer_weekly=50,
        weekly_acquisition=80,
        marketing_spend=10_000,
        development_spend=5_000,
        product_quality=0.6,
        capacity=5_000,
        reputation=0.6,
    )


def action(**updates: float | str) -> SimulationAction:
    values = {
        "name": "balanced",
        "price_per_customer_weekly": 50.0,
        "marketing_spend": 10_000.0,
        "development_spend": 5_000.0,
        "segment_focus": 1.0,
    }
    values.update(updates)
    return SimulationAction.model_validate(values)


def estimates() -> dict[WorldModelParameterName, float]:
    return {parameter.name: parameter.estimate for parameter in world_model().parameters}


def test_rollouts_are_reproducible_for_the_same_seed() -> None:
    inputs = {
        "state": state(),
        "action": action(),
        "world_model": world_model(),
        "horizon_weeks": 12,
        "n_rollouts": 20,
        "seed": 42,
    }
    assert simulate(**inputs) == simulate(**inputs)


def test_sampled_parameters_stay_inside_declared_uncertainty() -> None:
    model = world_model()
    random = Random(7)

    for _ in range(100):
        sampled = sample_parameters(model, random)
        for parameter in model.parameters:
            assert parameter.lower_bound <= sampled[parameter.name] <= parameter.upper_bound


def test_marketing_response_saturates_and_state_invariants_hold() -> None:
    current = state()
    parameters = estimates()
    low = advance_simulation_week(current, action(marketing_spend=10_000), parameters)
    medium = advance_simulation_week(current, action(marketing_spend=20_000), parameters)
    high = advance_simulation_week(current, action(marketing_spend=30_000), parameters)

    assert low.weekly_acquisition < medium.weekly_acquisition < high.weekly_acquisition
    assert medium.weekly_acquisition - low.weekly_acquisition > (
        high.weekly_acquisition - medium.weekly_acquisition
    )
    assert 0 <= high.churn_rate <= 1
    assert 0 <= high.product_quality <= 1
    assert 0 <= high.customers <= high.capacity


def test_price_increase_reduces_acquisition_and_increases_churn() -> None:
    current = state()
    parameters = estimates()
    unchanged = advance_simulation_week(current, action(), parameters)
    increased = advance_simulation_week(
        current,
        action(price_per_customer_weekly=55),
        parameters,
    )

    assert increased.weekly_acquisition < unchanged.weekly_acquisition
    assert increased.churn_rate > unchanged.churn_rate


def test_model_tier_intervention_trades_quality_for_variable_compute_cost() -> None:
    current = state().model_copy(
        update={
            "operating_cost_per_customer_weekly": 1.0,
            "model_tier_a": 1,
            "model_tier_b": 1,
            "model_tier_c": 1,
        }
    )
    held = advance_simulation_week(
        current,
        action(model_tier_a=1, model_tier_b=1, model_tier_c=1),
        estimates(),
    )
    upgraded = advance_simulation_week(
        current,
        action(model_tier_a=2, model_tier_b=2, model_tier_c=2),
        estimates(),
    )

    assert upgraded.weekly_acquisition > held.weekly_acquisition
    assert upgraded.operating_cost_per_customer_weekly > (
        held.operating_cost_per_customer_weekly
    )
    assert upgraded.cash < held.cash


def test_marketing_can_bootstrap_acquisition_from_a_zero_customer_company() -> None:
    initial = state().model_copy(
        update={"customers": 0, "weekly_acquisition": 0, "marketing_spend": 0}
    )
    acquisition_action = action(marketing_spend=1_000)

    next_week = advance_simulation_week(initial, acquisition_action, estimates())

    assert next_week.weekly_acquisition > 0
    assert next_week.customers > 0


def test_only_incremental_marketing_can_reopen_acquisition_after_zero_conversion_week() -> None:
    observed_zero_conversion = state().model_copy(
        update={
            "week": 1,
            "customers": 0,
            "weekly_acquisition": 0,
            "marketing_spend": 1_000,
        }
    )

    held = advance_simulation_week(
        observed_zero_conversion,
        action(marketing_spend=1_000),
        estimates(),
    )
    increased = advance_simulation_week(
        observed_zero_conversion,
        action(marketing_spend=10_000),
        estimates(),
    )

    assert held.weekly_acquisition == 0
    assert held.customers == 0
    assert increased.weekly_acquisition > 0
    assert increased.customers > 0


def test_failed_leads_replace_the_startup_cac_prior_instead_of_resetting_it() -> None:
    first_evidence = state().model_copy(
        update={
            "week": 1,
            "customers": 0,
            "weekly_acquisition": 0,
            "weekly_leads": 20,
            "weekly_conversions": 0,
            "total_leads": 20,
            "total_conversions": 0,
            "marketing_spend": 1_000,
        }
    )
    repeated_failure = first_evidence.model_copy(
        update={"week": 6, "total_leads": 120}
    )

    early = advance_simulation_week(
        first_evidence, action(marketing_spend=2_000), estimates()
    )
    late = advance_simulation_week(
        repeated_failure, action(marketing_spend=2_000), estimates()
    )

    assert early.weekly_acquisition < 1
    assert late.weekly_acquisition < early.weekly_acquisition
    assert late.weekly_acquisition < 1_000 / 200, (
        "observed funnel failures must prevent the day-zero CAC prior from restarting"
    )


def test_lead_promotion_trades_conversion_for_first_period_revenue() -> None:
    funnel = state().model_copy(
        update={
            "week": 1,
            "customers": 0,
            "weekly_acquisition": 0,
            "weekly_leads": 20,
            "total_leads": 20,
            "entry_price_monthly": 25,
            "lead_promotion_monthly": 0,
        }
    )
    held = advance_simulation_week(
        funnel,
        action(lead_promotion_monthly=0),
        estimates(),
    )
    promoted = advance_simulation_week(
        funnel,
        action(lead_promotion_monthly=5),
        estimates(),
    )

    assert promoted.weekly_acquisition > held.weekly_acquisition
    assert (
        promoted.revenue_weekly / promoted.weekly_acquisition
        < held.revenue_weekly / held.weekly_acquisition
    )


def test_lead_promotion_experiment_reverts_after_one_week() -> None:
    current = state().model_copy(
        update={"week": 2, "entry_price_monthly": 25, "lead_promotion_monthly": 0}
    )
    experiment = action(
        lead_promotion_monthly=5,
        lead_promotion_until_week=3,
        lead_promotion_after_experiment=0,
    )

    experimental_week = advance_simulation_week(current, experiment, estimates())
    reverted_week = advance_simulation_week(experimental_week, experiment, estimates())

    assert experimental_week.lead_promotion_monthly == 5
    assert reverted_week.lead_promotion_monthly == 0


def test_development_investment_changes_quality_only_after_the_learned_lag() -> None:
    current = state()
    parameters = estimates()
    invested = advance_simulation_week(
        current,
        action(development_spend=25_000),
        parameters,
    )
    assert invested.product_quality == current.product_quality

    held = action(development_spend=0)
    for _ in range(3):
        invested = advance_simulation_week(invested, held, parameters)
        assert invested.product_quality == current.product_quality

    invested = advance_simulation_week(invested, held, parameters)
    # The magnitude belongs to the learned development response, so this pins the
    # lag and the direction rather than a coefficient: asserting an exact figure
    # here once locked in a linear form with a ceiling.
    assert invested.product_quality > current.product_quality

    bolder = advance_simulation_week(
        current,
        action(development_spend=250_000),
        parameters,
    )
    for _ in range(4):
        bolder = advance_simulation_week(bolder, held, parameters)
    assert bolder.product_quality > invested.product_quality


def test_targeted_development_matures_before_the_deferred_acquisition_probe() -> None:
    current = state().model_copy(
        update={
            "week": 2,
            "cash": 1_000_000,
            "customers": 0,
            "revenue_weekly": 0,
            "weekly_acquisition": 0,
            "marketing_spend": 0,
            "development_spend": 0,
            "product_quality": 0.2,
        }
    )
    experiment = action(
        marketing_spend=7_000,
        targeted_ad_allocations=(
            TargetedAdAllocation(
                channel="social_media", segment="S2", daily_spend=1_000
            ),
        ),
        marketing_spend_start_week=7,
        marketing_spend_until_week=8,
        marketing_spend_after_experiment=0,
        development_spend=0,
        targeted_development_allocations=(
            TargetedDevelopmentAllocation(segment="S2", daily_spend=2_000),
        ),
        targeted_development_spend_until_week=5,
        targeted_development_spend_after_experiment=0,
    )

    first = advance_simulation_week(current, experiment, estimates())
    assert first.targeted_development_spend == 14_000
    assert first.marketing_spend == 0
    assert first.cash == pytest.approx(986_000)

    progressed = first
    for _ in range(4):
        progressed = advance_simulation_week(progressed, experiment, estimates())
    assert progressed.product_quality > current.product_quality
    assert progressed.targeted_development_spend == 0
    assert progressed.marketing_spend == 0

    probe = advance_simulation_week(progressed, experiment, estimates())
    assert probe.marketing_spend == 7_000


def test_bounded_experiment_spend_expires_at_its_absolute_stop_week() -> None:
    current = state().model_copy(
        update={
            "week": 2,
            "cash": 1_000_000,
            "revenue_weekly": 0,
            "customers": 0,
            "weekly_acquisition": 0,
        }
    )
    experiment = action(
        marketing_spend=7_000,
        development_spend=56_000,
        marketing_spend_until_week=4,
        development_spend_until_week=4,
    )

    week_three = advance_simulation_week(current, experiment, estimates())
    week_four = advance_simulation_week(week_three, experiment, estimates())
    week_five = advance_simulation_week(week_four, experiment, estimates())

    assert week_three.cash == pytest.approx(937_000)
    assert week_four.cash == pytest.approx(874_000)
    assert week_five.cash == pytest.approx(874_000)
    assert week_five.marketing_spend == 0
    assert week_five.development_spend == 0


def test_bounded_experiment_reverts_to_recorded_operating_spend() -> None:
    current = state().model_copy(update={"week": 2})
    experiment = action(
        marketing_spend=7_000,
        development_spend=9_000,
        marketing_spend_until_week=3,
        development_spend_until_week=3,
        marketing_spend_after_experiment=2_000,
        development_spend_after_experiment=3_000,
    )

    experimental_week = advance_simulation_week(current, experiment, estimates())
    reverted_week = advance_simulation_week(experimental_week, experiment, estimates())

    assert experimental_week.marketing_spend == 7_000
    assert experimental_week.development_spend == 9_000
    assert reverted_week.marketing_spend == 2_000
    assert reverted_week.development_spend == 3_000


def test_process_noise_widens_outcomes_and_is_seed_reproducible() -> None:
    noise = ProcessNoise(acquisition_sigma=0.15, churn_sigma=0.01, revenue_sigma=0.03)
    arguments = {
        "state": state(),
        "action": action(),
        "world_model": world_model(),
        "horizon_weeks": 4,
        "n_rollouts": 40,
        "seed": 7,
    }

    quiet = simulate(**arguments)
    noisy = simulate(**arguments, process_noise=noise)
    repeated = simulate(**arguments, process_noise=noise)

    quiet_cash = {outcome.ending_cash for outcome in quiet}
    noisy_cash = [outcome.ending_cash for outcome in noisy]

    assert [outcome.ending_cash for outcome in repeated] == noisy_cash, (
        "the same seed must reproduce the same innovations"
    )
    assert len(set(noisy_cash)) > 1, "process noise must produce a real spread"
    assert max(noisy_cash) - min(noisy_cash) > max(quiet_cash) - min(quiet_cash)


def test_omitted_process_noise_keeps_the_legacy_transition_exact() -> None:
    arguments = {
        "state": state(),
        "action": action(),
        "world_model": world_model(),
        "horizon_weeks": 4,
        "n_rollouts": 10,
        "seed": 3,
    }

    legacy = simulate(**arguments)
    explicit_zero = simulate(**arguments, process_noise=ProcessNoise())

    assert [outcome.ending_cash for outcome in explicit_zero] == [
        outcome.ending_cash for outcome in legacy
    ]


def test_catalog_change_preserves_observed_arpu_realization_ratio() -> None:
    current = state().model_copy(
        update={
            "price_per_customer_weekly": 5.0,
            "catalog_price_per_customer_weekly": 20.0,
        }
    )
    next_state = advance_simulation_week(
        current,
        action(price_per_customer_weekly=22.0),
        sample_parameters(world_model(), Random(0)),
    )

    assert next_state.catalog_price_per_customer_weekly == pytest.approx(22.0)
    assert next_state.price_per_customer_weekly == pytest.approx(5.5)
    assert next_state.revenue_weekly == pytest.approx(next_state.customers * 5.5)


def test_additive_cash_flow_noise_widens_a_zero_revenue_company() -> None:
    arguments = {
        "state": state().model_copy(
            update={
                "revenue_weekly": 0.0,
                "customers": 0.0,
                "weekly_acquisition": 0.0,
            }
        ),
        "action": action(marketing_spend=0.0, development_spend=0.0),
        "world_model": world_model(),
        "horizon_weeks": 1,
        "n_rollouts": 40,
        "seed": 17,
        "process_noise": ProcessNoise(cash_flow_sigma=3_000.0),
    }

    outcomes = simulate(**arguments)

    assert len({outcome.ending_cash for outcome in outcomes}) == 40
    assert any(
        outcome.states[-1].cash_flow_adjustment_weekly != 0.0
        for outcome in outcomes
    )


def test_transient_revenue_shock_does_not_compound_into_later_arpu() -> None:
    current = state().model_copy(
        update={
            "price_per_customer_weekly": 5.0,
            "catalog_price_per_customer_weekly": 20.0,
            "price_realization_ratio": 0.25,
        }
    )
    parameters = sample_parameters(world_model(), Random(0))
    shocked = advance_simulation_week(
        current,
        action(price_per_customer_weekly=20.0),
        parameters,
        WeeklyShock(revenue_multiplier=2.0),
    )
    quiet = advance_simulation_week(
        shocked,
        action(price_per_customer_weekly=20.0),
        parameters,
    )

    assert shocked.price_per_customer_weekly == pytest.approx(10.0)
    assert quiet.price_per_customer_weekly == pytest.approx(5.0)
