"""Exact multi-week commitment schedules in the deterministic transition.

A committed program is an explicit week-by-week schedule: build weeks at the
treatment level, one probe week, then the explicit reversion. Every scheduled
outflow must appear exactly in the cash path — no smearing, no silent renewal.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from lithops.domain.models import ObservationSnapshot
from lithops.simulator import SimulationAction, SimulationState
from lithops.simulator.state_transition import advance_simulation_week
from lithops.world_model import bootstrap_world_model

RUN_ID = UUID("33333333-3333-3333-3333-333333333333")

BASELINE_DEVELOPMENT = 5_000.0
TREATMENT_DEVELOPMENT = 15_000.0
BASELINE_MARKETING = 10_000.0
PROBE_MARKETING = 8_000.0


def parameters():
    model = bootstrap_world_model(
        RUN_ID,
        ObservationSnapshot(
            day=0,
            cash=500_000,
            observed_at=datetime(2026, 8, 26, tzinfo=UTC),
        ),
    )
    return {parameter.name: parameter.estimate for parameter in model.parameters}


def initial_state() -> SimulationState:
    return SimulationState(
        week=0,
        cash=500_000,
        revenue_weekly=50_000,
        customers=1_000,
        churn_rate=0.04,
        price_per_customer_weekly=50,
        weekly_acquisition=80,
        marketing_spend=BASELINE_MARKETING,
        development_spend=BASELINE_DEVELOPMENT,
        product_quality=0.6,
        capacity=5_000,
        reputation=0.6,
    )


def scheduled_action() -> SimulationAction:
    """Three build weeks of elevated development, one probe week, reversion."""

    return SimulationAction(
        name="committed_development_program",
        price_per_customer_weekly=50.0,
        marketing_spend=PROBE_MARKETING,
        development_spend=TREATMENT_DEVELOPMENT,
        development_spend_until_week=3,
        development_spend_after_experiment=BASELINE_DEVELOPMENT,
        marketing_spend_start_week=3,
        marketing_spend_until_week=4,
        marketing_spend_after_experiment=BASELINE_MARKETING,
    )


def continuation_action() -> SimulationAction:
    return SimulationAction(
        name="continuation",
        price_per_customer_weekly=50.0,
        marketing_spend=BASELINE_MARKETING,
        development_spend=BASELINE_DEVELOPMENT,
    )


def walk(action: SimulationAction, weeks: int) -> list[SimulationState]:
    estimates = parameters()
    current = initial_state()
    states = []
    for _ in range(weeks):
        current = advance_simulation_week(current, action, estimates)
        states.append(current)
    return states


def test_three_build_weeks_probe_and_reversion_apply_exact_spend_levels() -> None:
    states = walk(scheduled_action(), 5)
    assert [state.development_spend for state in states] == [
        TREATMENT_DEVELOPMENT,
        TREATMENT_DEVELOPMENT,
        TREATMENT_DEVELOPMENT,
        BASELINE_DEVELOPMENT,
        BASELINE_DEVELOPMENT,
    ]
    assert [state.marketing_spend for state in states] == [
        BASELINE_MARKETING,
        BASELINE_MARKETING,
        BASELINE_MARKETING,
        PROBE_MARKETING,
        BASELINE_MARKETING,
    ]


def test_build_weeks_appear_as_exact_future_outflows_in_the_cash_path() -> None:
    committed = walk(scheduled_action(), 3)
    held = walk(continuation_action(), 3)
    weekly_increment = TREATMENT_DEVELOPMENT - BASELINE_DEVELOPMENT
    for index, (with_program, without_program) in enumerate(
        zip(committed, held, strict=True), start=1
    ):
        assert without_program.cash - with_program.cash == (
            weekly_increment * index
        ), "each build week must appear as one exact scheduled outflow"


def test_reversion_stops_the_outflow_without_residual_spend() -> None:
    committed = walk(scheduled_action(), 5)
    assert committed[3].development_spend == BASELINE_DEVELOPMENT
    assert committed[4].development_spend == BASELINE_DEVELOPMENT
    assert committed[4].marketing_spend == BASELINE_MARKETING
