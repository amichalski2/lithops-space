"""Regression cover for the live day-21 trap state and feasibility-first planning."""

from __future__ import annotations

import pytest
from lithops.application.weekly_planning import (
    partition_feasible_actions,
    sandbox_action_payload,
)
from lithops.domain.economics import EconomicInvariantCode
from lithops.simulator import SimulationAction, SimulationState
from lithops.simulator.invariants import evaluate_simulation_action
from lithops.simulator.strategy_search import (
    CandidateSimulation,
    RobustnessLevel,
    generate_candidate_actions,
    select_robust_strategy,
)


def trap_state() -> SimulationState:
    """The state observed on day 21 of `qwen32b-executable-12w-v1`.

    Fifteen customers carried the whole weekly compute bill, so the adapter's
    `weekly_compute_cost / active_customers` division reported a per-customer cost far
    above the roughly two-dollar weekly price. No bounded price change could close that
    gap, which emptied the candidate pool and failed the week.
    """

    return SimulationState(
        week=3,
        cash=980_357.60,
        revenue_weekly=29.70,
        customers=15.0,
        churn_rate=0.05,
        price_per_customer_weekly=1.98,
        weekly_acquisition=11.0,
        marketing_spend=1_000.0,
        development_spend=1_690.0,
        operations_spend=3_500.0,
        product_quality=0.33,
        capacity=1_000.0,
        reputation=0.5,
        operating_cost_per_customer_weekly=40.0,
    )


def candidate(name: str, *, bankruptcy: float, utility: float) -> CandidateSimulation:
    return CandidateSimulation(
        action=SimulationAction(
            name=name,
            price_per_customer_weekly=20.0,
            marketing_spend=1_000.0,
            development_spend=500.0,
        ),
        expected_ending_cash=100_000.0,
        downside_ending_cash=50_000.0,
        bankruptcy_probability=bankruptcy,
        expected_customer_growth=1.0,
        robustness=RobustnessLevel.HIGH,
        robust_utility=utility,
        rollout_count=10,
    )


def test_inherited_cost_breach_keeps_continuation_feasible() -> None:
    state = trap_state()
    continuation = SimulationAction(
        name="balanced_growth",
        price_per_customer_weekly=state.price_per_customer_weekly,
        marketing_spend=state.marketing_spend,
        development_spend=state.development_spend,
    )

    report = evaluate_simulation_action(state, continuation)

    assert report.valid
    assert {warning.code for warning in report.warnings} == {
        EconomicInvariantCode.PRICE_BELOW_VARIABLE_COST_INHERITED
    }


def test_inherited_cost_breach_still_rejects_a_deeper_price_cut() -> None:
    state = trap_state()
    deeper = SimulationAction(
        name="undercut",
        price_per_customer_weekly=state.price_per_customer_weekly * 0.9,
        marketing_spend=state.marketing_spend,
        development_spend=state.development_spend,
    )

    report = evaluate_simulation_action(state, deeper)

    assert not report.valid
    assert EconomicInvariantCode.PRICE_DEEPENS_VARIABLE_COST_BREACH in {
        violation.code for violation in report.violations
    }


def test_healthy_state_still_rejects_entering_a_cost_breach() -> None:
    healthy = trap_state().model_copy(
        update={"price_per_customer_weekly": 50.0, "operating_cost_per_customer_weekly": 45.0}
    )
    entering = SimulationAction(
        name="discount",
        price_per_customer_weekly=40.0,
        marketing_spend=healthy.marketing_spend,
        development_spend=healthy.development_spend,
    )

    report = evaluate_simulation_action(healthy, entering)

    assert not report.valid
    assert EconomicInvariantCode.PRICE_BELOW_VARIABLE_COST in {
        violation.code for violation in report.violations
    }


def test_day21_trap_state_leaves_a_feasible_pool_with_visible_evidence() -> None:
    state = trap_state()

    feasible, feasibility = partition_feasible_actions(
        state,
        generate_candidate_actions(state),
    )

    assert feasible, "the day-21 trap state must still yield an executable plan"
    assert feasibility.feasible_count == feasibility.generated_count
    assert not feasibility.degraded
    assert (
        EconomicInvariantCode.PRICE_BELOW_VARIABLE_COST_INHERITED.value
        in feasibility.warning_codes
    ), "an unresolved cost breach must stay visible even when the week proceeds"


def test_degraded_pool_records_every_rejection_code() -> None:
    state = trap_state()
    pool = (
        *generate_candidate_actions(state),
        SimulationAction(
            name="undercut",
            price_per_customer_weekly=state.price_per_customer_weekly * 0.5,
            marketing_spend=state.marketing_spend,
            development_spend=state.development_spend,
        ),
    )

    feasible, feasibility = partition_feasible_actions(state, pool)

    assert len(feasible) == len(pool) - 1
    assert feasibility.degraded
    assert [rejected.strategy for rejected in feasibility.rejected] == ["undercut"]
    payload = feasibility.as_payload()
    assert payload["rejected"][0]["violation_codes"], "rejection evidence must carry codes"


def test_selection_no_longer_forces_three_candidates() -> None:
    result = select_robust_strategy((candidate("only_feasible", bankruptcy=0.01, utility=1.0),))

    assert result.selected.action.name == "only_feasible"
    assert result.selection_reason_code == "survival_gated_robust_utility"


def test_selection_still_rejects_an_empty_pool() -> None:
    with pytest.raises(ValueError, match="at least one candidate"):
        select_robust_strategy(())


def test_sandbox_action_lifetime_preserves_persistent_and_temporary_semantics() -> None:
    state = trap_state()
    persistent = SimulationAction(
        name="persistent",
        price_per_customer_weekly=state.effective_catalog_price_per_customer_weekly,
        marketing_spend=2_000.0,
        development_spend=1_000.0,
        operations_spend=state.operations_spend,
    )
    temporary = persistent.model_copy(
        update={
            "name": "temporary",
            "marketing_spend_until_week": state.week + 2,
            "development_spend_until_week": state.week + 2,
        }
    )

    assert sandbox_action_payload(persistent, state, horizon_weeks=26)[
        "experiment_duration_weeks"
    ] == 26
    assert sandbox_action_payload(temporary, state, horizon_weeks=26)[
        "experiment_duration_weeks"
    ] == 2
    assert sandbox_action_payload(temporary, state, horizon_weeks=26)[
        "development_spend_duration_weeks"
    ] == 2


def test_sandbox_payload_represents_one_bounded_control_beside_a_persistent_control() -> None:
    current = trap_state().model_copy(update={"week": 3, "marketing_spend": 700.0})
    development_program = SimulationAction(
        name="development-program",
        price_per_customer_weekly=current.effective_catalog_price_per_customer_weekly,
        marketing_spend=700.0,
        development_spend=5_350.0,
        operations_spend=current.operations_spend,
        development_spend_until_week=8,
        development_spend_after_experiment=350.0,
    )

    payload = sandbox_action_payload(development_program, current, horizon_weeks=26)

    assert payload["experiment_duration_weeks"] == 5
    assert payload["development_spend_duration_weeks"] == 5
    assert payload["marketing_spend_after_experiment"] == 700.0
    assert payload["development_spend_after_experiment"] == 350.0


def test_sandbox_payload_represents_different_marketing_and_development_stops() -> None:
    current = trap_state().model_copy(update={"week": 3})
    incompatible = SimulationAction(
        name="incompatible-program",
        price_per_customer_weekly=current.effective_catalog_price_per_customer_weekly,
        marketing_spend=700.0,
        development_spend=5_350.0,
        operations_spend=current.operations_spend,
        marketing_spend_until_week=4,
        development_spend_until_week=8,
    )

    payload = sandbox_action_payload(incompatible, current, horizon_weeks=26)

    assert payload["experiment_duration_weeks"] == 1
    assert payload["development_spend_duration_weeks"] == 5
