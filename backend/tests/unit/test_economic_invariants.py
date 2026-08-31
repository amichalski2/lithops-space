from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest
from lithops.application.weekly_planning import (
    action_plan_from_simulation,
    simulation_action_from_action_plan,
)
from lithops.domain.economics import (
    AccountingPeriod,
    EconomicInvariantCode,
    PeriodicMoney,
    RatePeriod,
)
from lithops.domain.executable_model import (
    ModelOutcomeDistribution,
    ModelOutcomeSample,
)
from lithops.domain.models import ObservationSnapshot, RunRecord
from lithops.model_runtime import baseline
from lithops.model_runtime.invariants import evaluate_model_outcomes
from lithops.simulator import SimulationAction, SimulationState, state_transition
from lithops.simulator.invariants import (
    ActionEconomicPolicy,
    evaluate_simulation_action,
)
from pydantic import ValidationError

ARTIFACT_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
FITTED_ID = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


def state(**updates: float | int) -> SimulationState:
    values = {
        "week": 4,
        "cash": 100_000.0,
        "revenue_weekly": 2_000.0,
        "customers": 100.0,
        "churn_rate": 0.05,
        "price_per_customer_weekly": 20.0,
        "weekly_acquisition": 5.0,
        "marketing_spend": 1_000.0,
        "development_spend": 500.0,
        "product_quality": 0.5,
        "capacity": 500.0,
        "reputation": 0.5,
        "operating_cost_per_customer_weekly": 10.0,
    }
    values.update(updates)
    return SimulationState.model_validate(values)


def action(**updates: float | int | None | str) -> SimulationAction:
    values = {
        "name": "candidate",
        "price_per_customer_weekly": 20.0,
        "marketing_spend": 1_000.0,
        "development_spend": 500.0,
    }
    values.update(updates)
    return SimulationAction.model_validate(values)


def distribution(accounting: AccountingPeriod, *, cash: float | None = None):
    return ModelOutcomeDistribution(
        artifact_id=ARTIFACT_ID,
        artifact_hash="a" * 64,
        fitted_model_id=FITTED_ID,
        horizons_days=(7,),
        n_rollouts=1,
        samples=(
            ModelOutcomeSample(
                rollout_index=0,
                horizon_days=7,
                cash=accounting.ending_cash if cash is None else cash,
                revenue_weekly=2_000,
                customers=100,
                churn_rate=0.05,
                accounting=accounting,
            ),
        ),
    )


def test_periodic_money_month_week_round_trip_is_explicit() -> None:
    monthly = PeriodicMoney(amount=90, period=RatePeriod.MONTH_30_DAY)
    weekly = monthly.per(RatePeriod.WEEK)

    assert weekly.amount == pytest.approx(21)
    assert weekly.per(RatePeriod.MONTH_30_DAY).amount == pytest.approx(90)


def test_action_plan_price_round_trip_preserves_weekly_units() -> None:
    current = state(price_per_customer_weekly=21)
    selected = action(price_per_customer_weekly=21)
    observation = ObservationSnapshot(
        day=28,
        cash=current.cash,
        metrics={"price_a": 90, "price_b": 90, "price_c": 90},
    )
    plan = action_plan_from_simulation(RunRecord(), observation, selected, "test")

    restored = simulation_action_from_action_plan(plan, current)

    assert plan.commands[0].arguments == {"A": 90, "B": 90, "C": 90}
    assert restored.price_per_customer_weekly == pytest.approx(21)


def test_action_gate_rejects_price_collapse_below_cost_and_weekly_envelope() -> None:
    report = evaluate_simulation_action(
        state(),
        action(price_per_customer_weekly=0.10),
    )
    codes = {violation.code for violation in report.violations}

    assert not report.valid
    assert EconomicInvariantCode.PRICE_BELOW_ABSOLUTE_FLOOR in codes
    assert EconomicInvariantCode.PRICE_BELOW_VARIABLE_COST in codes
    assert EconomicInvariantCode.PRICE_CHANGE_TOO_LARGE in codes


def test_action_gate_bounds_new_lead_promotion_against_entry_price() -> None:
    report = evaluate_simulation_action(
        state(entry_price_monthly=25),
        action(lead_promotion_monthly=10),
    )

    assert EconomicInvariantCode.LEAD_PROMOTION_TOO_LARGE in {
        violation.code for violation in report.violations
    }


def test_inherited_margin_breach_allows_a_tier_upgrade_but_not_a_price_cut() -> None:
    # A tier upgrade buys delivered quality at a higher compute rate. Inside an
    # inherited breach it is a cost/quality tradeoff for the rollout to weigh,
    # not below-cost pricing, so it must not be vetoed before simulation.
    upgrade = evaluate_simulation_action(
        state(
            price_per_customer_weekly=8,
            operating_cost_per_customer_weekly=10,
            model_tier_a=1,
            model_tier_b=1,
            model_tier_c=1,
        ),
        action(
            price_per_customer_weekly=8,
            model_tier_a=2,
            model_tier_b=2,
            model_tier_c=2,
        ),
    )

    assert upgrade.valid
    assert EconomicInvariantCode.PRICE_BELOW_VARIABLE_COST_INHERITED in {
        warning.code for warning in upgrade.warnings
    }

    price_cut = evaluate_simulation_action(
        state(
            price_per_customer_weekly=8,
            operating_cost_per_customer_weekly=10,
            model_tier_a=1,
            model_tier_b=1,
            model_tier_c=1,
        ),
        action(price_per_customer_weekly=7),
    )

    assert EconomicInvariantCode.PRICE_DEEPENS_VARIABLE_COST_BREACH in {
        violation.code for violation in price_cut.violations
    }


def test_explicit_experiment_can_relax_price_rules_but_not_expired_spend() -> None:
    policy = ActionEconomicPolicy(
        allow_large_price_experiment=True,
        allow_below_cost_experiment=True,
    )
    report = evaluate_simulation_action(
        state(week=4),
        action(
            price_per_customer_weekly=5,
            marketing_spend=1_000,
            marketing_spend_until_week=4,
        ),
        policy=policy,
    )

    assert {violation.code for violation in report.violations} == {
        EconomicInvariantCode.EXPIRED_COMMITMENT
    }


def test_accounting_bridge_accepts_complete_costs() -> None:
    accounting = AccountingPeriod(
        period_days=7,
        starting_cash=100_000,
        recognized_revenue=2_000,
        operating_cost=1_000,
        marketing_spend=500,
        development_spend=250,
        ending_cash=100_250,
    )

    assert evaluate_model_outcomes(distribution(accounting)).valid


def test_accounting_bridge_rejects_omitted_costs_and_cash_mismatch() -> None:
    omitted_costs = AccountingPeriod(
        period_days=7,
        starting_cash=100_000,
        recognized_revenue=2_000,
        operating_cost=0,
        marketing_spend=0,
        development_spend=0,
        ending_cash=100_250,
    )
    report = evaluate_model_outcomes(distribution(omitted_costs, cash=99_000))

    assert {violation.code for violation in report.violations} == {
        EconomicInvariantCode.ACCOUNTING_MISMATCH,
        EconomicInvariantCode.CASH_SAMPLE_MISMATCH,
    }


def test_model_outcome_contract_rejects_negative_counts_and_invalid_rates() -> None:
    accounting = AccountingPeriod(
        period_days=7,
        starting_cash=100_000,
        recognized_revenue=0,
        operating_cost=0,
        marketing_spend=0,
        development_spend=0,
        ending_cash=100_000,
    )
    base = {
        "rollout_index": 0,
        "horizon_days": 7,
        "cash": 100_000,
        "revenue_weekly": 0,
        "customers": 1,
        "churn_rate": 0.1,
        "accounting": accounting,
    }

    with pytest.raises(ValidationError):
        ModelOutcomeSample.model_validate({**base, "customers": -1})
    with pytest.raises(ValidationError):
        ModelOutcomeSample.model_validate({**base, "churn_rate": 1.1})


class TestCashBridgeDeclaresEverySpend:
    """Every outflow the transition deducts must appear in the accounting bridge.

    `targeted_ops_spend` was subtracted from cash by the state transition while
    the bridge omitted it, so the period stopped reconciling the moment the
    control was used and the run died on `accounting_mismatch` — a hard failure
    at week 28 of a live run, from a control we had added weeks earlier.
    """

    def test_transition_and_bridge_deduct_the_same_terms(self) -> None:
        transition_source = Path(state_transition.__file__).read_text(encoding="utf-8")
        cash_expression = transition_source.split("    cash = (", 1)[1].split(")", 1)[0]
        deducted = {
            line.strip().removeprefix("- ").strip()
            for line in cash_expression.splitlines()
            if line.strip().startswith("- ")
        }

        bridge_source = Path(baseline.__file__).read_text(encoding="utf-8")
        bridge = bridge_source.split("def _accounting_period", 1)[1].split(
            "@staticmethod", 1
        )[0]

        missing = {term for term in deducted if term not in bridge}
        assert not missing, (
            "the state transition deducts these from cash but the accounting "
            f"bridge never declares them: {sorted(missing)}"
        )
