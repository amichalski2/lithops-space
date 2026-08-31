"""Feasibility and unit-economics gates independent from model-authored code."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from lithops.domain.economics import (
    EconomicInvariantCode,
    EconomicInvariantReport,
    EconomicInvariantViolation,
)
from lithops.simulator.models import SimulationAction, SimulationState
from lithops.simulator.state_transition import MODEL_TIER_COST_PER_USAGE_UNIT


@dataclass(frozen=True, slots=True)
class ActionEconomicPolicy:
    max_price_change_fraction: float = 0.25
    minimum_weekly_price: float = 1.0
    minimum_variable_cost_coverage: float = 1.0
    allow_large_price_experiment: bool = False
    allow_below_cost_experiment: bool = False
    max_lead_promotion_fraction: float = 0.25

    def __post_init__(self) -> None:
        if not 0 <= self.max_price_change_fraction <= 1:
            raise ValueError("maximum price change fraction must be between zero and one")
        if self.minimum_weekly_price < 0:
            raise ValueError("minimum weekly price cannot be negative")
        if self.minimum_variable_cost_coverage < 0:
            raise ValueError("minimum variable-cost coverage cannot be negative")
        if not 0 <= self.max_lead_promotion_fraction <= 1:
            raise ValueError("maximum lead promotion fraction must be between zero and one")


def evaluate_simulation_action(
    state: SimulationState,
    action: SimulationAction,
    *,
    policy: ActionEconomicPolicy | None = None,
) -> EconomicInvariantReport:
    rules = policy or ActionEconomicPolicy()
    violations: list[EconomicInvariantViolation] = []
    warnings: list[EconomicInvariantViolation] = []
    numeric_values = {
        "price_per_customer_weekly": action.price_per_customer_weekly,
        "marketing_spend": action.marketing_spend,
        "development_spend": action.development_spend,
        "operations_spend": (
            state.operations_spend
            if action.operations_spend is None
            else action.operations_spend
        ),
    }
    if action.marketing_spend_after_experiment is not None:
        numeric_values["marketing_spend_after_experiment"] = (
            action.marketing_spend_after_experiment
        )
    if action.development_spend_after_experiment is not None:
        numeric_values["development_spend_after_experiment"] = (
            action.development_spend_after_experiment
        )
    selected_lead_promotion = (
        state.lead_promotion_monthly
        if action.lead_promotion_monthly is None
        else action.lead_promotion_monthly
    )
    numeric_values["lead_promotion_monthly"] = selected_lead_promotion
    if action.lead_promotion_after_experiment is not None:
        numeric_values["lead_promotion_after_experiment"] = (
            action.lead_promotion_after_experiment
        )
    for name, value in numeric_values.items():
        if not isfinite(value):
            violations.append(
                EconomicInvariantViolation(
                    code=EconomicInvariantCode.NON_FINITE_VALUE,
                    path=f"action.{name}",
                    message="action economic values must be finite",
                    actual=str(value),
                )
            )
    if action.price_per_customer_weekly < rules.minimum_weekly_price:
        violations.append(
            EconomicInvariantViolation(
                code=EconomicInvariantCode.PRICE_BELOW_ABSOLUTE_FLOOR,
                path="action.price_per_customer_weekly",
                message="weekly price falls below the absolute policy floor",
                expected=rules.minimum_weekly_price,
                actual=action.price_per_customer_weekly,
            )
        )
    entry_price_monthly = state.entry_price_monthly or (
        state.effective_catalog_price_per_customer_weekly * 30.0 / 7.0
    )
    maximum_lead_promotion = entry_price_monthly * rules.max_lead_promotion_fraction
    if (
        selected_lead_promotion > maximum_lead_promotion
        and selected_lead_promotion > state.lead_promotion_monthly + 1e-9
    ):
        violations.append(
            EconomicInvariantViolation(
                code=EconomicInvariantCode.LEAD_PROMOTION_TOO_LARGE,
                path="action.lead_promotion_monthly",
                message=(
                    "new-lead promotion exceeds the bounded share of the entry plan price"
                ),
                expected=f"<= {maximum_lead_promotion:.8g}",
                actual=selected_lead_promotion,
            )
        )
    current_tiers = (state.model_tier_a, state.model_tier_b, state.model_tier_c)
    selected_tiers = (
        action.model_tier_a or state.model_tier_a,
        action.model_tier_b or state.model_tier_b,
        action.model_tier_c or state.model_tier_c,
    )
    current_compute_rate = sum(
        MODEL_TIER_COST_PER_USAGE_UNIT[tier] for tier in current_tiers
    ) / len(current_tiers)
    selected_compute_rate = sum(
        MODEL_TIER_COST_PER_USAGE_UNIT[tier] for tier in selected_tiers
    ) / len(selected_tiers)
    projected_variable_cost = state.operating_cost_per_customer_weekly * (
        selected_compute_rate / current_compute_rate
    )
    minimum_cost_covered_arpu = (
        projected_variable_cost * rules.minimum_variable_cost_coverage
    )
    current_catalog_price = state.effective_catalog_price_per_customer_weekly
    current_arpu = state.price_per_customer_weekly
    projected_arpu = state.projected_arpu(action.price_per_customer_weekly)
    # The gate is state-aware. A company whose *current* price already fails cost coverage
    # cannot be rescued inside one bounded weekly price envelope, so rejecting every
    # candidate there would leave the week with no feasible plan at all. In that inherited
    # breach only a candidate that deepens the shortfall is infeasible; holding or raising
    # price stays feasible and records the unresolved breach as a warning.
    state_covers_cost = current_arpu >= (
        state.operating_cost_per_customer_weekly
        * rules.minimum_variable_cost_coverage
    )
    # Coverage that fails only once the tier ratio is applied is a cost/quality
    # tradeoff, not below-cost pricing: the price itself still covers the observed
    # unit cost, and scaling the whole operating cost by the compute-rate ratio
    # overstates the compute share. Rollouts and robust selection weigh that
    # tradeoff on cash; a pre-simulation veto would remove the fastest quality
    # lever from the pool outright.
    price_only_cost_covered_arpu = (
        state.operating_cost_per_customer_weekly * rules.minimum_variable_cost_coverage
    )
    tier_driven_shortfall = (
        selected_compute_rate > current_compute_rate
        and projected_arpu >= price_only_cost_covered_arpu
    )
    if (
        not rules.allow_below_cost_experiment
        and projected_arpu < minimum_cost_covered_arpu
    ):
        if state_covers_cost and tier_driven_shortfall:
            warnings.append(
                EconomicInvariantViolation(
                    code=EconomicInvariantCode.TIER_COST_COVERAGE_PRESSURE,
                    path="action.model_tiers",
                    message=(
                        "the selected model tiers raise projected compute cost above "
                        "the configured coverage share while the price still covers "
                        "observed unit cost"
                    ),
                    expected=minimum_cost_covered_arpu,
                    actual=projected_arpu,
                )
            )
        elif state_covers_cost:
            violations.append(
                EconomicInvariantViolation(
                    code=EconomicInvariantCode.PRICE_BELOW_VARIABLE_COST,
                    path="action.projected_arpu_weekly",
                    message="projected ARPU does not cover the configured share of variable cost",
                    expected=minimum_cost_covered_arpu,
                    actual=projected_arpu,
                )
            )
        elif projected_arpu < current_arpu - 1e-9:
            # Inside an inherited breach only a genuine price cut deepens the
            # shortfall. A tier-driven cost increase is judged by the rollout,
            # not vetoed here, for the same reason as above.
            violations.append(
                EconomicInvariantViolation(
                    code=EconomicInvariantCode.PRICE_DEEPENS_VARIABLE_COST_BREACH,
                    path="action.projected_arpu_weekly",
                    message=(
                        "unit economics already fail variable-cost coverage and this "
                        "candidate cuts price further"
                    ),
                    expected=f">= {current_arpu:.8g}",
                    actual=projected_arpu,
                )
            )
        else:
            warnings.append(
                EconomicInvariantViolation(
                    code=EconomicInvariantCode.PRICE_BELOW_VARIABLE_COST_INHERITED,
                    path="action.projected_arpu_weekly",
                    message=(
                        "weekly price inherits an existing variable-cost shortfall that one "
                        "bounded price change cannot close"
                    ),
                    expected=minimum_cost_covered_arpu,
                    actual=projected_arpu,
                )
            )
    lower = current_catalog_price * (1.0 - rules.max_price_change_fraction)
    upper = current_catalog_price * (1.0 + rules.max_price_change_fraction)
    if (
        not rules.allow_large_price_experiment
        and not lower <= action.price_per_customer_weekly <= upper
    ):
        violations.append(
            EconomicInvariantViolation(
                code=EconomicInvariantCode.PRICE_CHANGE_TOO_LARGE,
                path="action.price_per_customer_weekly",
                message="price change exceeds the bounded weekly decision envelope",
                expected=f"{lower:.8g}..{upper:.8g}",
                actual=action.price_per_customer_weekly,
            )
        )
    for spend_name, spend, stop_week in (
        ("marketing_spend", action.marketing_spend, action.marketing_spend_until_week),
        (
            "development_spend",
            action.development_spend,
            action.development_spend_until_week,
        ),
        (
            "lead_promotion_monthly",
            selected_lead_promotion,
            action.lead_promotion_until_week,
        ),
    ):
        if stop_week is not None and stop_week <= state.week and spend > 0:
            violations.append(
                EconomicInvariantViolation(
                    code=EconomicInvariantCode.EXPIRED_COMMITMENT,
                    path=f"action.{spend_name}_until_week",
                    message="an expired temporary commitment still carries positive spend",
                    expected=f"> {state.week}",
                    actual=float(stop_week),
                )
            )
    return EconomicInvariantReport(
        violations=tuple(violations),
        warnings=tuple(warnings),
    )
