"""The run's own history reduced to a series a reader can reason about.

Rates of change — how fast quality actually moves, what a lever bought last
time — are only visible across weeks. Handing over one week at a time hides
exactly the trend a strategic decision turns on.
"""

from __future__ import annotations

from typing import Any

from lithops.domain.models import DecisionRecord

TRAJECTORY_METRICS = (
    "weekly_leads",
    "weekly_conversions",
    "active_customers",
    "churn_rate",
    "weekly_lost_leads",
    "weekly_revenue",
    "product_quality",
    "delivered_quality_plan_a",
    "delivered_quality_plan_b",
    "delivered_quality_plan_c",
    "competitor_quality_bar_shift",
    "model_tier_a",
    "usage_quota_a",
    "marketing_spend",
    "development_spend",
    "targeted_development_spend",
)


def weekly_trajectory(
    decision_history: tuple[DecisionRecord, ...],
) -> list[dict[str, Any]]:
    """The run's own history as a series, not a snapshot.

    Rates of change — how fast quality actually moves, what a lever bought last
    time — are only visible across weeks. Handing over one week at a time hides
    exactly the trend a strategic decision turns on.
    """

    rows: list[dict[str, Any]] = []
    for decision in sorted(decision_history, key=lambda item: item.week):
        outcome = decision.actual_outcome
        if outcome is None:
            continue
        metrics = outcome.metrics or {}
        row: dict[str, Any] = {
            "week": decision.week,
            "strategy": decision.action_plan.strategy_family,
            "cash": round(outcome.cash, 2),
        }
        for name in TRAJECTORY_METRICS:
            value = metrics.get(name)
            if isinstance(value, int | float):
                row[name] = round(float(value), 4)
        rows.append(row)
    # A long run's early weeks matter for the long trend but not row by row:
    # keep every 4th older week and the last 24 in full, so rates stay visible
    # while the series stops growing without bound.
    if len(rows) > 24:
        older = rows[:-24]
        rows = [row for index, row in enumerate(older) if index % 4 == 0] + rows[-24:]
    return rows


# Instrument definition, not a benchmark constant: a week in which at least
# half of an existing base (of three or more customers) cancels is treated as a
# failed retention re-test for the group as a whole.
MASS_CHURN_RATE = 0.5
MASS_CHURN_MINIMUM_CUSTOMERS = 3.0


def revealed_quality_bar_lower_bound(
    decision_history: tuple[DecisionRecord, ...],
) -> float | None:
    """The quality bar the run's own churn has revealed, as a lower bound.

    Retention is a re-test: a subscriber who cancels while delivered quality
    holds steady is reporting that the bar they judge against moved above it.
    Mass churn at delivered quality ``d`` therefore reveals ``bar > d`` for
    that group — a measurement taken from the run's own cohorts, needing no
    announcement parsing at all. Returns ``None`` when no mass-churn week has
    occurred: nothing revealed is not a bar of zero.
    """

    bound: float | None = None
    for decision in decision_history:
        outcome = decision.actual_outcome
        if outcome is None:
            continue
        metrics = outcome.metrics or {}
        churn = metrics.get("churn_rate")
        starting = decision.observation.metrics.get("active_customers")
        if not isinstance(churn, int | float) or not isinstance(starting, int | float):
            continue
        if float(churn) < MASS_CHURN_RATE or float(starting) < MASS_CHURN_MINIMUM_CUSTOMERS:
            continue
        delivered = [
            float(value)
            for name in (
                "delivered_quality_plan_a",
                "delivered_quality_plan_b",
                "delivered_quality_plan_c",
            )
            if isinstance((value := metrics.get(name)), int | float)
        ]
        if not delivered:
            continue
        revealed = max(delivered)
        bound = revealed if bound is None else max(bound, revealed)
    return bound
