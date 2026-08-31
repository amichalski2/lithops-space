"""Independent economic validation for agent-authored model output."""

from __future__ import annotations

from math import isclose, isfinite

from lithops.domain.economics import (
    EconomicInvariantCode,
    EconomicInvariantReport,
    EconomicInvariantViolation,
)
from lithops.domain.executable_model import ModelOutcomeDistribution


def evaluate_model_outcomes(
    distribution: ModelOutcomeDistribution,
    *,
    absolute_tolerance: float = 0.01,
) -> EconomicInvariantReport:
    violations: list[EconomicInvariantViolation] = []
    for sample in distribution.samples:
        path = f"samples.{sample.rollout_index}.{sample.horizon_days}"
        values = (
            sample.cash,
            sample.revenue_weekly,
            sample.customers,
            sample.churn_rate,
            sample.accounting.starting_cash,
            sample.accounting.recognized_revenue,
            sample.accounting.other_inflows,
            sample.accounting.operating_cost,
            sample.accounting.operations_spend,
            sample.accounting.marketing_spend,
            sample.accounting.development_spend,
            sample.accounting.other_outflows,
            sample.accounting.ending_cash,
        )
        if not all(isfinite(value) for value in values):
            violations.append(
                EconomicInvariantViolation(
                    code=EconomicInvariantCode.NON_FINITE_VALUE,
                    path=path,
                    message="model outcome contains a non-finite economic value",
                )
            )
            continue
        if sample.accounting.period_days != sample.horizon_days:
            violations.append(
                EconomicInvariantViolation(
                    code=EconomicInvariantCode.HORIZON_MISMATCH,
                    path=f"{path}.accounting.period_days",
                    message="accounting period must equal the prediction horizon",
                    expected=float(sample.horizon_days),
                    actual=float(sample.accounting.period_days),
                )
            )
        if not isclose(
            sample.cash,
            sample.accounting.ending_cash,
            rel_tol=0.0,
            abs_tol=absolute_tolerance,
        ):
            violations.append(
                EconomicInvariantViolation(
                    code=EconomicInvariantCode.CASH_SAMPLE_MISMATCH,
                    path=f"{path}.cash",
                    message="sample cash must equal accounting ending cash",
                    expected=sample.accounting.ending_cash,
                    actual=sample.cash,
                )
            )
        reconciled = sample.accounting.reconciled_ending_cash
        if not isclose(
            reconciled,
            sample.accounting.ending_cash,
            rel_tol=0.0,
            abs_tol=absolute_tolerance,
        ):
            violations.append(
                EconomicInvariantViolation(
                    code=EconomicInvariantCode.ACCOUNTING_MISMATCH,
                    path=f"{path}.accounting.ending_cash",
                    message="cash bridge does not reconcile declared inflows and outflows",
                    expected=reconciled,
                    actual=sample.accounting.ending_cash,
                )
            )
    return EconomicInvariantReport(violations=tuple(violations))
