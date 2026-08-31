"""Canonical economic units and invariant evidence shared by model runtimes."""

from __future__ import annotations

from enum import StrEnum
from math import isfinite

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RatePeriod(StrEnum):
    DAY = "day"
    WEEK = "week"
    MONTH_30_DAY = "month_30_day"
    YEAR_365_DAY = "year_365_day"

    @property
    def days(self) -> float:
        return {
            RatePeriod.DAY: 1.0,
            RatePeriod.WEEK: 7.0,
            RatePeriod.MONTH_30_DAY: 30.0,
            RatePeriod.YEAR_365_DAY: 365.0,
        }[self]


class Currency(StrEnum):
    USD = "USD"


class PeriodicMoney(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    amount: float
    currency: Currency = Currency.USD
    period: RatePeriod

    @model_validator(mode="after")
    def validate_finite_amount(self) -> PeriodicMoney:
        if not isfinite(self.amount):
            raise ValueError("periodic money amount must be finite")
        return self

    def per(self, period: RatePeriod) -> PeriodicMoney:
        return PeriodicMoney(
            amount=self.amount * period.days / self.period.days,
            currency=self.currency,
            period=period,
        )


class AccountingPeriod(BaseModel):
    """Cash bridge over one prediction horizon; all values use the same currency."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    period_days: int = Field(ge=1)
    starting_cash: float
    recognized_revenue: float = Field(ge=0.0)
    other_inflows: float = Field(default=0.0, ge=0.0)
    operating_cost: float = Field(ge=0.0)
    operations_spend: float = Field(default=0.0, ge=0.0)
    capacity_spend: float = Field(default=0.0, ge=0.0)
    marketing_spend: float = Field(ge=0.0)
    development_spend: float = Field(ge=0.0)
    other_outflows: float = Field(default=0.0, ge=0.0)
    ending_cash: float
    currency: Currency = Currency.USD

    @property
    def reconciled_ending_cash(self) -> float:
        return (
            self.starting_cash
            + self.recognized_revenue
            + self.other_inflows
            - self.operating_cost
            - self.operations_spend
            - self.capacity_spend
            - self.marketing_spend
            - self.development_spend
            - self.other_outflows
        )


class EconomicInvariantCode(StrEnum):
    ACCOUNTING_MISMATCH = "accounting_mismatch"
    CASH_SAMPLE_MISMATCH = "cash_sample_mismatch"
    HORIZON_MISMATCH = "horizon_mismatch"
    PRICE_BELOW_VARIABLE_COST = "price_below_variable_cost"
    PRICE_DEEPENS_VARIABLE_COST_BREACH = "price_deepens_variable_cost_breach"
    PRICE_BELOW_VARIABLE_COST_INHERITED = "price_below_variable_cost_inherited"
    TIER_COST_COVERAGE_PRESSURE = "tier_cost_coverage_pressure"
    PRICE_CHANGE_TOO_LARGE = "price_change_too_large"
    PRICE_BELOW_ABSOLUTE_FLOOR = "price_below_absolute_floor"
    LEAD_PROMOTION_TOO_LARGE = "lead_promotion_too_large"
    EXPIRED_COMMITMENT = "expired_commitment"
    NON_FINITE_VALUE = "non_finite_value"


class EconomicInvariantViolation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: EconomicInvariantCode
    path: str = Field(min_length=1, max_length=240)
    message: str = Field(min_length=1, max_length=1_000)
    expected: float | str | None = None
    actual: float | str | None = None


class EconomicInvariantReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    violations: tuple[EconomicInvariantViolation, ...] = ()
    warnings: tuple[EconomicInvariantViolation, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.violations
