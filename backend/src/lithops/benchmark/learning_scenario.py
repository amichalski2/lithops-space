"""Deterministic fake company trajectory for the measurable-learning demo."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from lithops.domain.errors import BenchmarkContractError
from lithops.domain.models import (
    ActionCommand,
    ActionReceipt,
    CashForecasts,
    ObservationSnapshot,
    ReceiptStatus,
)
from lithops.infrastructure.security.sql_guard import validate_readonly_sql

SCENARIO_NAME = "paid-acquisition-saturation-shock-v1"
SCENARIO_START = datetime(2026, 8, 25, 9, 0, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class ScenarioPoint:
    day: int
    cash: float
    customers: float
    weekly_revenue: float
    weekly_acquisition: float
    churn_rate: float


LEARNING_SHOCK_TRAJECTORY: tuple[ScenarioPoint, ...] = (
    ScenarioPoint(0, 1_000_000.0, 1_000.0, 50_000.0, 80.0, 0.040),
    ScenarioPoint(7, 850_000.0, 980.0, 48_000.0, 50.0, 0.050),
    ScenarioPoint(14, 720_000.0, 940.0, 44_000.0, 35.0, 0.060),
    ScenarioPoint(21, 550_000.0, 880.0, 38_000.0, 25.0, 0.070),
    ScenarioPoint(28, 550_000.0, 880.0, 32_000.0, 18.0, 0.080),
    ScenarioPoint(35, 500_000.0, 760.0, 28_000.0, 12.0, 0.090),
    ScenarioPoint(42, 470_000.0, 720.0, 25_000.0, 10.0, 0.100),
)


@dataclass(slots=True)
class _ScenarioSession:
    session_id: str
    horizon_days: int
    point_index: int = 0
    price_per_customer_weekly: float = 50.0
    marketing_spend: float = 10_000.0
    development_spend: float = 5_000.0
    targeted_development_spend: float = 0.0
    model_tier_a: int = 1
    model_tier_b: int = 1
    model_tier_c: int = 1
    # None: this legacy scenario models no allowance information at all. A zero
    # here would instead claim a modeled zero allowance, which serves nothing
    # and deadens the whole acquisition path under the None-vs-0 semantics.
    usage_quota_a: float | None = None
    usage_quota_b: float | None = None
    usage_quota_c: float | None = None
    capacity_tier: int = 0
    recurring_promotion_monthly: float = 0.0
    ads_strength: float = 0.0
    targeted_ops_spend: float = 0.0
    social_posts_weekly: float = 0.0
    action_receipts: dict[str, ActionReceipt] = field(default_factory=dict)


class LearningScenarioBenchmarkAdapter:
    """A transparent test harness; never presented as primary CEO-Bench evidence.

    The external trajectory represents a demand regime where paid acquisition has become
    materially less effective than the generic priors imply. Lithops only receives the
    normalized observations and its own action receipts, not the scenario explanation.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, _ScenarioSession] = {}
        self._session_by_run: dict[UUID, str] = {}
        self.create_session_calls = 0
        self.advance_week_calls = 0
        self.execute_action_calls = 0

    async def create_session(self, run_id: UUID, *, days: int) -> str:
        if days > LEARNING_SHOCK_TRAJECTORY[-1].day:
            raise BenchmarkContractError(
                f"learning scenario supports at most {LEARNING_SHOCK_TRAJECTORY[-1].day} days"
            )
        existing = self._session_by_run.get(run_id)
        if existing is not None:
            return existing
        self.create_session_calls += 1
        session_id = f"learning-scenario-{run_id}"
        self._sessions[session_id] = _ScenarioSession(
            session_id=session_id,
            horizon_days=days,
        )
        self._session_by_run[run_id] = session_id
        return session_id

    async def observe_status(self, session_id: str) -> ObservationSnapshot:
        session = self._get_session(session_id)
        point = LEARNING_SHOCK_TRAJECTORY[session.point_index]
        return ObservationSnapshot(
            day=point.day,
            cash=point.cash,
            metrics={
                "active_customers": point.customers,
                "weekly_revenue": point.weekly_revenue,
                "weekly_acquisition": point.weekly_acquisition,
                "churn_rate": point.churn_rate,
                "price_per_customer_weekly": session.price_per_customer_weekly,
                "marketing_spend": session.marketing_spend,
                "development_spend": session.development_spend,
                "targeted_development_spend": session.targeted_development_spend,
                "product_quality": 0.60,
                "model_tier_a": session.model_tier_a,
                "model_tier_b": session.model_tier_b,
                "model_tier_c": session.model_tier_c,
                "usage_quota_a": session.usage_quota_a,
                "usage_quota_b": session.usage_quota_b,
                "usage_quota_c": session.usage_quota_c,
                "capacity_tier": session.capacity_tier,
                "recurring_promotion_monthly": session.recurring_promotion_monthly,
                "ads_strength": session.ads_strength,
                "targeted_ops_spend": session.targeted_ops_spend,
                "social_posts_weekly": session.social_posts_weekly,
                "capacity": 5_000.0,
                "reputation": 0.55,
                "source": "deterministic_learning_scenario",
            },
            observed_at=SCENARIO_START + timedelta(days=point.day),
        )

    async def query_readonly(self, session_id: str, sql: str) -> list[dict[str, Any]]:
        validate_readonly_sql(sql)
        observation = await self.observe_status(session_id)
        return [{"day": observation.day, "cash": observation.cash}]

    async def execute_action(
        self,
        session_id: str,
        *,
        run_id: UUID,
        decision_id: UUID,
        command: ActionCommand,
    ) -> ActionReceipt:
        session = self._get_session(session_id)
        existing = session.action_receipts.get(command.idempotency_key)
        if existing is not None:
            return existing.model_copy(update={"status": ReceiptStatus.REPLAYED})

        result: dict[str, Any]
        if command.tool == "set_prices":
            prices = [self._nonnegative_number(command.arguments, key) for key in ("A", "B", "C")]
            if any(price <= 0 for price in prices):
                raise BenchmarkContractError("scenario prices must be positive")
            session.price_per_customer_weekly = sum(prices) / len(prices) * 7.0 / 30.0
            result = {"applied": True, "average_weekly_price": session.price_per_customer_weekly}
        elif command.tool == "set_model_tiers":
            tiers = [
                int(self._nonnegative_number(command.arguments, key))
                for key in ("A", "B", "C")
            ]
            if any(not 1 <= tier <= 5 for tier in tiers):
                raise BenchmarkContractError("scenario model tiers must be between 1 and 5")
            session.model_tier_a, session.model_tier_b, session.model_tier_c = tiers
            result = {"applied": True, "model_tiers": tiers}
        elif command.tool == "set_daily_spend":
            operations = self._nonnegative_number(command.arguments, "operations")
            development = self._nonnegative_number(command.arguments, "development")
            session.marketing_spend = operations * 7
            session.development_spend = development * 7
            result = {
                "applied": True,
                "weekly_marketing_spend": session.marketing_spend,
                "weekly_development_spend": session.development_spend,
            }
        elif command.tool == "set_targeted_ad_spend":
            targeted = command.arguments.get("targeted_spend")
            if not isinstance(targeted, dict):
                raise BenchmarkContractError("scenario targeted spend must be a mapping")
            daily_total = 0.0
            for groups in targeted.values():
                if not isinstance(groups, dict):
                    raise BenchmarkContractError("scenario targeted groups must be mappings")
                for amount in groups.values():
                    if not isinstance(amount, int | float) or amount < 0:
                        raise BenchmarkContractError("scenario targeted spend must be non-negative")
                    daily_total += float(amount)
            session.marketing_spend = daily_total * 7.0
            result = {
                "applied": True,
                "weekly_marketing_spend": session.marketing_spend,
            }
        elif command.tool == "set_targeted_dev_spend":
            targeted = command.arguments.get("targeted_spend")
            if not isinstance(targeted, dict):
                raise BenchmarkContractError(
                    "scenario targeted development spend must be a mapping"
                )
            daily_total = 0.0
            for amount in targeted.values():
                if not isinstance(amount, int | float) or amount < 0:
                    raise BenchmarkContractError(
                        "scenario targeted development spend must be non-negative"
                    )
                daily_total += float(amount)
            session.targeted_development_spend = daily_total * 7.0
            result = {
                "applied": True,
                "weekly_targeted_development_spend": (
                    session.targeted_development_spend
                ),
            }
        elif command.tool == "set_usage_quotas":
            quotas = [
                self._nonnegative_number(command.arguments, key) for key in ("A", "B", "C")
            ]
            session.usage_quota_a, session.usage_quota_b, session.usage_quota_c = quotas
            result = {"applied": True, "usage_quotas": quotas}
        elif command.tool == "set_capacity_tier":
            tier = int(self._nonnegative_number(command.arguments, "tier"))
            session.capacity_tier = tier
            result = {"applied": True, "capacity_tier": tier}
        elif command.tool == "set_promotion":
            promotion = self._nonnegative_number(command.arguments, "global_promotion")
            session.recurring_promotion_monthly = promotion
            result = {"applied": True, "recurring_promotion_monthly": promotion}
        elif command.tool == "set_ads_strength":
            strength = self._nonnegative_number(command.arguments, "global_strength")
            if strength > 1.0:
                raise BenchmarkContractError("scenario ads strength must be within 0-1")
            session.ads_strength = strength
            result = {"applied": True, "ads_strength": strength}
        elif command.tool == "set_targeted_ops_spend":
            targeted = command.arguments.get("targeted_spend")
            if not isinstance(targeted, dict):
                raise BenchmarkContractError("scenario targeted ops spend must be a mapping")
            daily_total = 0.0
            for amount in targeted.values():
                if not isinstance(amount, int | float) or amount < 0:
                    raise BenchmarkContractError(
                        "scenario targeted ops spend must be non-negative"
                    )
                daily_total += float(amount)
            session.targeted_ops_spend = daily_total * 7.0
            result = {"applied": True, "weekly_targeted_ops_spend": session.targeted_ops_spend}
        elif command.tool == "post_social_media":
            content = str(command.arguments.get("content") or "").strip()
            if not content:
                raise BenchmarkContractError("scenario social post requires content")
            session.social_posts_weekly += 1.0
            result = {"applied": True, "posted": True}
        else:
            raise BenchmarkContractError(f"scenario tool is not allowlisted: {command.tool}")

        self.execute_action_calls += 1
        receipt = ActionReceipt(
            run_id=run_id,
            decision_id=decision_id,
            idempotency_key=command.idempotency_key,
            tool=command.tool,
            semantic_command_hash=command.semantic_hash,
            status=ReceiptStatus.EXECUTED,
            external_reference=f"{session_id}:{command.idempotency_key}",
            result=result,
            created_at=SCENARIO_START
            + timedelta(days=LEARNING_SHOCK_TRAJECTORY[session.point_index].day),
        )
        session.action_receipts[command.idempotency_key] = receipt
        return receipt

    async def advance_week(
        self,
        session_id: str,
        *,
        rationale: str,
        forecasts: CashForecasts,
    ) -> ObservationSnapshot:
        session = self._get_session(session_id)
        if not rationale.strip():
            raise BenchmarkContractError("weekly rationale must not be empty")
        forecasts.ordered()
        if session.point_index + 1 >= len(LEARNING_SHOCK_TRAJECTORY):
            raise BenchmarkContractError("learning scenario trajectory is exhausted")
        next_point = LEARNING_SHOCK_TRAJECTORY[session.point_index + 1]
        if next_point.day > session.horizon_days:
            raise BenchmarkContractError("learning scenario cannot advance beyond the run horizon")
        session.point_index += 1
        self.advance_week_calls += 1
        return await self.observe_status(session_id)

    @staticmethod
    def _nonnegative_number(arguments: dict[str, Any], key: str) -> float:
        value = arguments.get(key, 0.0)
        if not isinstance(value, int | float) or isinstance(value, bool) or value < 0:
            raise BenchmarkContractError(f"scenario argument {key} must be non-negative")
        return float(value)

    def _get_session(self, session_id: str) -> _ScenarioSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise BenchmarkContractError(f"unknown benchmark session: {session_id}") from exc
