from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(slots=True)
class _FakeSession:
    session_id: str
    horizon_days: int
    day: int = 0
    cash: float = 1_000_000.0
    # The configuration the agent has actually set, so an observation reflects
    # its own executed actions rather than a fixed fiction.
    configuration: dict[str, float] = field(default_factory=dict)
    action_receipts: dict[str, ActionReceipt] = field(default_factory=dict)


class FakeBenchmarkAdapter:
    """Deterministic external environment used to verify orchestration semantics."""

    def __init__(
        self,
        *,
        initial_cash: float = 1_000_000.0,
        weekly_cash_change: float = 12_500.0,
    ) -> None:
        self._sessions: dict[str, _FakeSession] = {}
        self._session_by_run: dict[UUID, str] = {}
        self.initial_cash = initial_cash
        self.weekly_cash_change = weekly_cash_change
        self.create_session_calls = 0
        self.advance_week_calls = 0
        self.execute_action_calls = 0

    async def create_session(self, run_id: UUID, *, days: int) -> str:
        if run_id in self._session_by_run:
            return self._session_by_run[run_id]

        self.create_session_calls += 1
        session_id = f"fake-{run_id}"
        self._sessions[session_id] = _FakeSession(
            session_id=session_id,
            horizon_days=days,
            cash=self.initial_cash,
        )
        self._session_by_run[run_id] = session_id
        return session_id

    async def observe_status(self, session_id: str) -> ObservationSnapshot:
        session = self._get_session(session_id)
        return ObservationSnapshot(
            day=session.day,
            cash=session.cash,
            metrics={
                "active_customers": session.day * 3,
                "weekly_revenue": max(session.day, 1) * 250,
                "source": "fake_benchmark",
                **session.configuration,
            },
        )

    async def query_readonly(self, session_id: str, sql: str) -> list[dict[str, Any]]:
        session = self._get_session(session_id)
        validate_readonly_sql(sql)
        return [{"day": session.day, "cash": session.cash}]

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

        self.execute_action_calls += 1
        session.configuration.update(_configuration_effect(command))
        cost = float(command.arguments.get("weekly_spend", 0))
        if cost < 0:
            raise BenchmarkContractError("weekly_spend must not be negative")
        session.cash -= cost

        receipt = ActionReceipt(
            run_id=run_id,
            decision_id=decision_id,
            idempotency_key=command.idempotency_key,
            tool=command.tool,
            semantic_command_hash=command.semantic_hash,
            status=ReceiptStatus.EXECUTED,
            external_reference=f"{session_id}:{command.idempotency_key}",
            result={"applied": True, "cash_cost": cost},
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

        self.advance_week_calls += 1
        session.day = min(session.day + 7, session.horizon_days)
        session.cash += self.weekly_cash_change
        return await self.observe_status(session_id)

    def _get_session(self, session_id: str) -> _FakeSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise BenchmarkContractError(f"unknown benchmark session: {session_id}") from exc


def _configuration_effect(command: ActionCommand) -> dict[str, float]:
    """How one executed setter shows up in the next observation."""

    def number(key: str) -> float:
        value = command.arguments.get(key)
        return float(value) if isinstance(value, int | float) else 0.0

    if command.tool == "set_usage_quotas":
        return {f"usage_quota_{plan.lower()}": number(plan) for plan in "ABC"}
    if command.tool == "set_prices":
        return {
            **{f"price_{plan.lower()}": number(plan) for plan in "ABC"},
            **{f"configured_price_{plan.lower()}": number(plan) for plan in "ABC"},
        }
    if command.tool == "set_model_tiers":
        return {f"model_tier_{plan.lower()}": number(plan) for plan in "ABC"}
    if command.tool == "set_capacity_tier":
        return {"capacity_tier": number("tier")}
    if command.tool == "set_promotion":
        return {"recurring_promotion_monthly": number("global_promotion")}
    if command.tool == "set_ads_strength":
        return {"ads_strength": number("global_strength")}
    if command.tool == "set_lead_promotion":
        return {"lead_promotion_monthly": number("global_promotion")}
    if command.tool == "set_daily_spend":
        return {
            "operations_spend": number("operations") * 7.0,
            "development_spend": number("development") * 7.0,
        }
    if command.tool in {"set_targeted_ad_spend", "set_targeted_ops_spend"}:
        spend = command.arguments.get("targeted_spend")
        if not isinstance(spend, dict):
            return {}
        total = 0.0
        for value in spend.values():
            if isinstance(value, dict):
                total += sum(
                    float(amount)
                    for amount in value.values()
                    if isinstance(amount, int | float)
                )
            elif isinstance(value, int | float):
                total += float(value)
        key = (
            "marketing_spend"
            if command.tool == "set_targeted_ad_spend"
            else "targeted_ops_spend"
        )
        return {key: total * 7.0}
    return {}
