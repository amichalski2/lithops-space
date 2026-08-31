"""Contract for the external company simulation environment."""

from typing import Any, Protocol
from uuid import UUID

from lithops.domain.models import (
    ActionCommand,
    ActionReceipt,
    CashForecasts,
    ObservationSnapshot,
)


class BenchmarkPort(Protocol):
    async def create_session(self, run_id: UUID, *, days: int) -> str: ...

    async def observe_status(self, session_id: str) -> ObservationSnapshot: ...

    async def query_readonly(self, session_id: str, sql: str) -> list[dict[str, Any]]: ...

    async def collect_weekly_evidence(
        self, session_id: str, observation: ObservationSnapshot
    ) -> object: ...

    async def execute_action(
        self,
        session_id: str,
        *,
        run_id: UUID,
        decision_id: UUID,
        command: ActionCommand,
    ) -> ActionReceipt: ...

    async def advance_week(
        self,
        session_id: str,
        *,
        rationale: str,
        forecasts: CashForecasts,
    ) -> ObservationSnapshot: ...
