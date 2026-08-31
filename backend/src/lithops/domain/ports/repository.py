"""Persistence contract for the weekly execution state machine."""

from datetime import datetime
from typing import Protocol
from uuid import UUID

from lithops.domain.models import (
    ActionReceipt,
    CandidateEvaluationRecord,
    DecisionRecord,
    EventRecord,
    RunLease,
    RunRecord,
    StepOperation,
    StepResult,
)


class RunRepository(Protocol):
    async def create_run(self, run: RunRecord) -> RunRecord: ...

    async def get_run(self, run_id: UUID) -> RunRecord: ...

    async def save_run(self, run: RunRecord, *, expected_version: int) -> RunRecord: ...

    async def claim_run_lease(
        self,
        run_id: UUID,
        owner_id: str,
        *,
        now: datetime,
        ttl_seconds: int,
    ) -> RunLease | None: ...

    async def renew_run_lease(
        self,
        run_id: UUID,
        token: UUID,
        *,
        now: datetime,
        ttl_seconds: int,
    ) -> RunLease | None: ...

    async def release_run_lease(self, run_id: UUID, token: UUID) -> bool: ...

    async def get_decision_for_week(
        self, run_id: UUID, week: int
    ) -> DecisionRecord | None: ...

    async def get_decision(self, run_id: UUID, decision_id: UUID) -> DecisionRecord: ...

    async def list_decisions(self, run_id: UUID) -> list[DecisionRecord]: ...

    async def list_candidate_simulations(
        self,
        run_id: UUID,
        decision_id: UUID,
    ) -> list[CandidateEvaluationRecord]: ...

    async def save_decision(self, decision: DecisionRecord) -> DecisionRecord: ...

    async def update_decision(self, decision: DecisionRecord) -> DecisionRecord: ...

    async def get_receipt(self, run_id: UUID, idempotency_key: str) -> ActionReceipt | None: ...

    async def save_receipt(self, receipt: ActionReceipt) -> ActionReceipt: ...

    async def list_receipts(self, decision_id: UUID) -> list[ActionReceipt]: ...

    async def append_event(self, event: EventRecord) -> EventRecord: ...

    async def list_events(self, run_id: UUID) -> list[EventRecord]: ...

    async def get_operation(self, run_id: UUID, request_id: str) -> StepOperation | None: ...

    async def start_operation(self, operation: StepOperation) -> StepOperation: ...

    async def complete_operation(
        self, run_id: UUID, request_id: str, result: StepResult
    ) -> StepOperation: ...

    async def fail_operation(
        self, run_id: UUID, request_id: str, error: str
    ) -> StepOperation: ...
