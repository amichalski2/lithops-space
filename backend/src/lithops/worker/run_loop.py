"""Lease-aware autonomous loop around the recovery-safe weekly use case."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

from lithops.application.step_run import RunManager, RunStateError
from lithops.domain.models import (
    EventRecord,
    OperationStatus,
    RunLease,
    RunRecord,
    RunStatus,
    utc_now,
)
from lithops.domain.ports import RunRepository


class LeaseUnavailableError(RuntimeError):
    """Another live worker owns the requested run."""


class LeaseLostError(RuntimeError):
    """The worker no longer has authority to continue the run."""


@dataclass(frozen=True, slots=True)
class WorkerRunResult:
    run: RunRecord
    weeks_completed: int
    owner_id: str


class AutonomousRunWorker:
    def __init__(
        self,
        *,
        manager: RunManager,
        repository: RunRepository,
        owner_id: str | None = None,
        lease_ttl_seconds: int = 180,
        step_timeout_seconds: float = 120.0,
        max_step_attempts: int = 2,
        retry_backoff_seconds: float = 1.0,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        if lease_ttl_seconds <= step_timeout_seconds:
            raise ValueError("lease TTL must exceed the weekly step timeout")
        if max_step_attempts != 2:
            raise ValueError("P0 worker requires exactly one retry")
        if retry_backoff_seconds < 0:
            raise ValueError("retry backoff cannot be negative")
        self.manager = manager
        self.repository = repository
        self.owner_id = owner_id or f"worker-{uuid4()}"
        self.lease_ttl_seconds = lease_ttl_seconds
        self.step_timeout_seconds = step_timeout_seconds
        self.max_step_attempts = max_step_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.clock = clock

    async def run(
        self,
        run_id: UUID,
        *,
        max_weeks: int | None = None,
        on_checkpoint: Callable[[RunRecord, int], Awaitable[None]] | None = None,
    ) -> WorkerRunResult:
        if max_weeks is not None and max_weeks < 1:
            raise ValueError("max_weeks must be positive")
        lease = await self.repository.claim_run_lease(
            run_id,
            self.owner_id,
            now=self.clock(),
            ttl_seconds=self.lease_ttl_seconds,
        )
        if lease is None:
            raise LeaseUnavailableError(f"run lease is already held: {run_id}")

        weeks_completed = 0
        await self._event(
            run_id,
            "worker.lease_acquired",
            {"owner_id": self.owner_id},
        )
        try:
            while True:
                run = await self.repository.get_run(run_id)
                terminal = await self._stop_if_requested(run)
                if terminal is not None:
                    return WorkerRunResult(terminal, weeks_completed, self.owner_id)

                lease = await self._renew(lease)
                request_id = f"autonomous:{run_id}:week:{run.current_day // 7}"
                result = None
                last_error: Exception | None = None
                for attempt in range(1, self.max_step_attempts + 1):
                    try:
                        async with asyncio.timeout(self.step_timeout_seconds):
                            result = await self.manager.step_run(
                                run_id,
                                request_id=request_id,
                                recover_in_progress=True,
                            )
                        break
                    except Exception as exc:
                        last_error = exc
                        await self._fail_started_operation(run_id, request_id, str(exc))
                        if attempt < self.max_step_attempts:
                            await self._event(
                                run_id,
                                "worker.step_retrying",
                                {
                                    "request_id": request_id,
                                    "attempt": attempt + 1,
                                    "error": str(exc),
                                },
                            )
                            # A retry starts a new potentially long operation. Refresh the
                            # lease first so the combined duration of both attempts cannot
                            # make an otherwise healthy worker lose ownership at checkpoint.
                            lease = await self._renew(lease)
                            if self.retry_backoff_seconds:
                                await asyncio.sleep(self.retry_backoff_seconds)

                if result is None:
                    reason = (
                        f"weekly step failed after {self.max_step_attempts} attempts: "
                        f"{last_error}"
                    )
                    failed = await self.manager.fail_run(run_id, reason=reason)
                    return WorkerRunResult(failed, weeks_completed, self.owner_id)

                weeks_completed += 1
                if on_checkpoint is not None:
                    await on_checkpoint(result.run, weeks_completed)
                lease = await self._renew(lease)
                terminal = await self._stop_if_requested(result.run)
                if terminal is not None:
                    return WorkerRunResult(terminal, weeks_completed, self.owner_id)
                if max_weeks is not None and weeks_completed >= max_weeks:
                    return WorkerRunResult(result.run, weeks_completed, self.owner_id)
        finally:
            released = await self.repository.release_run_lease(run_id, lease.token)
            if released:
                await self._event(
                    run_id,
                    "worker.lease_released",
                    {"owner_id": self.owner_id},
                )

    async def _renew(self, lease: RunLease) -> RunLease:
        renewed = await self.repository.renew_run_lease(
            lease.run_id,
            lease.token,
            now=self.clock(),
            ttl_seconds=self.lease_ttl_seconds,
        )
        if renewed is None:
            raise LeaseLostError(f"worker lost run lease: {lease.run_id}")
        return renewed

    async def _stop_if_requested(self, run: RunRecord) -> RunRecord | None:
        if run.status == RunStatus.CREATED:
            raise RunStateError("run must be started before an autonomous worker can claim it")
        if run.status == RunStatus.PAUSING:
            request_id = f"autonomous:{run.id}:week:{run.current_day // 7}"
            operation = await self.repository.get_operation(run.id, request_id)
            if operation is not None and operation.status != OperationStatus.COMPLETED:
                return None
            return await self.manager.complete_pause(run.id)
        if run.status in {
            RunStatus.PAUSED,
            RunStatus.COMPLETED,
            RunStatus.BANKRUPT,
            RunStatus.FAILED,
        }:
            return run
        return None

    async def _fail_started_operation(
        self,
        run_id: UUID,
        request_id: str,
        error: str,
    ) -> None:
        operation = await self.repository.get_operation(run_id, request_id)
        if operation is not None and operation.status == OperationStatus.STARTED:
            await self.repository.fail_operation(run_id, request_id, error)

    async def _event(self, run_id: UUID, event_type: str, payload: dict) -> EventRecord:
        return await self.repository.append_event(
            EventRecord(run_id=run_id, type=event_type, payload=payload)
        )
