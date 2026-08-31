from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from lithops.application.step_run import RunManager, StaticDecisionEngine
from lithops.benchmark.fake import FakeBenchmarkAdapter
from lithops.domain.models import RunRecord, RunStatus, utc_now
from lithops.infrastructure.persistence.repositories import InMemoryRunRepository
from lithops.worker import AutonomousRunWorker, LeaseUnavailableError


def make_worker(
    *,
    adapter: FakeBenchmarkAdapter | None = None,
    owner_id: str = "test-worker",
) -> tuple[
    AutonomousRunWorker,
    RunManager,
    InMemoryRunRepository,
    FakeBenchmarkAdapter,
]:
    repository = InMemoryRunRepository()
    benchmark = adapter or FakeBenchmarkAdapter()
    manager = RunManager(
        repository=repository,
        benchmark=benchmark,
        decision_engine=StaticDecisionEngine(),
        planning_rollouts=5,
    )
    worker = AutonomousRunWorker(
        manager=manager,
        repository=repository,
        owner_id=owner_id,
        lease_ttl_seconds=2,
        step_timeout_seconds=1.0,
        retry_backoff_seconds=0,
    )
    return worker, manager, repository, benchmark


@pytest.mark.asyncio
async def test_leases_are_exclusive_and_expired_ownership_can_be_reclaimed() -> None:
    repository = InMemoryRunRepository()
    run = await repository.create_run(RunRecord())
    now = utc_now()

    first = await repository.claim_run_lease(
        run.id,
        "worker-a",
        now=now,
        ttl_seconds=10,
    )
    blocked = await repository.claim_run_lease(
        run.id,
        "worker-b",
        now=now + timedelta(seconds=1),
        ttl_seconds=10,
    )
    reclaimed = await repository.claim_run_lease(
        run.id,
        "worker-b",
        now=now + timedelta(seconds=11),
        ttl_seconds=10,
    )

    assert first is not None
    assert blocked is None
    assert reclaimed is not None
    assert reclaimed.owner_id == "worker-b"
    assert await repository.renew_run_lease(
        run.id,
        first.token,
        now=now + timedelta(seconds=11),
        ttl_seconds=10,
    ) is None


@pytest.mark.asyncio
async def test_second_live_worker_cannot_claim_the_same_run() -> None:
    worker, manager, repository, _ = make_worker(owner_id="worker-a")
    run = await manager.create_run(horizon_days=14)
    await manager.start_run(run.id)
    lease = await repository.claim_run_lease(
        run.id,
        "holder",
        now=utc_now(),
        ttl_seconds=30,
    )
    assert lease is not None

    with pytest.raises(LeaseUnavailableError):
        await worker.run(run.id)


@pytest.mark.asyncio
async def test_accelerated_worker_completes_500_days_unattended() -> None:
    worker, manager, repository, adapter = make_worker()
    run = await manager.create_run(horizon_days=500)
    await manager.start_run(run.id)

    result = await worker.run(run.id)

    assert result.run.status == RunStatus.COMPLETED
    assert result.run.current_day == 500
    assert result.weeks_completed == 72
    assert adapter.advance_week_calls == 72
    assert len(await repository.list_decisions(run.id)) == 72


class PauseDuringAdvance(FakeBenchmarkAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.manager: RunManager | None = None
        self.run_id = None
        self.requested = False

    async def advance_week(self, session_id, *, rationale, forecasts):
        actual = await super().advance_week(
            session_id,
            rationale=rationale,
            forecasts=forecasts,
        )
        if not self.requested:
            self.requested = True
            assert self.manager is not None and self.run_id is not None
            await self.manager.request_pause(self.run_id)
        return actual


@pytest.mark.asyncio
async def test_pause_requested_mid_step_takes_effect_at_the_checkpoint() -> None:
    adapter = PauseDuringAdvance()
    worker, manager, _, _ = make_worker(adapter=adapter)
    run = await manager.create_run(horizon_days=21)
    adapter.manager = manager
    adapter.run_id = run.id
    await manager.start_run(run.id)

    paused = await worker.run(run.id)

    assert paused.run.status == RunStatus.PAUSED
    assert paused.run.current_day == 7
    assert adapter.advance_week_calls == 1

    await manager.resume_run(run.id)
    resumed = await worker.run(run.id, max_weeks=1)
    assert resumed.run.status == RunStatus.RUNNING
    assert resumed.run.current_day == 14


class SimulatedProcessCrash(BaseException):
    pass


class CrashAfterAdvanceOnce(FakeBenchmarkAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.crash_once = True

    async def advance_week(self, session_id, *, rationale, forecasts):
        actual = await super().advance_week(
            session_id,
            rationale=rationale,
            forecasts=forecasts,
        )
        if self.crash_once:
            self.crash_once = False
            raise SimulatedProcessCrash()
        return actual


@pytest.mark.asyncio
async def test_restarted_worker_reclaims_started_operation_without_repeating_reality() -> None:
    adapter = CrashAfterAdvanceOnce()
    first_worker, manager, repository, _ = make_worker(
        adapter=adapter,
        owner_id="worker-before-crash",
    )
    run = await manager.create_run(horizon_days=14)
    await manager.start_run(run.id)

    with pytest.raises(SimulatedProcessCrash):
        await first_worker.run(run.id)

    operation = await repository.get_operation(
        run.id,
        f"autonomous:{run.id}:week:0",
    )
    assert operation is not None and operation.status.value == "started"
    assert adapter.advance_week_calls == 1

    restarted = AutonomousRunWorker(
        manager=manager,
        repository=repository,
        owner_id="worker-after-crash",
        lease_ttl_seconds=2,
        step_timeout_seconds=1.0,
    )
    recovered = await restarted.run(run.id, max_weeks=1)

    assert recovered.run.current_day == 7
    assert adapter.advance_week_calls == 1
    assert adapter.execute_action_calls == 5
    assert len(await repository.list_decisions(run.id)) == 1
    reclaimed = await repository.get_operation(
        run.id,
        f"autonomous:{run.id}:week:0",
    )
    assert reclaimed is not None and reclaimed.attempts == 2


class SlowBenchmark(FakeBenchmarkAdapter):
    async def advance_week(self, session_id, *, rationale, forecasts):
        await asyncio.sleep(0.05)
        return await super().advance_week(
            session_id,
            rationale=rationale,
            forecasts=forecasts,
        )


@pytest.mark.asyncio
async def test_repeated_step_timeout_moves_run_to_failed() -> None:
    repository = InMemoryRunRepository()
    adapter = SlowBenchmark()
    manager = RunManager(
        repository=repository,
        benchmark=adapter,
        decision_engine=StaticDecisionEngine(),
        planning_rollouts=5,
    )
    worker = AutonomousRunWorker(
        manager=manager,
        repository=repository,
        owner_id="timeout-worker",
        lease_ttl_seconds=1,
        step_timeout_seconds=0.01,
        retry_backoff_seconds=0,
    )
    run = await manager.create_run(horizon_days=14)
    await manager.start_run(run.id)

    result = await worker.run(run.id)

    assert result.run.status == RunStatus.FAILED
    assert "after 2 attempts" in (result.run.failure_reason or "")
    assert adapter.advance_week_calls == 0


@pytest.mark.asyncio
async def test_lease_is_renewed_before_retrying_a_long_weekly_step() -> None:
    repository = InMemoryRunRepository()
    run = await repository.create_run(RunRecord(status=RunStatus.RUNNING))
    current = [datetime(2026, 1, 1, tzinfo=UTC)]

    class FailOnceManager:
        calls = 0

        async def step_run(self, run_id, *, request_id, recover_in_progress):
            self.calls += 1
            if self.calls == 1:
                current[0] += timedelta(seconds=1.5)
                raise RuntimeError("transient model timeout")
            current[0] += timedelta(seconds=1)
            return SimpleNamespace(run=await repository.get_run(run_id))

    worker = AutonomousRunWorker(
        manager=FailOnceManager(),  # type: ignore[arg-type]
        repository=repository,
        owner_id="long-retry-worker",
        lease_ttl_seconds=2,
        step_timeout_seconds=1.0,
        retry_backoff_seconds=0,
        clock=lambda: current[0],
    )

    result = await worker.run(run.id, max_weeks=1)

    assert result.weeks_completed == 1
    assert result.owner_id == "long-retry-worker"


@pytest.mark.asyncio
async def test_bankruptcy_is_a_terminal_worker_outcome() -> None:
    adapter = FakeBenchmarkAdapter(initial_cash=100.0, weekly_cash_change=-200.0)
    worker, manager, _, _ = make_worker(adapter=adapter)
    run = await manager.create_run(horizon_days=500)
    await manager.start_run(run.id)

    result = await worker.run(run.id)

    assert result.run.status == RunStatus.BANKRUPT
    assert result.run.current_day == 7
    assert result.weeks_completed == 1


@pytest.mark.asyncio
async def test_split_worker_run_resumes_to_twelve_weeks_without_duplicates() -> None:
    worker, manager, repository, adapter = make_worker(owner_id="first-process")
    run = await manager.create_run(horizon_days=84)
    await manager.start_run(run.id)
    checkpoint_days: list[int] = []

    async def checkpoint(saved: RunRecord, _: int) -> None:
        checkpoint_days.append(saved.current_day)

    first = await worker.run(run.id, max_weeks=5, on_checkpoint=checkpoint)
    restarted = AutonomousRunWorker(
        manager=manager,
        repository=repository,
        owner_id="second-process",
        lease_ttl_seconds=2,
        step_timeout_seconds=1.0,
    )
    second = await restarted.run(run.id, on_checkpoint=checkpoint)

    assert first.run.current_day == 35
    assert second.run.status == RunStatus.COMPLETED
    assert second.run.current_day == 84
    assert checkpoint_days == list(range(7, 85, 7))
    assert adapter.advance_week_calls == 12
    decisions = await repository.list_decisions(run.id)
    assert len(decisions) == 12
    assert len({decision.week for decision in decisions}) == 12
