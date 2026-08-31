"""Run a short autonomous fake benchmark through live Supabase lease persistence."""

from __future__ import annotations

import asyncio
import json

from lithops.application.step_run import RunManager, StaticDecisionEngine
from lithops.benchmark.fake import FakeBenchmarkAdapter
from lithops.config import Settings
from lithops.infrastructure.persistence.repositories import SupabaseRunRepository
from lithops.worker import AutonomousRunWorker


async def verify() -> None:
    settings = Settings.from_env()
    if not settings.supabase_url or not settings.supabase_secret_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SECRET_KEY are required")

    repository = SupabaseRunRepository(
        url=settings.supabase_url,
        secret_key=settings.supabase_secret_key,
    )
    benchmark = FakeBenchmarkAdapter()
    try:
        manager = RunManager(
            repository=repository,
            benchmark=benchmark,
            decision_engine=StaticDecisionEngine(),
            planning_rollouts=5,
        )
        run = await manager.create_run(horizon_days=14)
        await manager.start_run(run.id)
        worker = AutonomousRunWorker(
            manager=manager,
            repository=repository,
            owner_id="supabase-live-verifier",
            lease_ttl_seconds=30,
            step_timeout_seconds=20.0,
        )
        result = await worker.run(run.id)
        if result.run.status.value != "completed" or result.run.current_day != 14:
            raise RuntimeError(f"unexpected terminal run state: {result.run}")

        events = await repository.list_events(run.id)
        print(
            json.dumps(
                {
                    "run_id": str(run.id),
                    "status": result.run.status,
                    "current_day": result.run.current_day,
                    "weeks_completed": result.weeks_completed,
                    "advance_week_calls": benchmark.advance_week_calls,
                    "lease_acquired_events": sum(
                        event.type == "worker.lease_acquired" for event in events
                    ),
                    "lease_released_events": sum(
                        event.type == "worker.lease_released" for event in events
                    ),
                },
                sort_keys=True,
            )
        )
    finally:
        await repository._client.aclose()


if __name__ == "__main__":
    asyncio.run(verify())
