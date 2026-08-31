"""Create a disposable fake-benchmark run against configured Supabase persistence."""

from __future__ import annotations

import asyncio
import json

from lithops.application.step_run import RunManager, StaticDecisionEngine
from lithops.benchmark.fake import FakeBenchmarkAdapter
from lithops.config import Settings
from lithops.infrastructure.persistence.repositories import SupabaseRunRepository


async def verify() -> None:
    settings = Settings.from_env()
    if not settings.supabase_url or not settings.supabase_secret_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SECRET_KEY are required")

    repository = SupabaseRunRepository(
        url=settings.supabase_url,
        secret_key=settings.supabase_secret_key,
    )
    try:
        manager = RunManager(
            repository=repository,
            benchmark=FakeBenchmarkAdapter(),
            decision_engine=StaticDecisionEngine(),
            planning_rollouts=20,
        )
        run = await manager.create_run(horizon_days=500)
        await manager.step_run(run.id, request_id="supabase-artifacts-week-0")
        await manager.step_run(run.id, request_id="supabase-artifacts-week-1")

        decisions = await repository.list_decisions(run.id)
        models = await repository.list_world_models(run.id)
        predictions = await repository.list_predictions(run.id)
        outcomes = await repository.list_prediction_outcomes(run.id)
        health = await repository.list_model_health_signals(run.id)
        relationships = await repository.list_world_model_relationships(models[0].id)
        candidates = await repository.list_candidate_simulations(run.id, decisions[0].id)

        expected_counts = (2, 2, 2, 1, 1)
        actual_counts = (
            len(decisions),
            len(models),
            len(predictions),
            len(outcomes),
            len(health),
        )
        if actual_counts != expected_counts or not relationships or len(candidates) < 3:
            raise RuntimeError(f"unexpected persisted artifact counts: {actual_counts}")

        print(
            json.dumps(
                {
                    "run_id": str(run.id),
                    "decisions": len(decisions),
                    "world_models": len(models),
                    "relationships": len(relationships),
                    "predictions": len(predictions),
                    "actuals": len(outcomes),
                    "candidate_simulations": len(candidates),
                    "model_health_signals": len(health),
                },
                sort_keys=True,
            )
        )
    finally:
        await repository._client.aclose()


if __name__ == "__main__":
    asyncio.run(verify())
