from __future__ import annotations

import pytest
from lithops.application.executable_model_challenge import ExecutableModelChallenge
from lithops.application.step_run import RunManager, StaticDecisionEngine
from lithops.benchmark.learning_scenario import LearningScenarioBenchmarkAdapter
from lithops.infrastructure.persistence.repositories import InMemoryRunRepository
from lithops.model_runtime import TemporalEvaluationPolicy, TemporalModelEvaluator

from backend.tests.unit.test_executable_model_challenge import (
    ArtifactTrendRuntime,
    ExactTrendAuthor,
    FailOnceActivationRepository,
)
from backend.tests.unit.test_temporal_model_evaluation import TrendRuntime


class AlwaysFailPromotionRepository(InMemoryRunRepository):
    async def append_model_promotion_decision(self, decision):
        del decision
        raise RuntimeError("persistent promotion persistence failure")


@pytest.mark.asyncio
async def test_degraded_health_runs_executable_challenge_from_weekly_loop() -> None:
    repository = InMemoryRunRepository()
    author = ExactTrendAuthor()
    challenge = ExecutableModelChallenge(
        repository=repository,
        authors=(author,),
        evaluator=TemporalModelEvaluator(
            TemporalEvaluationPolicy(
                min_training_observations=2,
                n_rollouts=2,
                runtime_budget_ms=10_000,
            )
        ),
        baseline=TrendRuntime("health-trigger-baseline", -25_000.0),
        runtime_factory=lambda artifact: ArtifactTrendRuntime(
            artifact,
            weekly_cash_delta=-112_500.0,
        ),
    )
    manager = RunManager(
        repository=repository,
        benchmark=LearningScenarioBenchmarkAdapter(),
        decision_engine=StaticDecisionEngine(),
        executable_model_challenge=challenge,
        model_challenge_cooldown_days=28,
        planning_rollouts=5,
    )
    run = await manager.create_run(horizon_days=42)

    for week in range(5):
        await manager.step_run(run.id, request_id=f"executable-health-{week}")

    events = await repository.list_events(run.id)
    starts = [event for event in events if event.type == "executable_model_challenge.started"]
    completions = [
        event for event in events if event.type == "executable_model_challenge.completed"
    ]
    active = await repository.get_active_model(run.id)

    assert len(starts) == 1
    assert len(completions) == 1
    assert starts[0].payload["observation_count"] == 4
    assert completions[0].payload["disposition"] == "promoted"
    assert active is not None
    assert author.calls == 1


@pytest.mark.asyncio
async def test_weekly_retry_completes_same_challenge_after_activation_crash() -> None:
    repository = FailOnceActivationRepository()
    benchmark = LearningScenarioBenchmarkAdapter()
    author = ExactTrendAuthor()
    challenge = ExecutableModelChallenge(
        repository=repository,
        authors=(author,),
        evaluator=TemporalModelEvaluator(
            TemporalEvaluationPolicy(
                min_training_observations=2,
                n_rollouts=2,
                runtime_budget_ms=10_000,
            )
        ),
        baseline=TrendRuntime("health-retry-baseline", -25_000.0),
        runtime_factory=lambda artifact: ArtifactTrendRuntime(
            artifact,
            weekly_cash_delta=-112_500.0,
        ),
    )
    manager = RunManager(
        repository=repository,
        benchmark=benchmark,
        decision_engine=StaticDecisionEngine(),
        executable_model_challenge=challenge,
        model_challenge_cooldown_days=28,
        planning_rollouts=5,
    )
    run = await manager.create_run(horizon_days=42)
    for week in range(3):
        await manager.step_run(run.id, request_id=f"executable-retry-{week}")

    recovered = await manager.step_run(run.id, request_id="executable-retry-3")

    events = await repository.list_events(run.id)
    assert recovered.run.current_day == 28
    assert benchmark.advance_week_calls == 4
    assert author.calls == 1
    assert sum(event.type == "executable_model_challenge.started" for event in events) == 1
    assert sum(event.type == "executable_model_challenge.retrying" for event in events) == 1
    assert sum(event.type == "executable_model_challenge.completed" for event in events) == 1
    assert await repository.get_active_model(run.id) is not None


@pytest.mark.asyncio
async def test_persistent_challenge_error_is_contained_and_week_continues() -> None:
    repository = AlwaysFailPromotionRepository()
    benchmark = LearningScenarioBenchmarkAdapter()
    author = ExactTrendAuthor()
    challenge = ExecutableModelChallenge(
        repository=repository,
        authors=(author,),
        evaluator=TemporalModelEvaluator(
            TemporalEvaluationPolicy(
                min_training_observations=2,
                n_rollouts=2,
                runtime_budget_ms=10_000,
            )
        ),
        baseline=TrendRuntime("contained-internal-error-baseline", -25_000.0),
        runtime_factory=lambda artifact: ArtifactTrendRuntime(
            artifact,
            weekly_cash_delta=-112_500.0,
        ),
    )
    manager = RunManager(
        repository=repository,
        benchmark=benchmark,
        decision_engine=StaticDecisionEngine(),
        executable_model_challenge=challenge,
        model_challenge_cooldown_days=28,
        planning_rollouts=5,
    )
    run = await manager.create_run(horizon_days=42)
    for week in range(4):
        result = await manager.step_run(
            run.id,
            request_id=f"contained-internal-error-{week}",
        )

    events = await repository.list_events(run.id)
    failure = next(
        event
        for event in events
        if event.type == "executable_model_challenge.failed"
    )
    assert result.run.current_day == 28
    assert benchmark.advance_week_calls == 4
    assert author.calls == 1
    assert failure.payload["reason_code"] == "challenge_internal_error"
    assert failure.payload["failure_codes"] == ["challenge:RuntimeError"]
    assert failure.payload["contained"] is True


@pytest.mark.asyncio
async def test_failed_baseline_challenge_records_evidence_and_company_week_continues() -> None:
    repository = InMemoryRunRepository()
    challenge = ExecutableModelChallenge(
        repository=repository,
        authors=(ExactTrendAuthor(),),
        evaluator=TemporalModelEvaluator(
            TemporalEvaluationPolicy(
                min_training_observations=2,
                n_rollouts=2,
                runtime_budget_ms=10_000,
            )
        ),
        baseline=TrendRuntime(
            "failed-health-baseline",
            -25_000.0,
            fail_predict=True,
        ),
        runtime_factory=ArtifactTrendRuntime,
    )
    manager = RunManager(
        repository=repository,
        benchmark=LearningScenarioBenchmarkAdapter(),
        decision_engine=StaticDecisionEngine(),
        executable_model_challenge=challenge,
        model_challenge_cooldown_days=28,
        planning_rollouts=5,
    )
    run = await manager.create_run(horizon_days=42)

    for week in range(4):
        result = await manager.step_run(
            run.id,
            request_id=f"contained-baseline-failure-{week}",
        )

    events = await repository.list_events(run.id)
    failure = next(
        event
        for event in events
        if event.type == "executable_model_challenge.failed"
    )

    assert result.run.current_day == 28
    assert failure.payload["disposition"] == "no_update"
    assert failure.payload["contained"] is True
    assert failure.payload["failure_codes"] == [
        "champion:fold_0:RuntimeError",
        "baseline:fold_0:RuntimeError",
    ]
