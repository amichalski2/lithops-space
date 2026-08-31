from __future__ import annotations

import pytest
from lithops.application.executable_model_planning import ExecutableModelPlanner
from lithops.application.step_run import RunManager, StaticDecisionEngine
from lithops.benchmark.fake import FakeBenchmarkAdapter
from lithops.domain.models import ObservationSnapshot
from lithops.infrastructure.persistence.repositories import InMemoryRunRepository


class CrashBeforeFirstDecisionRepository(InMemoryRunRepository):
    def __init__(self) -> None:
        super().__init__()
        self.crash_once = True

    async def save_decision(self, decision):
        if self.crash_once:
            self.crash_once = False
            raise RuntimeError("simulated crash after executable fit")
        return await super().save_decision(decision)


def manager_with_executable_planning(repository=None):
    resolved_repository = repository or InMemoryRunRepository()
    executive = StaticDecisionEngine()
    planner = ExecutableModelPlanner(
        repository=resolved_repository,
        executive=executive,
        n_rollouts=5,
    )
    return RunManager(
        repository=resolved_repository,
        benchmark=FakeBenchmarkAdapter(),
        decision_engine=executive,
        planning_rollouts=5,
        executable_model_planner=planner,
    ), resolved_repository


def test_executable_history_exposes_observed_cost_and_capacity_without_future_data() -> None:
    row = ExecutableModelPlanner._history_row(
        ObservationSnapshot(
            day=35,
            cash=800_000.0,
            metrics={
                "revenue_weekly": 500.0,
                "active_customers": 80.0,
                "operations_spend": 4_000.0,
                "capacity_spend_weekly": 700.0,
                "capacity": 1_200.0,
                "operating_cost_per_customer_weekly": 23.0,
            },
        )
    )

    assert row["day"] == 35
    assert row["operations_spend"] == 4_000.0
    assert row["capacity_spend_weekly"] == 700.0
    assert row["capacity"] == 1_200.0
    assert row["operating_cost_per_customer_weekly"] == 23.0


@pytest.mark.asyncio
async def test_weekly_loop_commits_exact_executable_artifact_and_fit_lineage() -> None:
    manager, repository = manager_with_executable_planning()
    run = await manager.create_run(horizon_days=28)

    result = await manager.step_run(run.id, request_id="executable-week-0")
    replay = await manager.step_run(run.id, request_id="executable-week-0")
    prediction = await repository.get_prediction(result.decision.prediction_id)
    artifact = await repository.get_model_artifact(result.decision.model_artifact_id)
    fitted = await repository.get_fitted_model(run.id, result.decision.fitted_model_id)

    assert replay.replayed is True
    assert result.decision.model_artifact_hash == artifact.content_hash
    assert result.decision.fitted_state_hash == fitted.state_hash
    assert fitted.artifact_id == artifact.id
    assert prediction.model_artifact_id == artifact.id
    assert prediction.model_artifact_hash == artifact.content_hash
    assert prediction.fitted_model_id == fitted.id
    assert prediction.fitted_state_hash == fitted.state_hash
    assert any(
        reference.startswith(f"model-artifact:{artifact.id}:")
        for reference in prediction.evidence_references
    )

    second = await manager.step_run(run.id, request_id="executable-week-1")
    second_fitted = await repository.get_fitted_model(
        run.id, second.decision.fitted_model_id
    )
    assert second.decision.model_artifact_id == artifact.id
    assert second_fitted.id != fitted.id
    assert second_fitted.training_start_day == 0
    assert second_fitted.training_end_day == 7
    assert len(second_fitted.observation_ids) == 2


@pytest.mark.asyncio
async def test_recovery_reuses_persisted_fit_without_duplicate_week() -> None:
    repository = CrashBeforeFirstDecisionRepository()
    manager, _ = manager_with_executable_planning(repository)
    run = await manager.create_run(horizon_days=28)

    with pytest.raises(RuntimeError, match="after executable fit"):
        await manager.step_run(run.id, request_id="recover-fit")
    recovered = await manager.step_run(run.id, request_id="recover-fit")

    assert recovered.run.current_day == 7
    assert recovered.replayed is False
    assert len(await repository.list_decisions(run.id)) == 1
    assert len(await repository.list_predictions(run.id)) == 1


@pytest.mark.asyncio
async def test_a_failed_proposal_stage_degrades_the_week_instead_of_killing_the_run() -> None:
    """Regression for the run that a refused proposal killed.

    Without executive authority the proposal stage used to re-raise, the generic
    worker retry reproduced the same deterministic refusal, and the run was
    failed. A missing opinion is not a failed week: the deterministic pool still
    carries it, and the degradation is named rather than silent.
    """

    manager, repository = manager_with_executable_planning()

    async def failing_proposals(**kwargs):
        raise RuntimeError("proposal stage unavailable")

    manager.decision_engine.propose_actions = failing_proposals
    run = await manager.create_run(horizon_days=28)

    result = await manager.step_run(run.id, request_id="degraded-week-0")

    assert result.decision.action_plan is not None
    events = {event.type for event in await repository.list_events(run.id)}
    assert "decision.candidate_pool_degraded" in events
    degraded = next(
        event
        for event in await repository.list_events(run.id)
        if event.type == "decision.candidate_pool_degraded"
    )
    assert "executive_proposal_stage_failed" in degraded.payload["warning_codes"]


@pytest.mark.asyncio
async def test_refused_proposals_reach_the_pool_record_and_the_next_brief() -> None:
    """A refusal is carried by name to the only party that can replace it."""

    from lithops.domain.models import ProposalBatch, ProposalRejection

    manager, repository = manager_with_executable_planning()
    briefs: list[tuple[dict, ...] | None] = []

    async def refusing_proposals(*, run, observation, decision_history=(), **kwargs):
        briefs.append(kwargs.get("rejection_feedback"))
        return ProposalBatch(
            rejections=(
                ProposalRejection(
                    week=observation.day // 7,
                    candidate_index=0,
                    name="probe without exposure",
                    hypothesis_id="h_untested_regime",
                    stage="construction",
                    veto_codes=("acquisition_probe_missing",),
                    detail="targeted development programs require acquisition probe spend",
                ),
            )
        )

    manager.decision_engine.propose_actions = refusing_proposals
    run = await manager.create_run(horizon_days=28)

    await manager.step_run(run.id, request_id="refused-week-0")
    degraded = next(
        event
        for event in await repository.list_events(run.id)
        if event.type == "decision.candidate_pool_degraded"
    )
    rejected = degraded.payload["rejected"]
    assert any(
        item["violation_codes"] == ["acquisition_probe_missing"] for item in rejected
    )

    await manager.step_run(run.id, request_id="refused-week-1")

    assert briefs[0] is None or briefs[0] == ()
    feedback = briefs[-1]
    assert feedback, "the second week must see the first week's refusal"
    assert any(
        entry["veto_codes"] == ["acquisition_probe_missing"] for entry in feedback
    )
