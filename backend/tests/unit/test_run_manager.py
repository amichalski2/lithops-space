from __future__ import annotations

import pytest
from lithops.application.step_run import RunManager, RunStateError, StaticDecisionEngine
from lithops.benchmark.fake import FakeBenchmarkAdapter
from lithops.domain.errors import BenchmarkContractError
from lithops.domain.models import CashForecast, CashForecasts, DecisionStatus
from lithops.infrastructure.persistence.repositories import InMemoryRunRepository
from lithops.infrastructure.security.sql_guard import validate_readonly_sql
from lithops.simulator.models import SimulationAction
from lithops.simulator.strategy_search import (
    CandidateSimulation,
    NoViableStrategyError,
    RobustnessLevel,
)
from pydantic import ValidationError


def make_manager(
    benchmark: FakeBenchmarkAdapter | None = None,
) -> tuple[RunManager, FakeBenchmarkAdapter, InMemoryRunRepository]:
    adapter = benchmark or FakeBenchmarkAdapter()
    repository = InMemoryRunRepository()
    manager = RunManager(
        repository=repository,
        benchmark=adapter,
        decision_engine=StaticDecisionEngine(),
    )
    return manager, adapter, repository


def test_cash_forecasts_require_all_horizons_and_valid_intervals() -> None:
    with pytest.raises(ValidationError):
        CashForecasts(
            items=[
                CashForecast(horizon_days=7, point=100, lower=90, upper=110),
                CashForecast(horizon_days=28, point=110, lower=90, upper=130),
                CashForecast(horizon_days=84, point=120, lower=100, upper=140),
                CashForecast(horizon_days=84, point=130, lower=100, upper=160),
            ]
        )

    with pytest.raises(ValidationError):
        CashForecast(horizon_days=7, point=100, lower=101, upper=110)


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE customers SET active = false",
        "WITH removed AS (DELETE FROM customers RETURNING *) SELECT * FROM removed",
        "SELECT 1; DROP TABLE customers",
        "PRAGMA table_info(customers)",
    ],
)
def test_readonly_sql_guard_rejects_writes(sql: str) -> None:
    with pytest.raises(BenchmarkContractError):
        validate_readonly_sql(sql)


def test_readonly_sql_guard_accepts_select_and_with() -> None:
    assert validate_readonly_sql("SELECT day, cash FROM ledger;") == (
        "SELECT day, cash FROM ledger"
    )
    assert validate_readonly_sql(
        "WITH recent AS (SELECT * FROM ledger) SELECT * FROM recent"
    ).startswith("WITH")


class MismatchedSemanticReceiptAdapter(FakeBenchmarkAdapter):
    async def execute_action(self, session_id, *, run_id, decision_id, command):
        receipt = await super().execute_action(
            session_id,
            run_id=run_id,
            decision_id=decision_id,
            command=command,
        )
        return receipt.model_copy(update={"semantic_command_hash": "0" * 64})


@pytest.mark.asyncio
async def test_week_does_not_advance_when_receipt_does_not_match_selected_action() -> None:
    manager, adapter, _ = make_manager(MismatchedSemanticReceiptAdapter())
    run = await manager.create_run()

    with pytest.raises(RunStateError, match="semantic hash"):
        await manager.step_run(run.id, request_id="mismatched-action")

    assert adapter.advance_week_calls == 0


@pytest.mark.asyncio
async def test_weekly_step_is_idempotent_for_the_same_request_key() -> None:
    manager, adapter, repository = make_manager()
    run = await manager.create_run()

    first = await manager.step_run(run.id, request_id="week-one")
    replay = await manager.step_run(run.id, request_id="week-one")

    assert first.run.current_day == 7
    assert first.decision.status == DecisionStatus.COMMITTED
    assert first.replayed is False
    assert replay.replayed is True
    assert replay.run.current_day == 7
    assert adapter.create_session_calls == 1
    assert adapter.execute_action_calls == 5
    assert adapter.advance_week_calls == 1

    events = await repository.list_events(run.id)
    assert [event.type for event in events] == [
        "run.created",
        "benchmark.session_created",
        "world_model.created",
        "decision.exploration_assessed",
        "decision.prepared",
        "prediction.committed",
        "action.executed",
        "action.executed",
        "action.executed",
        "action.executed",
        "action.executed",
        "decision.committed",
    ]
    assert first.decision.world_model_version_id is not None
    assert first.decision.prediction_id is not None
    assert len(first.decision.candidate_evaluations) >= 3


@pytest.mark.asyncio
async def test_certain_commercial_extinction_is_audited_before_any_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, adapter, repository = make_manager()
    run = await manager.create_run()
    doomed = CandidateSimulation(
        action=SimulationAction(
            name="doomed",
            price_per_customer_weekly=10,
            marketing_spend=0,
            development_spend=0,
        ),
        expected_ending_cash=500_000,
        downside_ending_cash=400_000,
        bankruptcy_probability=0,
        going_concern_failure_probability=1,
        expected_customer_growth=-10,
        robustness=RobustnessLevel.LOW,
        robust_utility=0.5,
        rollout_count=10,
    )

    async def no_viable_plan(**_kwargs):
        raise NoViableStrategyError((doomed,))

    monkeypatch.setattr(
        "lithops.application.step_run.prepare_weekly_plan",
        no_viable_plan,
    )

    with pytest.raises(NoViableStrategyError):
        await manager.step_run(run.id, request_id="no-viable-week")

    events = await repository.list_events(run.id)
    event = next(item for item in events if item.type == "decision.no_viable_plan")
    assert event.payload == {
        "week": 0,
        "reason_code": "all_candidates_certain_going_concern_failure",
        "candidate_risks": [
            {"strategy": "doomed", "going_concern_failure_probability": 1.0}
        ],
    }
    assert adapter.execute_action_calls == 0


class CrashAfterExternalAdvance(FakeBenchmarkAdapter):
    def __init__(self) -> None:
        super().__init__()
        self._crash_once = True

    async def advance_week(self, session_id, *, rationale, forecasts):
        result = await super().advance_week(
            session_id,
            rationale=rationale,
            forecasts=forecasts,
        )
        if self._crash_once:
            self._crash_once = False
            raise RuntimeError("simulated crash after external advance")
        return result


class CrashAfterOutcomeAppendRepository(InMemoryRunRepository):
    def __init__(self) -> None:
        super().__init__()
        self._crash_once = True

    async def append_prediction_outcome(self, outcome):
        saved = await super().append_prediction_outcome(outcome)
        if self._crash_once:
            self._crash_once = False
            raise RuntimeError("simulated crash after outcome persistence")
        return saved


@pytest.mark.asyncio
async def test_retry_reconciles_advance_after_crash_without_advancing_twice() -> None:
    adapter = CrashAfterExternalAdvance()
    manager, _, repository = make_manager(adapter)
    run = await manager.create_run()

    with pytest.raises(RuntimeError, match="simulated crash"):
        await manager.step_run(run.id, request_id="recoverable-week")

    recovered = await manager.step_run(run.id, request_id="recoverable-week")

    assert recovered.run.current_day == 7
    assert recovered.decision.status == DecisionStatus.COMMITTED
    assert adapter.advance_week_calls == 1
    assert adapter.execute_action_calls == 5
    events = await repository.list_events(run.id)
    assert "benchmark.advance_reconciled" in [event.type for event in events]


@pytest.mark.asyncio
async def test_new_request_key_advances_the_next_week() -> None:
    manager, adapter, _ = make_manager()
    run = await manager.create_run()

    first = await manager.step_run(run.id, request_id="week-0")
    second = await manager.step_run(run.id, request_id="week-1")

    assert first.run.current_day == 7
    assert second.run.current_day == 14
    assert adapter.advance_week_calls == 2
    assert adapter.execute_action_calls == 10
    predictions = await manager.learning_repository.list_predictions(run.id)
    outcomes = await manager.learning_repository.list_prediction_outcomes(run.id)
    models = await manager.learning_repository.list_world_models(run.id)
    assert len(predictions) == 2
    assert len(outcomes) == 1
    assert len(models) == 2
    assert second.decision.world_model_version_id == models[-1].id

    replay = await manager.step_run(run.id, request_id="week-1")
    assert replay.replayed is True
    assert len(await manager.learning_repository.list_predictions(run.id)) == 2
    assert len(await manager.learning_repository.list_prediction_outcomes(run.id)) == 1
    assert len(await manager.learning_repository.list_world_models(run.id)) == 2
    assert adapter.advance_week_calls == 2
    assert adapter.execute_action_calls == 10


@pytest.mark.asyncio
async def test_retry_finishes_recalibration_after_outcome_was_already_persisted() -> None:
    repository = CrashAfterOutcomeAppendRepository()
    adapter = FakeBenchmarkAdapter()
    manager = RunManager(
        repository=repository,
        benchmark=adapter,
        decision_engine=StaticDecisionEngine(),
        planning_rollouts=20,
    )
    run = await manager.create_run()
    await manager.step_run(run.id, request_id="week-0")

    with pytest.raises(RuntimeError, match="outcome persistence"):
        await manager.step_run(run.id, request_id="week-1")

    recovered = await manager.step_run(run.id, request_id="week-1")
    assert recovered.run.current_day == 14
    assert len(await repository.list_prediction_outcomes(run.id)) == 1
    models = await repository.list_world_models(run.id)
    assert len(models) == 2
    assert models[-1].version == 2
    assert all(change.evidence for change in models[-1].changes)
    assert adapter.advance_week_calls == 2
    assert adapter.execute_action_calls == 10
