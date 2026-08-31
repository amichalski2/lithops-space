from __future__ import annotations

from statistics import fmean
from typing import Protocol, cast
from uuid import UUID

from lithops.application.enterprise_negotiation import negotiate_open_threads
from lithops.application.executable_model_challenge import ExecutableModelChallenge
from lithops.application.executable_model_planning import ExecutableModelPlanner
from lithops.application.executive_selection import (
    ExecutiveAuthorityContext,
    learned_information_costs,
)
from lithops.application.model_challenge import ModelChallengeOrchestrator
from lithops.application.strategy_portfolio import (
    insight_summaries,
    portfolio_context_for_proposals,
    update_strategic_portfolio,
)
from lithops.application.weekly_planning import (
    estimate_cash_sensitivities,
    prepare_weekly_plan,
    sandbox_action_payload,
    simulation_action_from_action_plan,
    simulation_state_from_observation,
)
from lithops.benchmark.ceobench.insight_parser import (
    extract_payload_text,
    parse_insight,
)
from lithops.domain.errors import ConflictError, NotFoundError, OperationInProgressError
from lithops.domain.evaluation import ModelHealthSignal, ParameterResidualAttribution
from lithops.domain.insights import (
    InformationRequest,
    InsightParseStatus,
    fresh_insight_identities,
    insight_record_id,
    measured_quality_floor_metrics,
)
from lithops.domain.models import (
    ActionCommand,
    ActionPlan,
    ActionReceipt,
    CashForecast,
    CashForecasts,
    DecisionRecord,
    DecisionStatus,
    EventRecord,
    ObservationSnapshot,
    OperationStatus,
    RunRecord,
    RunStatus,
    StepOperation,
    StepResult,
    WorkflowStep,
    utc_now,
)
from lithops.domain.ports import (
    BenchmarkPort,
    LearningRepository,
    RunRepository,
    StrategyRepository,
)
from lithops.domain.predictions import (
    PredictionLedgerEntry,
    PredictionOutcome,
    PredictionOutcomeAttribution,
)
from lithops.domain.strategy import ExperimentOutcome, ExperimentOutcomeStatus
from lithops.domain.world_model import EvidenceKind, WorldModelVersion
from lithops.evaluation.action_fidelity import action_fidelity_violations
from lithops.evaluation.experiment_outcomes import reduce_experiment_outcome
from lithops.evaluation.model_health import evaluate_model_health
from lithops.evaluation.prediction_ledger import (
    attribute_prediction_policy_path,
    create_cash_prediction,
    mature_cash_predictions,
)
from lithops.evaluation.trajectory import revealed_quality_bar_lower_bound
from lithops.infrastructure.observability import span
from lithops.infrastructure.security.model_armor import REDACTION_NOTICE, get_screener
from lithops.model_runtime import TemporalObservation
from lithops.simulator.strategy_search import NoViableStrategyError
from lithops.world_model import (
    assemble_model_challenge_package,
    bootstrap_world_model,
    recalibrate_world_model,
)
from lithops.world_model.bootstrap import ParameterPrior


class RunStateError(RuntimeError):
    """The requested transition is invalid for the current run state."""


class DecisionEngine(Protocol):
    async def decide(
        self,
        *,
        run: RunRecord,
        observation: ObservationSnapshot,
    ) -> tuple[ActionPlan, CashForecasts]: ...


class StaticDecisionEngine:
    """Deterministic Executive proposal used for local and recovery verification."""

    prompt_version = "static-executive-v1"

    async def decide(
        self,
        *,
        run: RunRecord,
        observation: ObservationSnapshot,
    ) -> tuple[ActionPlan, CashForecasts]:
        week = observation.day // 7
        plan = ActionPlan(
            name="Controlled operating baseline",
            strategy_family="balanced_growth",
            rationale=(
                "Fund a bounded operations and development baseline while preserving "
                "runway and collecting a measurable response."
            ),
            commands=[
                ActionCommand(
                    tool="set_daily_spend",
                    arguments={"operations": 500.0, "development": 250.0},
                    idempotency_key=f"{run.id}:week-{week}:executive-spend",
                )
            ],
        )
        forecasts = []
        for horizon in (7, 28, 84, 182):
            point = observation.cash + 10_000.0 * (horizon / 7)
            uncertainty = max(20_000.0, abs(point) * (0.03 + horizon / 2_000))
            forecasts.append(
                CashForecast(
                    horizon_days=horizon,
                    point=point,
                    lower=point - uncertainty,
                    upper=point + uncertainty,
                )
            )
        return plan, CashForecasts(items=forecasts)


class RunManager:
    def __init__(
        self,
        *,
        repository: RunRepository,
        benchmark: BenchmarkPort,
        decision_engine: DecisionEngine,
        learning_repository: LearningRepository | None = None,
        model_challenge_orchestrator: ModelChallengeOrchestrator | None = None,
        model_challenge_cooldown_days: int = 7,
        planning_rollouts: int = 200,
        executable_model_planner: ExecutableModelPlanner | None = None,
        executable_model_challenge: ExecutableModelChallenge | None = None,
        executive_authority_v2: bool = False,
        strategy_repository: StrategyRepository | None = None,
        bootstrap_priors: tuple[ParameterPrior, ...] | None = None,
    ) -> None:
        self.repository = repository
        self.learning_repository = learning_repository or cast(LearningRepository, repository)
        self.executive_authority_v2 = executive_authority_v2
        self.strategy_repository = strategy_repository or cast(StrategyRepository, repository)
        self.benchmark = benchmark
        self.decision_engine = decision_engine
        self.model_challenge_orchestrator = model_challenge_orchestrator
        if model_challenge_cooldown_days < 7:
            raise ValueError("model challenge cooldown must be at least seven days")
        self.model_challenge_cooldown_days = model_challenge_cooldown_days
        self.planning_rollouts = planning_rollouts
        self.executable_model_planner = executable_model_planner
        self.executable_model_challenge = executable_model_challenge
        # None means the current P0 prior catalog; a caller may pin a fixed
        # tuple so its runs stay reproducible as the catalog grows.
        self.bootstrap_priors = bootstrap_priors

    async def create_run(self, *, horizon_days: int = 500) -> RunRecord:
        run = await self.repository.create_run(RunRecord(horizon_days=horizon_days))
        await self._event(run.id, "run.created", {"horizon_days": horizon_days})
        return run

    async def get_run(self, run_id: UUID) -> RunRecord:
        return await self.repository.get_run(run_id)

    async def start_run(self, run_id: UUID) -> RunRecord:
        run = await self.repository.get_run(run_id)
        if run.status in {RunStatus.BOOTSTRAPPING, RunStatus.RUNNING}:
            return run
        if run.status != RunStatus.CREATED:
            raise RunStateError(f"run cannot start from status: {run.status}")
        saved = await self.repository.save_run(
            run.model_copy(update={"status": RunStatus.BOOTSTRAPPING}, deep=True),
            expected_version=run.version,
        )
        await self._event(saved.id, "run.started", {"status": saved.status})
        return saved

    async def request_pause(self, run_id: UUID) -> RunRecord:
        run = await self.repository.get_run(run_id)
        if run.status in {RunStatus.PAUSING, RunStatus.PAUSED}:
            return run
        if run.status not in {RunStatus.BOOTSTRAPPING, RunStatus.RUNNING}:
            raise RunStateError(f"run cannot pause from status: {run.status}")
        saved = await self.repository.save_run(
            run.model_copy(update={"status": RunStatus.PAUSING}, deep=True),
            expected_version=run.version,
        )
        await self._event(
            saved.id,
            "run.pause_requested",
            {"effective_after_day": run.current_day},
        )
        return saved

    async def complete_pause(self, run_id: UUID) -> RunRecord:
        run = await self.repository.get_run(run_id)
        if run.status == RunStatus.PAUSED:
            return run
        if run.status != RunStatus.PAUSING:
            raise RunStateError(f"run cannot complete pause from status: {run.status}")
        if run.workflow_step not in {WorkflowStep.READY, WorkflowStep.CHECKPOINT}:
            raise RunStateError("run can pause only at a safe checkpoint")
        saved = await self.repository.save_run(
            run.model_copy(update={"status": RunStatus.PAUSED}, deep=True),
            expected_version=run.version,
        )
        await self._event(saved.id, "run.paused", {"day": saved.current_day})
        return saved

    async def resume_run(self, run_id: UUID) -> RunRecord:
        run = await self.repository.get_run(run_id)
        if run.status in {RunStatus.BOOTSTRAPPING, RunStatus.RUNNING}:
            return run
        if run.status != RunStatus.PAUSED:
            raise RunStateError(f"run cannot resume from status: {run.status}")
        next_status = (
            RunStatus.RUNNING if run.benchmark_session_id is not None else RunStatus.BOOTSTRAPPING
        )
        saved = await self.repository.save_run(
            run.model_copy(update={"status": next_status}, deep=True),
            expected_version=run.version,
        )
        await self._event(saved.id, "run.resumed", {"day": saved.current_day})
        return saved

    async def retry_failed_run(self, run_id: UUID) -> RunRecord:
        """Re-open a failed run so its persisted week can be reconciled and retried."""

        run = await self.repository.get_run(run_id)
        if run.status != RunStatus.FAILED:
            raise RunStateError(f"run cannot retry from status: {run.status}")
        next_status = (
            RunStatus.RUNNING if run.benchmark_session_id is not None else RunStatus.BOOTSTRAPPING
        )
        saved = await self.repository.save_run(
            run.model_copy(
                update={"status": next_status, "failure_reason": None},
                deep=True,
            ),
            expected_version=run.version,
        )
        await self._event(
            saved.id,
            "run.retry_started",
            {"day": saved.current_day, "previous_failure": run.failure_reason},
        )
        return saved

    async def fail_run(self, run_id: UUID, *, reason: str) -> RunRecord:
        run = await self.repository.get_run(run_id)
        if run.status == RunStatus.FAILED:
            return run
        if run.status in {RunStatus.COMPLETED, RunStatus.BANKRUPT}:
            raise RunStateError(f"terminal run cannot be failed: {run.status}")
        saved = await self.repository.save_run(
            run.model_copy(
                update={"status": RunStatus.FAILED, "failure_reason": reason[:2_000]},
                deep=True,
            ),
            expected_version=run.version,
        )
        await self._event(saved.id, "run.failed", {"reason": saved.failure_reason})
        return saved

    async def list_events(self, run_id: UUID) -> list[EventRecord]:
        await self.repository.get_run(run_id)
        return await self.repository.list_events(run_id)

    async def list_decisions(self, run_id: UUID) -> list[DecisionRecord]:
        await self.repository.get_run(run_id)
        return await self.repository.list_decisions(run_id)

    async def get_decision(self, run_id: UUID, decision_id: UUID) -> DecisionRecord:
        await self.repository.get_run(run_id)
        return await self.repository.get_decision(run_id, decision_id)

    async def get_latest_world_model(self, run_id: UUID) -> WorldModelVersion:
        await self.repository.get_run(run_id)
        world_model = await self.learning_repository.get_latest_world_model(run_id)
        if world_model is None:
            raise NotFoundError(f"world model not found for run: {run_id}")
        return world_model

    async def get_world_model(
        self,
        run_id: UUID,
        model_id: UUID,
    ) -> WorldModelVersion:
        await self.repository.get_run(run_id)
        world_model = await self.learning_repository.get_world_model(model_id)
        if world_model.run_id != run_id:
            raise NotFoundError(f"world model not found: {model_id}")
        return world_model

    async def get_prediction(
        self,
        run_id: UUID,
        prediction_id: UUID,
    ) -> PredictionLedgerEntry:
        await self.repository.get_run(run_id)
        prediction = await self.learning_repository.get_prediction(prediction_id)
        if prediction.run_id != run_id:
            raise NotFoundError(f"prediction not found: {prediction_id}")
        return prediction

    async def list_predictions(self, run_id: UUID) -> list[PredictionLedgerEntry]:
        await self.repository.get_run(run_id)
        return await self.learning_repository.list_predictions(run_id)

    async def list_prediction_outcomes(self, run_id: UUID) -> list[PredictionOutcome]:
        await self.repository.get_run(run_id)
        return await self.learning_repository.list_prediction_outcomes(run_id)

    async def list_model_health_signals(self, run_id: UUID) -> list[ModelHealthSignal]:
        await self.repository.get_run(run_id)
        return await self.learning_repository.list_model_health_signals(run_id)

    async def step_run(
        self,
        run_id: UUID,
        *,
        request_id: str,
        recover_in_progress: bool = False,
    ) -> StepResult:
        existing = await self.repository.get_operation(run_id, request_id)
        if existing is not None:
            if existing.status == OperationStatus.COMPLETED and existing.result is not None:
                result = StepResult.model_validate(existing.result)
                return result.model_copy(update={"replayed": True}, deep=True)
            if existing.status == OperationStatus.STARTED:
                if not recover_in_progress:
                    raise OperationInProgressError(f"operation already in progress: {request_id}")
                await self.repository.fail_operation(
                    run_id,
                    request_id,
                    "reclaimed after previous worker lease expired",
                )
                await self._event(
                    run_id,
                    "operation.reclaimed",
                    {"request_id": request_id},
                )

        operation = StepOperation(run_id=run_id, request_id=request_id)
        claimed = await self.repository.start_operation(operation)
        if claimed.status == OperationStatus.COMPLETED and claimed.result is not None:
            result = StepResult.model_validate(claimed.result)
            return result.model_copy(update={"replayed": True}, deep=True)

        try:
            with span("lithops.week", run_id=str(run_id), request_id=request_id):
                result = await self._execute_week(
                    run_id,
                    allow_pausing=recover_in_progress,
                )
        except Exception as exc:
            await self.repository.fail_operation(run_id, request_id, str(exc))
            if isinstance(exc, NoViableStrategyError):
                current = await self.repository.get_run(run_id)
                week = current.current_day // 7
                events = await self.repository.list_events(run_id)
                if not any(
                    event.type == "decision.no_viable_plan"
                    and event.payload.get("week") == week
                    for event in events
                ):
                    await self._event(
                        run_id,
                        "decision.no_viable_plan",
                        {
                            "week": week,
                            "reason_code": "all_candidates_certain_going_concern_failure",
                            "candidate_risks": [
                                {"strategy": name, "going_concern_failure_probability": risk}
                                for name, risk in exc.candidate_risks
                            ],
                        },
                    )
            await self._event(
                run_id,
                "operation.failed",
                {"request_id": request_id, "error": str(exc)},
            )
            raise

        await self.repository.complete_operation(run_id, request_id, result)
        return result

    async def _execute_week(
        self,
        run_id: UUID,
        *,
        allow_pausing: bool,
    ) -> StepResult:
        run = await self.repository.get_run(run_id)
        self._assert_runnable(run, allow_pausing=allow_pausing)
        run = await self._ensure_benchmark_session(run)
        session_id = run.benchmark_session_id
        if session_id is None:
            raise RunStateError("benchmark session was not persisted")

        with span("week.observe"):
            external_state = await self._observe_with_evidence(session_id)
        external_state = await self._screen_untrusted_text(run, external_state)
        checkpoint_day = run.current_day
        expected_next_day = min(checkpoint_day + 7, run.horizon_days)
        if external_state.day not in {checkpoint_day, expected_next_day}:
            raise RunStateError(
                "benchmark day cannot be reconciled with the Lithops checkpoint: "
                f"external={external_state.day}, checkpoint={checkpoint_day}"
            )

        week = checkpoint_day // 7
        decision = await self.repository.get_decision_for_week(run.id, week)
        if external_state.day == expected_next_day:
            if decision is None:
                raise RunStateError(
                    "benchmark advanced but no prepared decision exists for recovery"
                )
            await self._ensure_prediction(decision)
            receipts = await self.repository.list_receipts(decision.id)
            missing_keys = {
                command.idempotency_key for command in decision.action_plan.commands
            } - {receipt.idempotency_key for receipt in receipts}
            if missing_keys:
                raise RunStateError(
                    "benchmark advanced before all action receipts were persisted: "
                    f"{sorted(missing_keys)}"
                )
            await self._event(
                run.id,
                "benchmark.advance_reconciled",
                {"decision_id": str(decision.id), "observed_day": external_state.day},
            )
            return await self._commit_week(run, decision, receipts, external_state)

        if decision is None:
            with span("week.learn", week=week):
                world_model = await self._process_learning(run, external_state)
            with span("week.decide", week=week):
                decision = await self._prepare_decision(run, external_state, world_model)
        else:
            await self._ensure_prediction(decision)

        if decision.status == DecisionStatus.COMMITTED:
            if decision.actual_outcome is None:
                raise RunStateError("committed decision is missing its actual outcome")
            receipts = await self.repository.list_receipts(decision.id)
            return await self._commit_week(run, decision, receipts, decision.actual_outcome)

        with span("week.execute_actions", week=week):
            receipts = await self._execute_missing_actions(
                run=run,
                session_id=session_id,
                decision=decision,
            )
        after_actions = await self._observe_with_evidence(session_id)
        program = decision.action_plan.experiment_program
        if (
            program is not None
            and program.protocol_version == "experiment-program-v2"
            and callable(getattr(self.benchmark, "collect_weekly_evidence", None))
        ):
            violations = action_fidelity_violations(
                decision.action_plan.commands, after_actions
            )
            if violations:
                await self._event(
                    run.id,
                    "action.execution_mismatch",
                    {
                        "decision_id": str(decision.id),
                        "commitment_id": program.commitment_id,
                        "violations": list(violations),
                    },
                )
                raise RunStateError(
                    "post-action configuration does not match the selected treatment: "
                    + ", ".join(violations)
                )
        if self.executive_authority_v2 and decision.action_plan.enterprise_engage:
            receipts.extend(
                await self._negotiate_enterprise_threads(
                    run=run,
                    session_id=session_id,
                    decision=decision,
                    observation=after_actions,
                )
            )
        if after_actions.day == decision.observation.day:
            with span("week.advance", week=week):
                actual = await self.benchmark.advance_week(
                    session_id,
                    rationale=decision.action_plan.rationale,
                    forecasts=decision.forecasts,
                )
                actual = await self._attach_evidence(session_id, actual)
        elif after_actions.day == expected_next_day:
            actual = after_actions
            await self._event(
                run.id,
                "benchmark.advance_reconciled",
                {"decision_id": str(decision.id), "observed_day": actual.day},
            )
        else:
            raise RunStateError(
                "benchmark changed by more than one week during a decision: "
                f"before={decision.observation.day}, after={after_actions.day}"
            )
        with span("week.commit", week=week):
            return await self._commit_week(run, decision, receipts, actual)

    async def _negotiate_enterprise_threads(
        self,
        *,
        run: RunRecord,
        session_id: str,
        decision: DecisionRecord,
        observation: ObservationSnapshot,
    ) -> list[ActionReceipt]:
        """Answer open enterprise threads inside the authorized envelope."""

        inbox = observation.metrics.get("enterprise_inbox")
        if not isinstance(inbox, str) or not inbox:
            return []
        run_id = run.id

        async def execute_action(command: ActionCommand) -> ActionReceipt:
            existing = await self.repository.get_receipt(run_id, command.idempotency_key)
            if existing is not None:
                return existing
            receipt = await self.benchmark.execute_action(
                session_id,
                run_id=run_id,
                decision_id=decision.id,
                command=command,
            )
            return await self.repository.save_receipt(receipt)

        async def emit_event(event_type: str, payload: dict) -> None:
            await self._event(run_id, event_type, payload)

        cost_per_customer = observation.metrics.get(
            "operating_cost_per_customer_weekly", 0.0
        )
        outcome = await negotiate_open_threads(
            run_id=run_id,
            week=decision.week,
            plan=decision.action_plan,
            inbox=inbox,
            variable_cost_per_seat_weekly=(
                float(cost_per_customer)
                if isinstance(cost_per_customer, int | float)
                else 0.0
            ),
            execute_action=execute_action,
            emit_event=emit_event,
        )
        if outcome.offers_made or outcome.skipped:
            await self._event(
                run.id,
                "enterprise.negotiation_completed",
                {
                    "week": decision.week,
                    "threads_seen": outcome.threads_seen,
                    "offers_made": outcome.offers_made,
                    "skipped": list(outcome.skipped),
                },
            )
        return list(outcome.receipts)

    async def _answer_data_queries(
        self,
        run: RunRecord,
        *,
        week: int,
        queries: tuple[str, ...],
    ) -> None:
        """Answer the Executive's own read-only questions about the business.

        The observation contract is what deterministic code decided to expose.
        This is how the Executive looks at something that contract does not
        carry. The SQL guard keeps it read-only; a bad query is reported back
        rather than failing the week.
        """

        if not queries or run.benchmark_session_id is None:
            return
        for sequence, query in enumerate(queries[:3]):
            try:
                rows = await self.benchmark.query_readonly(
                    run.benchmark_session_id, query
                )
                answer = {
                    "week": week,
                    "query": query[:2_000],
                    "row_count": len(rows),
                    # Bounded so one broad question cannot flood the next brief.
                    "rows": [dict(row) for row in rows[:20]],
                }
            except Exception as error:
                answer = {
                    "week": week,
                    "query": query[:2_000],
                    "error": str(error)[:500],
                }
            await self._event(
                run.id,
                "executive.data_query_answered",
                {"sequence": sequence, **answer},
            )

    async def _execute_information_requests(
        self,
        run: RunRecord,
        *,
        decision_id: UUID,
        week: int,
        requests: tuple[InformationRequest, ...],
    ) -> None:
        """Buy the authorized information and record what it said.

        These tools change no configuration, so they run outside the simulated
        action plan. Replay is guarded by the record's deterministic identity: an
        answer already on file is never paid for twice.
        """

        if not requests or run.benchmark_session_id is None:
            return
        existing = {
            record.id for record in await self.strategy_repository.list_insight_records(run.id)
        }
        cash_before = (
            await self.benchmark.observe_status(run.benchmark_session_id)
        ).cash
        for sequence, request in enumerate(requests):
            record_id = insight_record_id(run.id, week, request.identity)
            if record_id in existing:
                continue
            arguments: dict[str, object] = {}
            if request.target_group is not None:
                arguments["group_id"] = request.target_group
            if request.tool == "research_group" and request.target_level is not None:
                arguments["target_level"] = request.target_level
            command = ActionCommand(
                tool=request.tool,
                arguments=arguments,
                idempotency_key=(
                    f"{run.id}:week-{week}:info-{request.tool}-"
                    f"{request.target_group or 'all'}-{sequence}"
                ),
            )
            try:
                receipt = await self.benchmark.execute_action(
                    run.benchmark_session_id,
                    run_id=run.id,
                    decision_id=decision_id,
                    command=command,
                )
            except Exception as error:
                await self._event(
                    run.id,
                    "information.request_failed",
                    {
                        "week": week,
                        "tool": request.tool,
                        "target_group": request.target_group,
                        "error": str(error)[:500],
                    },
                )
                continue
            # What information costs is not declared anywhere we can read, so it
            # is measured: the cash the purchase actually consumed. That measured
            # price is what bounds later purchases of the same kind.
            try:
                after = await self.benchmark.observe_status(run.benchmark_session_id)
                cost = max(0.0, cash_before - after.cash)
                cash_before = after.cash
            except Exception:
                cost = 0.0
            insight = parse_insight(
                run_id=run.id,
                week=week,
                request=request,
                payload=extract_payload_text(receipt.result),
                cost=cost,
                created_at=utc_now(),
            )
            await self.strategy_repository.append_insight_record(insight)
            if insight.parse_status is InsightParseStatus.FAILED:
                await self._event(
                    run.id,
                    "insight.parse_failed",
                    {
                        "week": week,
                        "tool": request.tool,
                        "target_group": request.target_group,
                        "parser_version": insight.parser_version,
                    },
                )
            else:
                await self._event(
                    run.id,
                    "insight.recorded",
                    {
                        "week": week,
                        "tool": request.tool,
                        "target_group": insight.target_group,
                        "info_level": insight.info_level,
                        "noise_band": insight.noise_band,
                        "quality_floor": insight.quality_floor,
                        "willingness_to_pay_monthly": (
                            insight.willingness_to_pay_monthly
                        ),
                        "usage_units_per_day": insight.usage_units_per_day,
                        "discovered_group": insight.discovered_group,
                        "parse_status": insight.parse_status.value,
                    },
                )

    async def _screen_untrusted_text(
        self, run: RunRecord, observation: ObservationSnapshot
    ) -> ObservationSnapshot:
        """Screen environment-authored text through Model Armor before any brief is built.

        Only the decision-facing snapshot is screened here; outcome snapshots
        re-enter through the next week's observe and are screened then. The
        verdict always lands on the event ledger, including screening errors,
        so an API outage never reads as "nothing was flagged".
        """

        screener = get_screener()
        if screener is None:
            return observation
        texts = {
            key: value
            for key, value in observation.metrics.items()
            if isinstance(value, str) and value.strip()
        }
        if not texts:
            return observation
        with span("week.screen_untrusted_text"):
            results = await screener.screen(texts)
        flagged = [result for result in results if result.flagged]
        errored = [result for result in results if result.error]
        await self._event(
            run.id,
            "security.model_armor",
            {
                "day": observation.day,
                "mode": screener.mode,
                "screened_fields": sorted(texts),
                "flagged": [
                    {"field": result.field, "filters": list(result.filters)}
                    for result in flagged
                ],
                "errors": [
                    {"field": result.field, "error": result.error} for result in errored
                ],
            },
        )
        if screener.mode != "enforce" or not flagged:
            return observation
        metrics = dict(observation.metrics)
        for result in flagged:
            metrics[result.field] = REDACTION_NOTICE
        return observation.model_copy(update={"metrics": metrics}, deep=True)

    async def _observe_with_evidence(self, session_id: str) -> ObservationSnapshot:
        observation = await self.benchmark.observe_status(session_id)
        return await self._attach_evidence(session_id, observation)

    async def _attach_evidence(
        self, session_id: str, observation: ObservationSnapshot
    ) -> ObservationSnapshot:
        collector = getattr(self.benchmark, "collect_weekly_evidence", None)
        if collector is None:
            return observation
        evidence = await collector(session_id, observation)
        return observation.model_copy(update={"evidence": evidence}, deep=True)

    async def _ensure_benchmark_session(self, run: RunRecord) -> RunRecord:
        if run.benchmark_session_id is not None:
            return run
        session_id = await self.benchmark.create_session(run.id, days=run.horizon_days)
        next_status = RunStatus.PAUSING if run.status == RunStatus.PAUSING else RunStatus.RUNNING
        saved = await self.repository.save_run(
            run.model_copy(
                update={"benchmark_session_id": session_id, "status": next_status},
                deep=True,
            ),
            expected_version=run.version,
        )
        await self._event(saved.id, "benchmark.session_created", {"session_id": session_id})
        return saved

    async def _process_learning(
        self,
        run: RunRecord,
        observation: ObservationSnapshot,
    ) -> WorldModelVersion:
        world_model = await self.learning_repository.get_latest_world_model(run.id)
        if world_model is None:
            world_model = await self.learning_repository.append_world_model(
                bootstrap_world_model(run.id, observation)
                if self.bootstrap_priors is None
                else bootstrap_world_model(
                    run.id, observation, priors=self.bootstrap_priors
                ),
                expected_latest_version=None,
            )
            await self._event(
                run.id,
                "world_model.created",
                {"model_version_id": str(world_model.id), "version": world_model.version},
            )

        entries = tuple(await self.learning_repository.list_predictions(run.id))
        existing_outcomes = tuple(await self.learning_repository.list_prediction_outcomes(run.id))
        decisions = tuple(await self.repository.list_decisions(run.id))
        entries_by_id = {entry.id: entry for entry in entries}
        matured_outcomes = mature_cash_predictions(
            entries,
            observation,
            observation_reference=f"observation:{run.id}:{observation.day}",
            existing_outcomes=existing_outcomes,
        )
        new_outcomes = tuple(
            attribute_prediction_policy_path(
                outcome,
                entry=entries_by_id[outcome.ledger_entry_id],
                decisions=decisions,
            )
            for outcome in matured_outcomes
        )
        for outcome in new_outcomes:
            await self.learning_repository.append_prediction_outcome(outcome)
            await self._event(
                run.id,
                "prediction.matured",
                {
                    "prediction_outcome_id": str(outcome.id),
                    "target_id": str(outcome.target_id),
                    "observed_day": outcome.actual.observed_day,
                    "normalized_absolute_error": outcome.score.normalized_absolute_error,
                    "interval_hit": outcome.score.interval_hit,
                    "attribution": outcome.attribution,
                    "policy_divergence_week": outcome.policy_divergence_week,
                },
            )

        all_outcomes = tuple(await self.learning_repository.list_prediction_outcomes(run.id))
        models = tuple(await self.learning_repository.list_world_models(run.id))
        processed_references = {
            evidence.reference
            for model in models
            for parameter in model.parameters
            for evidence in parameter.evidence
            if evidence.kind is EvidenceKind.PREDICTION_RESIDUAL
        }
        pending_outcomes = tuple(
            outcome
            for outcome in all_outcomes
            if f"prediction-outcome:{outcome.id}" not in processed_references
        )
        pending_model_outcomes = tuple(
            outcome
            for outcome in pending_outcomes
            if outcome.attribution is PredictionOutcomeAttribution.MODEL_PERFORMANCE
        )
        if not pending_model_outcomes:
            return world_model

        model_outcomes = tuple(
            outcome
            for outcome in all_outcomes
            if outcome.attribution is PredictionOutcomeAttribution.MODEL_PERFORMANCE
        )
        health = evaluate_model_health(
            model_version_id=world_model.id,
            entries=entries,
            outcomes=model_outcomes,
            observations=tuple(
                decision.observation
                for decision in decisions
                if decision.observation.day <= observation.day
            )
            + (observation,),
        )
        health = await self.learning_repository.append_model_health_signal(health)
        events = await self.repository.list_events(run.id)
        if not any(
            event.type == "model_health.evaluated"
            and event.payload.get("model_health_signal_id") == str(health.id)
            for event in events
        ):
            await self._event(
                run.id,
                "model_health.evaluated",
                {
                    "model_health_signal_id": str(health.id),
                    "model_version_id": str(world_model.id),
                    "status": health.status,
                    "rebuild_recommended": health.rebuild_recommended,
                    "trigger_codes": list(health.trigger_codes),
                },
            )
        if health.rebuild_recommended and (
            self.executable_model_challenge is not None
            or self.model_challenge_orchestrator is not None
        ):
            challenge_observations = tuple(
                decision.observation
                for decision in decisions
                if decision.observation.day <= observation.day
            ) + (observation,)
            package = assemble_model_challenge_package(
                health_signal=health,
                active_model=world_model,
                observations=challenge_observations,
                predictions=entries,
                outcomes=all_outcomes,
            )
            if self.executable_model_challenge is not None:
                temporal_observations = self._temporal_observations(
                    run.id,
                    decisions,
                    observation,
                )
                model_registry = self.executable_model_challenge.repository
                find_promotion = model_registry.get_model_promotion_decision_for_challenge
                existing_promotion = await find_promotion(
                    run.id,
                    package.challenge_id,
                )
                prior_started_days = [
                    int(event.payload["evaluated_day"])
                    for event in events
                    if event.type == "executable_model_challenge.started"
                    and event.payload.get("challenge_id") != str(package.challenge_id)
                    and isinstance(event.payload.get("evaluated_day"), int)
                ]
                last_started_day = max(prior_started_days, default=None)
                cooldown_active = (
                    existing_promotion is None
                    and last_started_day is not None
                    and health.evaluated_day - last_started_day < self.model_challenge_cooldown_days
                )
                if cooldown_active:
                    await self._event_once(
                        events,
                        run.id,
                        "executable_model_challenge.skipped",
                        package.challenge_id,
                        {
                            "reason_code": "cooldown_active",
                            "evaluated_day": health.evaluated_day,
                            "last_challenge_day": last_started_day,
                            "cooldown_days": self.model_challenge_cooldown_days,
                        },
                    )
                elif len(temporal_observations) < 3 and existing_promotion is None:
                    await self._event_once(
                        events,
                        run.id,
                        "executable_model_challenge.skipped",
                        package.challenge_id,
                        {
                            "reason_code": "insufficient_temporal_observations",
                            "evaluated_day": health.evaluated_day,
                            "observation_count": len(temporal_observations),
                        },
                    )
                else:
                    await self._event_once(
                        events,
                        run.id,
                        "executable_model_challenge.started",
                        package.challenge_id,
                        {
                            "evaluated_day": health.evaluated_day,
                            "observation_count": len(temporal_observations),
                            "trigger_codes": list(health.trigger_codes),
                        },
                    )
                    challenge_error: Exception | None = None
                    try:
                        executable_outcome = await self.executable_model_challenge.run(
                            package=package,
                            observations=temporal_observations,
                            world_model=world_model,
                            seed=health.evaluated_day + 70_001,
                        )
                    except Exception as first_error:
                        await self._event_once(
                            events,
                            run.id,
                            "executable_model_challenge.retrying",
                            package.challenge_id,
                            {
                                "reason_code": "challenge_internal_error",
                                "error_code": type(first_error).__name__,
                            },
                        )
                        try:
                            executable_outcome = await self.executable_model_challenge.run(
                                package=package,
                                observations=temporal_observations,
                                world_model=world_model,
                                seed=health.evaluated_day + 70_001,
                            )
                        except Exception as retry_error:
                            challenge_error = retry_error
                            executable_outcome = None
                    if challenge_error is not None:
                        await self._event_once(
                            events,
                            run.id,
                            "executable_model_challenge.failed",
                            package.challenge_id,
                            {
                                "disposition": "no_update",
                                "reason_code": "challenge_internal_error",
                                "failure_codes": [
                                    f"challenge:{type(challenge_error).__name__}"
                                ],
                                "contained": True,
                                "operational_fallback_artifact_id": str(
                                    self.executable_model_challenge.baseline.artifact.id
                                ),
                            },
                        )
                    elif executable_outcome is not None and executable_outcome.promotion is None:
                        await self._event_once(
                            events,
                            run.id,
                            "executable_model_challenge.failed",
                            package.challenge_id,
                            {
                                "disposition": "no_update",
                                "reason_code": executable_outcome.failure_reason_code,
                                "failure_codes": list(executable_outcome.failure_codes),
                                "contained": True,
                                "operational_fallback_artifact_id": (
                                    str(
                                        executable_outcome.operational_fallback_artifact_id
                                    )
                                    if executable_outcome.operational_fallback_artifact_id
                                    is not None
                                    else None
                                ),
                            },
                        )
                    elif executable_outcome is not None:
                        await self._event_once(
                            events,
                            run.id,
                            "executable_model_challenge.completed",
                            package.challenge_id,
                            {
                                "promotion_decision_id": str(
                                    executable_outcome.promotion.id
                                ),
                                "disposition": executable_outcome.promotion.disposition,
                                "reason_code": executable_outcome.promotion.reason_code,
                                "activated_model_assignment_id": (
                                    str(executable_outcome.activation.id)
                                    if executable_outcome.activation is not None
                                    else None
                                ),
                            },
                        )
            elif self.model_challenge_orchestrator is not None:
                existing_challenge = await self.learning_repository.get_model_challenge(
                    package.challenge_id
                )
                previous_challenge_days: list[int] = []
                for event in events:
                    if event.type != "model_challenge.started":
                        continue
                    raw_challenge_id = event.payload.get("challenge_id")
                    try:
                        challenge_id = UUID(str(raw_challenge_id))
                    except (TypeError, ValueError):
                        continue
                    if challenge_id == package.challenge_id:
                        continue
                    previous_package = await self.learning_repository.get_model_challenge_package(
                        challenge_id
                    )
                    if previous_package is not None:
                        previous_challenge_days.append(previous_package.health_signal.evaluated_day)
                last_challenge_day = max(previous_challenge_days, default=None)
                if (
                    existing_challenge is None
                    and last_challenge_day is not None
                    and health.evaluated_day - last_challenge_day
                    < self.model_challenge_cooldown_days
                ):
                    await self._event(
                        run.id,
                        "model_challenge.skipped",
                        {
                            "reason_code": "cooldown_active",
                            "evaluated_day": health.evaluated_day,
                            "last_challenge_day": last_challenge_day,
                            "cooldown_days": self.model_challenge_cooldown_days,
                        },
                    )
                    challenge_outcome = None
                else:
                    challenge_outcome = await self.model_challenge_orchestrator.run(package)
                if challenge_outcome is not None and challenge_outcome.activated_model is not None:
                    updated = challenge_outcome.activated_model
                    await self._event(
                        run.id,
                        "world_model.updated",
                        {
                            "model_version_id": str(updated.id),
                            "version": updated.version,
                            "parent_model_version_id": str(world_model.id),
                            "changed_parameters": [
                                change.parameter_name for change in updated.changes
                            ],
                            "rebuild_recommended": True,
                            "model_challenge_id": str(package.challenge_id),
                        },
                    )
                    return updated
        attributions = self._residual_attributions(entries, pending_model_outcomes)
        if not attributions:
            return world_model
        updated = recalibrate_world_model(
            world_model=world_model,
            entries=entries,
            outcomes=pending_model_outcomes,
            attributions=attributions,
        )
        updated = await self.learning_repository.append_world_model(
            updated,
            expected_latest_version=world_model.version,
        )
        await self._event(
            run.id,
            "world_model.updated",
            {
                "model_version_id": str(updated.id),
                "version": updated.version,
                "parent_model_version_id": str(world_model.id),
                "changed_parameters": [change.parameter_name for change in updated.changes],
                "rebuild_recommended": health.rebuild_recommended,
            },
        )
        return updated

    @staticmethod
    def _temporal_observations(
        run_id: UUID,
        decisions: tuple[DecisionRecord, ...],
        current: ObservationSnapshot,
    ) -> tuple[TemporalObservation, ...]:
        ordered_decisions = tuple(
            sorted(decisions, key=lambda item: (item.observation.day, item.week))
        )
        observations_by_day = {
            decision.observation.day: decision.observation
            for decision in ordered_decisions
            if decision.observation.day <= current.day
        }
        observations_by_day[current.day] = current
        temporal: list[TemporalObservation] = []
        for day, snapshot in sorted(observations_by_day.items()):
            previous_decision = next(
                (
                    decision
                    for decision in reversed(ordered_decisions)
                    if decision.observation.day < day
                ),
                None,
            )
            action = {}
            if previous_decision is not None:
                origin_state = simulation_state_from_observation(previous_decision.observation)
                simulation_action = simulation_action_from_action_plan(
                    previous_decision.action_plan,
                    origin_state,
                )
                action = sandbox_action_payload(
                    simulation_action,
                    origin_state,
                    horizon_weeks=1,
                )
            temporal.append(
                TemporalObservation(
                    observation_id=f"observation:{run_id}:{day}",
                    day=day,
                    state=simulation_state_from_observation(snapshot).model_dump(mode="json"),
                    action_from_previous=action,
                )
            )
        return tuple(temporal)

    async def _recent_refusals(
        self,
        run_id: UUID,
        *,
        week: int,
        events: list[EventRecord],
    ) -> tuple[dict, ...]:
        """What the Executive proposed recently that was refused, and under which codes.

        The author of a refused proposal is the only party who can replace it, so
        the refusal is carried back to it by name. Bounded to the last two weeks so
        the brief stays about the decision at hand rather than the whole run.
        """

        window = {week - 2, week - 1}
        refusals: list[dict] = []
        for event in events:
            payload = event.payload
            if payload.get("week") not in window:
                continue
            if event.type == "decision.candidate_pool_degraded":
                for rejected in payload.get("rejected") or ():
                    refusals.append(
                        {
                            "week": payload.get("week"),
                            "candidate": rejected.get("strategy"),
                            "veto_codes": list(rejected.get("violation_codes") or ()),
                            "stage": "pool",
                            "detail": str(rejected.get("detail") or "")[:300],
                        }
                    )
            elif event.type == "decision.candidate_construction_failed":
                refusals.append(
                    {
                        "week": payload.get("week"),
                        "candidate": payload.get("candidate_id"),
                        "veto_codes": list(payload.get("veto_codes") or ()),
                        "stage": "evaluation_build",
                        "detail": str(payload.get("detail") or "")[:300],
                    }
                )
        card_reader = getattr(
            self.strategy_repository, "get_candidate_evaluation_set", None
        )
        for past_week in sorted(window):
            # Cards only exist where the selection stage ran; a run without it
            # still gets the pool- and construction-stage refusals above.
            if past_week < 0 or card_reader is None:
                continue
            try:
                evaluation_set = await card_reader(run_id, past_week)
            except NotImplementedError:
                break
            if evaluation_set is None:
                continue
            refusals.extend(
                {
                    "week": past_week,
                    "candidate": card.candidate_id,
                    "veto_codes": list(card.veto_codes),
                    "stage": "evaluation",
                    "detail": "",
                }
                for card in evaluation_set.cards
                if not card.eligible
            )
        return tuple(refusals[-12:])

    async def _event_once(
        self,
        events: list[EventRecord],
        run_id: UUID,
        event_type: str,
        challenge_id: UUID,
        payload: dict,
    ) -> EventRecord | None:
        if any(
            event.type == event_type and event.payload.get("challenge_id") == str(challenge_id)
            for event in events
        ):
            return None
        event = await self._event(
            run_id,
            event_type,
            {"challenge_id": str(challenge_id), **payload},
        )
        events.append(event)
        return event

    @staticmethod
    def _residual_attributions(
        entries: tuple[PredictionLedgerEntry, ...],
        outcomes: tuple[PredictionOutcome, ...],
    ) -> tuple[ParameterResidualAttribution, ...]:
        entries_by_id = {entry.id: entry for entry in entries}
        attributions: list[ParameterResidualAttribution] = []
        for outcome in outcomes:
            entry = entries_by_id[outcome.ledger_entry_id]
            horizon = next(
                target.horizon_days for target in entry.targets if target.id == outcome.target_id
            )
            for sensitivity in entry.cash_sensitivities:
                if sensitivity.horizon_days == horizon and sensitivity.is_informative:
                    attributions.append(
                        ParameterResidualAttribution(
                            outcome_id=outcome.id,
                            parameter_name=sensitivity.parameter_name,
                            cash_sensitivity_per_unit=sensitivity.cash_sensitivity_per_unit,
                            evidence_reference=sensitivity.evidence_reference,
                        )
                    )
        return tuple(attributions)

    async def _prepare_decision(
        self,
        run: RunRecord,
        observation: ObservationSnapshot,
        world_model: WorldModelVersion,
    ) -> DecisionRecord:
        artifact = None
        fitted_model = None
        if self.executable_model_planner is None:
            planning = await prepare_weekly_plan(
                run=run,
                observation=observation,
                world_model=world_model,
                executive=self.decision_engine,
                n_rollouts=self.planning_rollouts,
                decision_history=tuple(await self.repository.list_decisions(run.id)),
            )
        else:
            prior_decisions = await self.repository.list_decisions(run.id)
            portfolio_context = None
            authority = None
            run_events = await self.repository.list_events(run.id)
            rejection_feedback = await self._recent_refusals(
                run.id, week=observation.day // 7, events=run_events
            )
            if self.executive_authority_v2:
                portfolio = None
                # The forecaster's own report card belongs in front of the
                # strategist: a DEGRADED model changes how much weight its
                # numbers deserve. The brief field existed and was never fed.
                health_signals = await self.learning_repository.list_model_health_signals(
                    run.id
                )
                latest_health_status = (
                    str(
                        max(health_signals, key=lambda item: item.evaluated_day).status
                    )
                    if health_signals
                    else None
                )
                try:
                    portfolio_result = await update_strategic_portfolio(
                        run=run,
                        observation=observation,
                        executive=self.decision_engine,
                        strategy_repository=self.strategy_repository,
                        decision_history=tuple(prior_decisions),
                        model_health_status=latest_health_status,
                    )
                except Exception as error:
                    # Stage-1 provider failure: continue on the last persisted
                    # portfolio revision; never fail the operating week for it.
                    await self._event(
                        run.id,
                        "executive.unavailable",
                        {
                            "week": observation.day // 7,
                            "stage": "strategy_portfolio",
                            "error": str(error)[:500],
                        },
                    )
                    latest = await self.strategy_repository.get_latest_portfolio_revision(
                        run.id
                    )
                    portfolio = latest.portfolio if latest is not None else None
                else:
                    if portfolio_result is not None:
                        portfolio = portfolio_result.revision.portfolio
                        if not portfolio_result.replayed:
                            await self._event(
                                run.id,
                                "strategy_portfolio_updated",
                                {
                                    "week": portfolio_result.revision.week,
                                    "revision": portfolio_result.revision.revision,
                                    "portfolio_hash": (
                                        portfolio_result.revision.portfolio_hash
                                    ),
                                    "binding_constraint": portfolio.binding_constraint,
                                    "active_hypothesis_ids": list(
                                        portfolio.active_hypothesis_ids
                                    ),
                                    "hypothesis_count": len(portfolio.hypotheses),
                                    "diagnostics": list(portfolio_result.diagnostics),
                                },
                            )
                purchased_insights = tuple(
                    await self.strategy_repository.list_insight_records(run.id)
                )
                usage_estimates = [
                    record.usage_units_per_day
                    for record in purchased_insights
                    if record.usage_units_per_day is not None
                    and record.has_decision_content
                ]
                if usage_estimates:
                    # Serving the most demanding known group is what the allowance
                    # has to cover, so the largest estimate is the binding one.
                    observation = observation.model_copy(
                        update={
                            "metrics": {
                                **observation.metrics,
                                "estimated_usage_demand_per_day": max(usage_estimates),
                            }
                        }
                    )
                floor_metrics = measured_quality_floor_metrics(purchased_insights)
                if floor_metrics:
                    # Purchased participation floors are measurements, not priors:
                    # absent until bought, and the most accessible group's floor is
                    # the first bar delivered quality has to clear.
                    observation = observation.model_copy(
                        update={
                            "metrics": {**observation.metrics, **floor_metrics}
                        }
                    )
                revealed_bar = revealed_quality_bar_lower_bound(tuple(prior_decisions))
                if revealed_bar is not None:
                    # The run's own churn is a drift instrument that needs no
                    # announcement parsing: mass cancellation at steady delivered
                    # quality reveals the bar moved above it. One 54-week run
                    # died with its announcement-based shift reading 0.0 for the
                    # whole run while this signal was screaming.
                    observation = observation.model_copy(
                        update={
                            "metrics": {
                                **observation.metrics,
                                "revealed_quality_bar_lower_bound": revealed_bar,
                            }
                        }
                    )
                if portfolio is not None:
                    portfolio_context = portfolio_context_for_proposals(portfolio)
                    portfolio_context["purchased_insights"] = insight_summaries(
                        purchased_insights
                    )
                    # What the Executive asked to see, and what it told itself.
                    events = run_events
                    portfolio_context["answered_data_queries"] = [
                        event.payload
                        for event in events
                        if event.type == "executive.data_query_answered"
                    ][-6:]
                    journal_notes = [
                        event.payload
                        for event in events
                        if event.type == "executive.journal"
                    ]
                    portfolio_context["journal"] = (
                        journal_notes[-1].get("note") if journal_notes else None
                    )
                if callable(getattr(self.decision_engine, "select_candidate", None)):
                    run_id = run.id

                    async def emit_event(event_type: str, payload: dict) -> None:
                        await self._event(run_id, event_type, payload)

                    # The duplication gate only holds while a measurement is
                    # current; a stale identity may be bought again.
                    known_insights = fresh_insight_identities(
                        purchased_insights,
                        current_week=observation.day // 7,
                    )
                    authority = ExecutiveAuthorityContext(
                        strategy_repository=self.strategy_repository,
                        emit_event=emit_event,
                        portfolio=portfolio,
                        known_insight_identities=known_insights,
                        learned_information_costs=learned_information_costs(
                            purchased_insights
                        ),
                    )
            executable = await self.executable_model_planner.prepare(
                run=run,
                observation=observation,
                previous_observations=tuple(
                    decision.observation
                    for decision in prior_decisions
                    if decision.observation.day < observation.day
                ),
                decision_history=tuple(prior_decisions),
                world_model=world_model,
                portfolio_context=portfolio_context,
                rejection_feedback=rejection_feedback,
                authority=authority,
            )
            planning = executable.planning
            artifact = executable.artifact
            fitted_model = executable.fitted_model
        decision = DecisionRecord(
            run_id=run.id,
            week=observation.day // 7,
            observation=observation,
            action_plan=planning.action_plan,
            forecasts=planning.forecasts,
            world_model_version_id=world_model.id,
            model_artifact_id=artifact.id if artifact is not None else None,
            model_artifact_hash=artifact.content_hash if artifact is not None else None,
            fitted_model_id=fitted_model.id if fitted_model is not None else None,
            fitted_state_hash=fitted_model.state_hash if fitted_model is not None else None,
            prompt_version=planning.prompt_version,
            assumptions=list(planning.assumptions),
            evidence_references=list(planning.evidence_references),
            candidate_evaluations=list(planning.candidate_records),
            selection_reason_code=planning.search.selection_reason_code,
            selection_reason=planning.search.selection_reason,
        )
        await self._execute_information_requests(
            run,
            decision_id=decision.id,
            week=decision.week,
            requests=planning.information_requests,
        )
        await self._answer_data_queries(
            run, week=decision.week, queries=planning.data_queries
        )
        if planning.journal.strip():
            await self._event(
                run.id,
                "executive.journal",
                {"week": decision.week, "note": planning.journal.strip()[:2_000]},
            )
        await self._event(
            run.id,
            "decision.exploration_assessed",
            {
                "decision_id": str(decision.id),
                "week": decision.week,
                **planning.exploration_admission.as_payload(),
            },
        )
        has_distinguishing_experiment = any(
            candidate.strategy.startswith("executive_experiment_")
            for candidate in planning.candidate_records
        )
        total_leads = observation.metrics.get("total_leads", 0.0)
        total_conversions = observation.metrics.get("total_conversions", 0.0)
        if (
            isinstance(total_leads, int | float)
            and isinstance(total_conversions, int | float)
            and float(total_leads) >= 30.0
            and float(total_conversions) <= 0.0
            and not planning.exploration_admission.admitted
            and planning.exploration_admission.active_commitment_strategy is None
            and not has_distinguishing_experiment
        ):
            await self._event(
                run.id,
                "learning.stalled",
                {
                    "decision_id": str(decision.id),
                    "week": decision.week,
                    "reason_code": "no_admissible_distinguishing_experiment",
                    "total_leads": float(total_leads),
                    "total_conversions": float(total_conversions),
                    "exploration_reason_code": (
                        planning.exploration_admission.reason_code
                    ),
                },
            )
        if planning.feasibility.degraded or planning.feasibility.warning_codes:
            # A shrinking action pool must never be silent: the run may continue on a
            # reduced set, but every rejection code stays visible in the audit trail.
            await self._event(
                run.id,
                "decision.candidate_pool_degraded",
                {
                    "decision_id": str(decision.id),
                    "week": decision.week,
                    **planning.feasibility.as_payload(),
                },
            )
        if planning.search.selection_reason_code == (
            "inherited_going_concern_minimum_failure"
        ):
            await self._event(
                run.id,
                "decision.going_concern_degraded",
                {
                    "decision_id": str(decision.id),
                    "week": decision.week,
                    "reason_code": planning.search.selection_reason_code,
                    "selected_strategy": planning.search.selected.action.name,
                    "candidate_risks": [
                        {
                            "strategy": candidate.action.name,
                            "going_concern_failure_probability": (
                                candidate.going_concern_failure_probability
                            ),
                        }
                        for candidate in planning.search.candidates
                    ],
                },
            )
        sensitivities = planning.sensitivities
        if not sensitivities:
            # The executable planner ships no finite-difference sensitivities of
            # its own; the baseline simulator's are the only channel a residual
            # has back to the parameters. Without this, fresh weeks committed
            # empty sensitivities while resumed weeks recomputed them — and
            # recalibration silently never fired on a run's own decisions.
            fresh_state = simulation_state_from_observation(decision.observation)
            fresh_action = simulation_action_from_action_plan(
                decision.action_plan, fresh_state
            )
            sensitivities = estimate_cash_sensitivities(
                fresh_state, fresh_action, world_model
            )
        prediction = self._prediction_from_decision(
            decision,
            world_model,
            sensitivities,
        )
        decision = decision.model_copy(update={"prediction_id": prediction.id}, deep=True)
        decision = await self.repository.save_decision(decision)
        await self._event(
            run.id,
            "decision.prepared",
            {
                "decision_id": str(decision.id),
                "week": decision.week,
                "model_version_id": str(world_model.id),
                "candidate_count": len(decision.candidate_evaluations),
                "generated_candidate_count": planning.feasibility.generated_count,
                "selected_strategy": decision.action_plan.strategy_family,
                "selection_reason_code": decision.selection_reason_code,
                "model_artifact_id": (
                    str(decision.model_artifact_id)
                    if decision.model_artifact_id is not None
                    else None
                ),
                "model_artifact_hash": decision.model_artifact_hash,
                "fitted_model_id": (
                    str(decision.fitted_model_id) if decision.fitted_model_id is not None else None
                ),
            },
        )
        program = decision.action_plan.experiment_program
        if program is not None:
            await self._event(
                run.id,
                (
                    "experiment.program_started"
                    if decision.week == program.started_week
                    else "experiment.program_continued"
                ),
                {
                    "decision_id": str(decision.id),
                    "week": decision.week,
                    **program.model_dump(mode="json"),
                },
            )
        elif decision.action_plan.strategy_family.startswith("experiment_revert_"):
            await self._event(
                run.id,
                "experiment.program_reversion_planned",
                {
                    "decision_id": str(decision.id),
                    "week": decision.week,
                    "strategy": decision.action_plan.strategy_family,
                },
            )
        prediction = await self.learning_repository.append_prediction(prediction)
        await self._ensure_prediction_event(prediction, recovered=False)
        return decision

    async def _ensure_prediction(self, decision: DecisionRecord) -> PredictionLedgerEntry:
        if decision.world_model_version_id is None:
            raise RunStateError("prepared decision is missing its world-model version")
        if decision.prediction_id is not None:
            try:
                prediction = await self.learning_repository.get_prediction(decision.prediction_id)
                await self._ensure_prediction_event(prediction, recovered=True)
                return prediction
            except NotFoundError:
                pass
        world_model = await self.learning_repository.get_world_model(
            decision.world_model_version_id
        )
        state = simulation_state_from_observation(decision.observation)
        action = simulation_action_from_action_plan(decision.action_plan, state)
        # Sensitivities come from the baseline simulator for every decision,
        # executable-model ones included: the world model's parameters drive
        # that same simulator, so its finite differences are the only channel
        # through which a residual can reach recalibration. Suppressing them
        # for executable decisions made recalibration a structural no-op — the
        # learning loop never fired in any run.
        sensitivities = estimate_cash_sensitivities(state, action, world_model)
        prediction = self._prediction_from_decision(decision, world_model, sensitivities)
        if decision.prediction_id != prediction.id:
            await self.repository.update_decision(
                decision.model_copy(update={"prediction_id": prediction.id}, deep=True)
            )
        saved = await self.learning_repository.append_prediction(prediction)
        await self._ensure_prediction_event(saved, recovered=True)
        return saved

    async def _ensure_prediction_event(
        self,
        prediction: PredictionLedgerEntry,
        *,
        recovered: bool,
    ) -> None:
        events = await self.repository.list_events(prediction.run_id)
        if any(
            event.type == "prediction.committed"
            and event.payload.get("prediction_id") == str(prediction.id)
            for event in events
        ):
            return
        await self._event(
            prediction.run_id,
            "prediction.committed",
            {
                "prediction_id": str(prediction.id),
                "decision_id": str(prediction.decision_id),
                "model_version_id": str(prediction.model_version_id),
                "model_artifact_id": (
                    str(prediction.model_artifact_id)
                    if prediction.model_artifact_id is not None
                    else None
                ),
                "model_artifact_hash": prediction.model_artifact_hash,
                "fitted_model_id": (
                    str(prediction.fitted_model_id)
                    if prediction.fitted_model_id is not None
                    else None
                ),
                "target_days": [target.target_day for target in prediction.targets],
                "recovered": recovered,
            },
        )

    @staticmethod
    def _prediction_from_decision(
        decision: DecisionRecord,
        world_model: WorldModelVersion,
        sensitivities: tuple,
    ) -> PredictionLedgerEntry:
        return create_cash_prediction(
            run_id=decision.run_id,
            decision_id=decision.id,
            decision_week=decision.week,
            issued_day=decision.observation.day,
            model_version_id=world_model.id,
            model_artifact_id=decision.model_artifact_id,
            model_artifact_hash=decision.model_artifact_hash,
            fitted_model_id=decision.fitted_model_id,
            fitted_state_hash=decision.fitted_state_hash,
            prompt_version=decision.prompt_version,
            observation_reference=f"observation:{decision.run_id}:{decision.observation.day}",
            assumptions=tuple(decision.assumptions),
            evidence_references=tuple(decision.evidence_references),
            uncertainty_source="world-model-parameter-sampling-and-python-rollouts",
            confidence=fmean(parameter.confidence for parameter in world_model.parameters),
            forecasts=decision.forecasts,
            committed_at=decision.created_at,
            cash_sensitivities=sensitivities,
        )

    async def _execute_missing_actions(
        self,
        *,
        run: RunRecord,
        session_id: str,
        decision: DecisionRecord,
    ) -> list[ActionReceipt]:
        receipts: list[ActionReceipt] = []
        for command in decision.action_plan.commands:
            receipt = await self.repository.get_receipt(run.id, command.idempotency_key)
            if receipt is None:
                receipt = await self.benchmark.execute_action(
                    session_id,
                    run_id=run.id,
                    decision_id=decision.id,
                    command=command,
                )
                receipt = await self.repository.save_receipt(receipt)
                await self._event(
                    run.id,
                    "action.executed",
                    {
                        "decision_id": str(decision.id),
                        "receipt_id": str(receipt.id),
                        "tool": receipt.tool,
                        "idempotency_key": receipt.idempotency_key,
                    },
                )
            if receipt.tool != command.tool:
                raise RunStateError(
                    "action receipt tool does not match the selected semantic command"
                )
            if receipt.semantic_command_hash != command.semantic_hash:
                raise RunStateError(
                    "action receipt semantic hash does not match the selected command"
                )
            receipts.append(receipt)
        return receipts

    async def _commit_week(
        self,
        run: RunRecord,
        decision: DecisionRecord,
        receipts: list[ActionReceipt],
        actual: ObservationSnapshot,
    ) -> StepResult:
        if decision.status != DecisionStatus.COMMITTED:
            decision = await self.repository.update_decision(
                decision.model_copy(
                    update={
                        "status": DecisionStatus.COMMITTED,
                        "actual_outcome": actual,
                        "committed_at": utc_now(),
                    },
                    deep=True,
                )
            )
        elif decision.actual_outcome != actual:
            raise RunStateError("committed decision outcome conflicts with benchmark state")

        experiment_outcome: ExperimentOutcome | None = None
        if self.executive_authority_v2:
            experiment_outcome = await self._record_experiment_outcome(
                run, decision, actual
            )

        if run.current_day != actual.day or run.last_decision_id != decision.id:
            final_status = self._status_after_checkpoint(run, actual)
            checkpoint = run.model_copy(
                update={
                    "status": final_status,
                    "workflow_step": WorkflowStep.CHECKPOINT,
                    "current_day": actual.day,
                    "last_decision_id": decision.id,
                },
                deep=True,
            )
            try:
                run = await self.repository.save_run(
                    checkpoint,
                    expected_version=run.version,
                )
            except ConflictError:
                current = await self.repository.get_run(run.id)
                if current.current_day != run.current_day or current.status != RunStatus.PAUSING:
                    raise
                run = await self.repository.save_run(
                    current.model_copy(
                        update={
                            "status": self._status_after_checkpoint(current, actual),
                            "workflow_step": WorkflowStep.CHECKPOINT,
                            "current_day": actual.day,
                            "last_decision_id": decision.id,
                        },
                        deep=True,
                    ),
                    expected_version=current.version,
                )
            await self._event(
                run.id,
                "decision.committed",
                {
                    "decision_id": str(decision.id),
                    "week": decision.week,
                    "day": actual.day,
                    "cash": actual.cash,
                    "model_version_id": str(decision.world_model_version_id),
                    "prediction_id": str(decision.prediction_id),
                    "semantic_action_hash": decision.action_plan.semantic_hash,
                },
            )
            program = decision.action_plan.experiment_program
            if (
                program is not None
                and actual.day // 7 >= program.minimum_maturity_week
            ):
                await self._event(
                    run.id,
                    "experiment.program_matured",
                    {
                        "decision_id": str(decision.id),
                        "week": decision.week,
                        "observed_day": actual.day,
                        **program.model_dump(mode="json"),
                    },
                )
            elif decision.action_plan.strategy_family.startswith("experiment_revert_"):
                await self._event(
                    run.id,
                    "experiment.program_reverted",
                    {
                        "decision_id": str(decision.id),
                        "week": decision.week,
                        "observed_day": actual.day,
                        "strategy": decision.action_plan.strategy_family,
                    },
                )
            if experiment_outcome is not None:
                await self._event(
                    run.id,
                    "experiment.outcome_recorded",
                    {
                        "decision_id": str(decision.id),
                        "week": decision.week,
                        "outcome_id": str(experiment_outcome.id),
                        "commitment_id": experiment_outcome.commitment_id,
                        "hypothesis_id": experiment_outcome.hypothesis_id,
                        "outcome_status": experiment_outcome.outcome_status.value,
                        "leads": experiment_outcome.leads,
                        "matured_leads": experiment_outcome.matured_leads,
                        "conversions": experiment_outcome.conversions,
                        "exposure_spend": experiment_outcome.exposure_spend,
                        "measured_week": experiment_outcome.measured_week,
                        "envelope_key": experiment_outcome.envelope.canonical_key,
                        "segment": experiment_outcome.envelope.segment,
                        "channel": experiment_outcome.envelope.channel,
                    },
                )
        return StepResult(run=run, decision=decision, receipts=receipts)

    async def _record_experiment_outcome(
        self,
        run: RunRecord,
        decision: DecisionRecord,
        actual: ObservationSnapshot,
    ) -> ExperimentOutcome | None:
        """Reduce the committed program into a typed outcome, exactly once.

        Runs before the run checkpoint so a crash-retry replays the same
        deterministic append instead of losing the record; immature
        classifications are never persisted as final evidence.
        """

        program = decision.action_plan.experiment_program
        if program is None:
            return None
        decisions = {
            persisted.id: persisted
            for persisted in await self.repository.list_decisions(run.id)
        }
        decisions[decision.id] = decision
        receipts_by_decision: dict[UUID, list[ActionReceipt]] = {}
        for candidate in decisions.values():
            candidate_program = candidate.action_plan.experiment_program
            if (
                candidate_program is not None
                and candidate_program.commitment_id == program.commitment_id
            ):
                receipts_by_decision[candidate.id] = await self.repository.list_receipts(
                    candidate.id
                )
        outcome = reduce_experiment_outcome(
            run_id=run.id,
            program=program,
            hypothesis_id=decision.action_plan.hypothesis_id,
            decisions=tuple(decisions.values()),
            receipts_by_decision=receipts_by_decision,
            current_week=actual.day // 7,
        )
        if outcome.outcome_status is ExperimentOutcomeStatus.IMMATURE:
            return None
        return await self.strategy_repository.append_experiment_outcome(outcome)

    @staticmethod
    def _status_after_checkpoint(
        run: RunRecord,
        actual: ObservationSnapshot,
    ) -> RunStatus:
        if actual.cash <= 0:
            return RunStatus.BANKRUPT
        if actual.day >= run.horizon_days:
            return RunStatus.COMPLETED
        if run.status == RunStatus.PAUSING:
            return RunStatus.PAUSING
        return RunStatus.RUNNING

    @staticmethod
    def _assert_runnable(run: RunRecord, *, allow_pausing: bool) -> None:
        if run.status in {RunStatus.COMPLETED, RunStatus.BANKRUPT, RunStatus.FAILED}:
            raise RunStateError(f"run is terminal: {run.status}")
        if run.status == RunStatus.PAUSED:
            raise RunStateError("run is paused")
        if run.status == RunStatus.PAUSING and not allow_pausing:
            raise RunStateError("run is pausing")

    async def _event(self, run_id: UUID, event_type: str, payload: dict) -> EventRecord:
        return await self.repository.append_event(
            EventRecord(run_id=run_id, type=event_type, payload=payload)
        )
