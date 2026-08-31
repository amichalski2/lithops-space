from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from lithops.agents.candidate_model_builder import (
    ACQUISITION_BUILDER,
    PRICING_BUILDER,
    RETENTION_BUILDER,
    CandidateModelBuilder,
)
from lithops.agents.component_architect import (
    SMOOTH_CONVERSION_ARCHITECT,
    THRESHOLD_CONVERSION_ARCHITECT,
    ConversionComponentAuthor,
)
from lithops.agents.executive import ExecutiveDecisionEngine
from lithops.agents.model_coding_agent import (
    ACQUISITION_MODEL_CODER,
    CAPACITY_MODEL_CODER,
    PRICING_MODEL_CODER,
    RETENTION_MODEL_CODER,
    ModelCodingAgent,
)
from lithops.api.schemas import DecisionExplanation, PredictionView, RunReport
from lithops.application.executable_model_challenge import ExecutableModelChallenge
from lithops.application.executable_model_planning import ExecutableModelPlanner
from lithops.application.model_challenge import ModelChallengeOrchestrator
from lithops.application.step_run import RunManager, RunStateError, StaticDecisionEngine
from lithops.benchmark.ceobench import CeobenchAdapter, CeobenchCli
from lithops.benchmark.fake import FakeBenchmarkAdapter
from lithops.config import Settings
from lithops.domain.errors import BenchmarkContractError, NotFoundError, OperationInProgressError
from lithops.domain.models import DecisionRecord, EventRecord, RunRecord, StepResult
from lithops.domain.world_model import WorldModelVersion
from lithops.infrastructure.llm import OpenRouterProvider
from lithops.infrastructure.persistence.repositories import (
    InMemoryRunRepository,
    SupabaseRunRepository,
)

LOGGER = logging.getLogger("lithops.api")


def get_manager(request: Request) -> RunManager:
    return request.app.state.run_manager


RunManagerDependency = Annotated[RunManager, Depends(get_manager)]
IdempotencyKeyHeader = Annotated[
    str,
    Header(min_length=1, max_length=200, alias="Idempotency-Key"),
]
def get_execution_manager(request: Request) -> RunManager:
    if request.app.state.replay_only:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This deployment is a read-only replay; runs cannot be advanced here.",
        )
    return request.app.state.run_manager


ExecutionManagerDependency = Annotated[RunManager, Depends(get_execution_manager)]


def build_manager(settings: Settings | None = None) -> RunManager:
    resolved = settings or Settings.from_env()
    resolved.validate()

    if resolved.storage_backend == "supabase":
        repository = SupabaseRunRepository(
            url=resolved.supabase_url or "",
            secret_key=resolved.supabase_secret_key or "",
        )
    else:
        repository = InMemoryRunRepository()

    if resolved.benchmark_backend == "ceobench":
        executable = Path(resolved.ceobench_executable or "").expanduser().resolve()
        if not executable.is_file():
            raise ValueError(f"CEO-Bench executable does not exist: {executable}")
        command = (
            (resolved.ceobench_python, str(executable))
            if resolved.ceobench_python
            else (str(executable),)
        )
        benchmark = CeobenchAdapter(
            cli=CeobenchCli(
                command=command,
                working_directory=executable.parent,
            ),
            seed=resolved.ceobench_seed,
        )
    else:
        benchmark = FakeBenchmarkAdapter()

    structured_provider = None
    if resolved.model_provider == "openrouter":
        structured_provider = OpenRouterProvider(
            api_key=resolved.openrouter_api_key or "",
            model=resolved.openrouter_model,
            timeout_seconds=180.0,
        )
        decision_engine = ExecutiveDecisionEngine(structured_provider)
    elif resolved.model_provider == "gemini":
        from lithops.infrastructure.llm.gemini_adk_provider import GeminiAdkProvider

        structured_provider = GeminiAdkProvider(
            api_key=resolved.gemini_api_key or "",
            model=resolved.gemini_model,
        )
        decision_engine = ExecutiveDecisionEngine(structured_provider)
    else:
        decision_engine = StaticDecisionEngine()

    challenge_orchestrator = None
    if structured_provider is not None and not resolved.executable_model_planning:
        challenge_orchestrator = ModelChallengeOrchestrator(
            repository=repository,
            builders=tuple(
                CandidateModelBuilder(
                    spec=spec,
                    provider=structured_provider,
                    provider_name=resolved.model_provider,
                )
                for spec in (PRICING_BUILDER, ACQUISITION_BUILDER, RETENTION_BUILDER)
            ),
            builder_timeout_seconds=60.0,
        )

    executable_model_planner = (
        ExecutableModelPlanner(
            repository=repository,
            executive=decision_engine,
        )
        if resolved.executable_model_planning
        else None
    )
    executable_model_challenge = (
        ExecutableModelChallenge(
            repository=repository,
            authors=(
                ConversionComponentAuthor(
                    spec=SMOOTH_CONVERSION_ARCHITECT,
                    provider=structured_provider,
                    provider_name=resolved.model_provider,
                ),
                ConversionComponentAuthor(
                    spec=THRESHOLD_CONVERSION_ARCHITECT,
                    provider=structured_provider,
                    provider_name=resolved.model_provider,
                ),
                *tuple(
                    ModelCodingAgent(
                        spec=spec,
                        provider=structured_provider,
                        provider_name=resolved.model_provider,
                    )
                    for spec in (
                        PRICING_MODEL_CODER,
                        ACQUISITION_MODEL_CODER,
                        RETENTION_MODEL_CODER,
                        CAPACITY_MODEL_CODER,
                    )
                ),
            ),
        )
        if resolved.executable_model_planning and structured_provider is not None
        else None
    )
    return RunManager(
        repository=repository,
        benchmark=benchmark,
        decision_engine=decision_engine,
        model_challenge_orchestrator=challenge_orchestrator,
        model_challenge_cooldown_days=28,
        executable_model_planner=executable_model_planner,
        executable_model_challenge=executable_model_challenge,
        executive_authority_v2=resolved.executive_authority_v2,
    )


def create_app(
    manager: RunManager | None = None,
    *,
    replay_only: bool | None = None,
) -> FastAPI:
    resolved = Settings.from_env() if manager is None else None
    application = FastAPI(
        title="Lithops",
        version="0.1.0",
        description="Auditable weekly execution spine for a self-calibrating CEO-Bench agent.",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Idempotency-Key"],
        expose_headers=["X-Correlation-ID"],
    )
    application.state.run_manager = manager or build_manager(resolved)
    # The public cockpit deployment (static engine over the shared ledger) must never
    # advance a run: a step there would mint decisions no model ever made.
    application.state.replay_only = (
        replay_only
        if replay_only is not None
        else (
            resolved is not None
            and resolved.model_provider == "static"
            and resolved.storage_backend == "supabase"
        )
    )

    @application.middleware("http")
    async def structured_request_log(request: Request, call_next):
        correlation_id = request.headers.get("Idempotency-Key") or uuid4().hex
        request.state.correlation_id = correlation_id
        started = time.perf_counter()
        status_code = 500
        error_type = None
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Correlation-ID"] = correlation_id
            return response
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            LOGGER.info(
                json.dumps(
                    {
                        "severity": "ERROR" if status_code >= 500 else "INFO",
                        "event": "http.request",
                        "correlation_id": correlation_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status": status_code,
                        "latency_ms": round((time.perf_counter() - started) * 1_000, 2),
                        "error_type": error_type,
                    },
                    separators=(",", ":"),
                )
            )

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post("/runs", response_model=RunRecord, status_code=status.HTTP_201_CREATED)
    async def create_run(
        run_manager: RunManagerDependency,
    ) -> RunRecord:
        return await run_manager.create_run()

    @application.get("/runs/{run_id}", response_model=RunRecord)
    async def get_run(
        run_id: UUID,
        run_manager: RunManagerDependency,
    ) -> RunRecord:
        try:
            return await run_manager.get_run(run_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @application.get("/runs/{run_id}/state", response_model=RunRecord)
    async def get_run_state(
        run_id: UUID,
        run_manager: RunManagerDependency,
    ) -> RunRecord:
        try:
            return await run_manager.get_run(run_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @application.post("/runs/{run_id}/start", response_model=RunRecord)
    async def start_run(
        run_id: UUID,
        run_manager: RunManagerDependency,
    ) -> RunRecord:
        try:
            return await run_manager.start_run(run_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RunStateError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.post("/runs/{run_id}/pause", response_model=RunRecord)
    async def pause_run(
        run_id: UUID,
        run_manager: RunManagerDependency,
    ) -> RunRecord:
        try:
            return await run_manager.request_pause(run_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RunStateError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.post("/runs/{run_id}/resume", response_model=RunRecord)
    async def resume_run(
        run_id: UUID,
        run_manager: RunManagerDependency,
    ) -> RunRecord:
        try:
            return await run_manager.resume_run(run_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RunStateError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.get("/runs/{run_id}/decisions", response_model=list[DecisionRecord])
    async def list_decisions(
        run_id: UUID,
        run_manager: RunManagerDependency,
    ) -> list[DecisionRecord]:
        try:
            return await run_manager.list_decisions(run_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @application.get(
        "/runs/{run_id}/decisions/{decision_id}",
        response_model=DecisionExplanation,
    )
    async def get_decision_explanation(
        run_id: UUID,
        decision_id: UUID,
        run_manager: RunManagerDependency,
    ) -> DecisionExplanation:
        try:
            decision = await run_manager.get_decision(run_id, decision_id)
            if decision.world_model_version_id is None or decision.prediction_id is None:
                raise NotFoundError(f"decision learning artifacts are incomplete: {decision_id}")
            world_model = await run_manager.get_world_model(
                run_id,
                decision.world_model_version_id,
            )
            prediction = await run_manager.get_prediction(run_id, decision.prediction_id)
            outcomes = [
                outcome
                for outcome in await run_manager.list_prediction_outcomes(run_id)
                if outcome.ledger_entry_id == prediction.id
            ]
            health_signals = [
                signal
                for signal in await run_manager.list_model_health_signals(run_id)
                if signal.model_version_id == world_model.id
            ]
            return DecisionExplanation(
                decision=decision,
                world_model=world_model,
                prediction=PredictionView(prediction=prediction, outcomes=outcomes),
                model_health_signals=health_signals,
            )
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @application.get("/runs/{run_id}/world-model", response_model=WorldModelVersion)
    async def get_world_model(
        run_id: UUID,
        run_manager: RunManagerDependency,
    ) -> WorldModelVersion:
        try:
            return await run_manager.get_latest_world_model(run_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @application.get("/runs/{run_id}/predictions", response_model=list[PredictionView])
    async def list_predictions(
        run_id: UUID,
        run_manager: RunManagerDependency,
    ) -> list[PredictionView]:
        try:
            predictions = await run_manager.list_predictions(run_id)
            outcomes = await run_manager.list_prediction_outcomes(run_id)
            outcomes_by_prediction = {
                prediction.id: [
                    outcome for outcome in outcomes if outcome.ledger_entry_id == prediction.id
                ]
                for prediction in predictions
            }
            return [
                PredictionView(
                    prediction=prediction,
                    outcomes=outcomes_by_prediction[prediction.id],
                )
                for prediction in predictions
            ]
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @application.post("/runs/{run_id}/step", response_model=StepResult)
    async def step_run(
        run_id: UUID,
        idempotency_key: IdempotencyKeyHeader,
        run_manager: ExecutionManagerDependency,
    ) -> StepResult:
        try:
            return await run_manager.step_run(run_id, request_id=idempotency_key)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except OperationInProgressError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (RunStateError, BenchmarkContractError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.get("/runs/{run_id}/events", response_model=list[EventRecord])
    async def list_events(
        run_id: UUID,
        run_manager: RunManagerDependency,
    ) -> list[EventRecord]:
        try:
            return await run_manager.list_events(run_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @application.get("/runs/{run_id}/report", response_model=RunReport)
    async def get_run_report(
        run_id: UUID,
        run_manager: RunManagerDependency,
    ) -> RunReport:
        try:
            run = await run_manager.get_run(run_id)
            decisions = await run_manager.list_decisions(run_id)
            predictions = await run_manager.list_predictions(run_id)
            outcomes = await run_manager.list_prediction_outcomes(run_id)
            health_signals = await run_manager.list_model_health_signals(run_id)
            try:
                world_model = await run_manager.get_latest_world_model(run_id)
            except NotFoundError:
                world_model = None
            return RunReport(
                run=run,
                decision_count=len(decisions),
                prediction_count=len(predictions),
                matured_outcome_count=len(outcomes),
                world_model_version=world_model.version if world_model else None,
                latest_model_health=health_signals[-1] if health_signals else None,
            )
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    frontend_root_value = os.getenv("LITHOPS_FRONTEND_DIST", "").strip()
    if frontend_root_value:
        frontend_root = Path(frontend_root_value).resolve()
        index_path = frontend_root / "index.html"
        if not index_path.is_file():
            raise ValueError(f"frontend index does not exist: {index_path}")

        @application.get("/{full_path:path}", include_in_schema=False)
        async def frontend_application(full_path: str):
            candidate = (frontend_root / full_path).resolve()
            if candidate.is_relative_to(frontend_root) and candidate.is_file():
                return FileResponse(candidate)
            runtime_config = json.dumps(
                {
                    "demoRunId": os.getenv("LITHOPS_DEMO_RUN_ID") or None,
                    "runMode": "cloud_simulation",
                },
                separators=(",", ":"),
            )
            document = index_path.read_text(encoding="utf-8").replace(
                "window.__LITHOPS_CONFIG__ = {};",
                f"window.__LITHOPS_CONFIG__ = {runtime_config};",
            )
            return HTMLResponse(document)

    return application


app = create_app()
