"""Weekly fitting and planning through the exact persisted executable-model lineage."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from pydantic import JsonValue

from lithops.application.executive_selection import ExecutiveAuthorityContext
from lithops.application.weekly_planning import (
    ExecutiveProposalEngine,
    WeeklyPlanningResult,
    prepare_executable_weekly_plan,
    simulation_state_from_observation,
)
from lithops.domain.executable_model import (
    CompanyModelFitRequest,
    FittedModel,
    ModelArtifact,
    ModelRuntimeKind,
)
from lithops.domain.models import DecisionRecord, ObservationSnapshot, RunRecord
from lithops.domain.ports.executable_model import ExecutableCompanyModel
from lithops.domain.ports.model_registry_repository import ModelRegistryRepository
from lithops.domain.world_model import WorldModelVersion
from lithops.model_runtime import FixedBaselineModel, runtime_for_artifact


@dataclass(frozen=True, slots=True)
class ExecutablePlanningResult:
    planning: WeeklyPlanningResult
    artifact: ModelArtifact
    fitted_model: FittedModel


class ExecutableModelPlanner:
    """Resolve, refit, persist, and use one exact model artifact per weekly decision."""

    def __init__(
        self,
        *,
        repository: ModelRegistryRepository,
        executive: ExecutiveProposalEngine,
        n_rollouts: int = 200,
    ) -> None:
        self.repository = repository
        self.executive = executive
        self.n_rollouts = n_rollouts
        self.baseline = FixedBaselineModel()

    async def prepare(
        self,
        *,
        run: RunRecord,
        observation: ObservationSnapshot,
        previous_observations: tuple[ObservationSnapshot, ...],
        world_model: WorldModelVersion,
        decision_history: tuple[DecisionRecord, ...] = (),
        portfolio_context: dict | None = None,
        rejection_feedback: tuple[dict, ...] | None = None,
        authority: ExecutiveAuthorityContext | None = None,
    ) -> ExecutablePlanningResult:
        runtime = await self._resolve_runtime(run)
        artifact = await self.repository.append_model_artifact(runtime.artifact)
        observations = self._ordered_observations(previous_observations + (observation,))
        fit_request = CompanyModelFitRequest(
            observation_ids=tuple(f"observation:{run.id}:{item.day}" for item in observations),
            training_start_day=observations[0].day,
            training_end_day=observations[-1].day,
            history=tuple(self._history_row(item) for item in observations),
            prior=self._fit_prior(runtime, observations, world_model),
            seed=observation.day + 40_001,
        )
        fitted = runtime.fit(fit_request)
        fitted = await self.repository.append_fitted_model(run.id, fitted)
        planning = await prepare_executable_weekly_plan(
            run=run,
            observation=observation,
            world_model=world_model,
            executive=self.executive,
            runtime=runtime,
            fitted_model=fitted,
            n_rollouts=self.n_rollouts,
            decision_history=decision_history,
            portfolio_context=portfolio_context,
            rejection_feedback=rejection_feedback,
            authority=authority,
        )
        return ExecutablePlanningResult(
            planning=planning,
            artifact=artifact,
            fitted_model=fitted,
        )

    async def _resolve_runtime(self, run: RunRecord) -> ExecutableCompanyModel:
        active = await self.repository.get_active_model(run.id)
        if active is None:
            return self.baseline
        artifact = await self.repository.get_model_artifact(active.artifact_id)
        if artifact.content_hash != active.artifact_hash:
            raise ValueError("active artifact hash does not resolve")
        if artifact.runtime_kind is ModelRuntimeKind.TRUSTED_BASELINE:
            if artifact.id != self.baseline.artifact.id:
                raise ValueError("unknown trusted executable-model artifact")
            return self.baseline
        return runtime_for_artifact(artifact)

    @staticmethod
    def _ordered_observations(
        observations: tuple[ObservationSnapshot, ...],
    ) -> tuple[ObservationSnapshot, ...]:
        by_day = {item.day: item for item in observations}
        return tuple(by_day[day] for day in sorted(by_day))

    @staticmethod
    def _history_row(observation: ObservationSnapshot) -> dict[str, JsonValue]:
        state = simulation_state_from_observation(observation)
        return {"day": observation.day, **state.model_dump(mode="json")}

    @staticmethod
    def _fit_prior(
        runtime: ExecutableCompanyModel,
        observations: tuple[ObservationSnapshot, ...],
        world_model: WorldModelVersion,
    ) -> dict[str, JsonValue]:
        if runtime.artifact.runtime_kind in {
            ModelRuntimeKind.TRUSTED_BASELINE,
            ModelRuntimeKind.TYPED_COMPONENT_ASSEMBLY,
        }:
            return {"legacy_world_model": world_model.model_dump(mode="json")}
        weekly_cash_delta = -25_000.0
        if len(observations) >= 2:
            deltas = [
                (current.cash - previous.cash) / max((current.day - previous.day) / 7.0, 1.0)
                for previous, current in zip(
                    observations,
                    observations[1:],
                    strict=False,
                )
            ]
            weekly_cash_delta = fmean(deltas)
        canonical: dict[str, JsonValue] = {
            "weekly_cash_delta": weekly_cash_delta,
            "marketing_cash_return": 0.5,
            "marketing_saturation_scale_weekly": 10_000.0,
            "price_elasticity": 0.2,
            "churn_sensitivity": 0.2,
            "quality_lag_weeks": 4.0,
        }
        missing = set(runtime.artifact.required_priors) - set(canonical)
        if missing:
            raise ValueError(
                "active artifact requires unsupported priors: " + ", ".join(sorted(missing))
            )
        return {name: canonical[name] for name in runtime.artifact.required_priors}
