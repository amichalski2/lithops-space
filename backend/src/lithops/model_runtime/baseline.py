"""Adapter that registers the existing hand-written simulator as a trusted baseline."""

from __future__ import annotations

from math import ceil, sqrt
from statistics import fmean

from pydantic import JsonValue

from lithops.domain.economics import AccountingPeriod
from lithops.domain.executable_model import (
    CompanyModelFitRequest,
    CompanyModelPredictRequest,
    FittedModel,
    ModelArtifact,
    ModelOutcomeDistribution,
    ModelOutcomeSample,
    ModelRuntimeKind,
)
from lithops.domain.world_model import WorldModelVersion
from lithops.model_runtime.invariants import evaluate_model_outcomes
from lithops.simulator import (
    SimulationAction,
    SimulationState,
    simulate,
    simulate_action_path,
)
from lithops.simulator.components import (
    BASELINE_TRANSITION_ASSEMBLY,
    TransitionModelAssembly,
)
from lithops.simulator.models import ProcessNoise

# Parameter uncertainty alone collapses as recalibration tightens the world-model
# bounds, which produced zero-width forecast intervals in the live run. These weekly
# innovations keep an irreducible surprise floor that fitting cannot explain away.
BASELINE_PROCESS_NOISE = ProcessNoise(
    acquisition_sigma=0.15,
    churn_sigma=0.01,
    revenue_sigma=0.03,
)
MINIMUM_CASH_FLOW_SIGMA_WEEKLY = 1_000.0
ACTION_CHANNEL_EXPOSURE_FIELDS = tuple(
    f"marketing_spend_{channel}_weekly"
    for channel in (
        "social_media",
        "search_ads",
        "linkedin",
        "content_marketing",
        "referral_program",
    )
)


class FixedBaselineModel:
    """Transitional baseline; generated candidates must beat it before promotion."""

    def __init__(self) -> None:
        self._artifact = ModelArtifact.create(
            name="fixed-baseline-v12",
            protocol_version="1.1",
            runtime_kind=ModelRuntimeKind.TRUSTED_BASELINE,
            scope="full_company",
            hypothesis="Legacy fixed transition model retained only as a baseline and fallback.",
            authoring_agent="lithops_core",
            provider="deterministic",
            model_name="fixed-python-transition-v11",
            prompt_version="not_applicable",
            trusted_entrypoint="lithops.model_runtime.baseline:FixedBaselineModel",
        )

    @property
    def artifact(self) -> ModelArtifact:
        return self._artifact

    def fit(self, request: CompanyModelFitRequest) -> FittedModel:
        raw_model = request.prior.get("legacy_world_model")
        if not isinstance(raw_model, dict):
            raise ValueError("fixed baseline requires prior.legacy_world_model")
        world_model = WorldModelVersion.model_validate(raw_model)
        cash_flow_sigma = self._cash_flow_residual_sigma(request.history)
        return FittedModel.create(
            artifact=self.artifact,
            request=request,
            fitted_state={
                "legacy_world_model": world_model.model_dump(mode="json"),
                "cash_flow_residual_sigma_weekly": cash_flow_sigma,
            },
        )

    def predict(self, request: CompanyModelPredictRequest) -> ModelOutcomeDistribution:
        if request.fitted_model.artifact_id != self.artifact.id:
            raise ValueError("fitted model does not belong to fixed-baseline-v12")
        if any(day % 7 for day in request.horizons_days):
            raise ValueError("fixed baseline currently supports whole-week horizons only")

        raw_model = request.fitted_model.fitted_state.get("legacy_world_model")
        if not isinstance(raw_model, dict):
            raise ValueError("fixed baseline fitted state is missing legacy_world_model")
        world_model = WorldModelVersion.model_validate(raw_model)
        state = SimulationState.model_validate(request.state)
        action = self._simulation_action(request.action, state)
        residual_sigma = request.fitted_model.fitted_state.get(
            "cash_flow_residual_sigma_weekly",
            0.0,
        )
        if not isinstance(residual_sigma, int | float):
            raise ValueError("fixed baseline fitted cash-flow sigma is invalid")
        cash_flow_sigma = max(
            float(residual_sigma),
            MINIMUM_CASH_FLOW_SIGMA_WEEKLY,
        )
        process_noise = BASELINE_PROCESS_NOISE.model_copy(
            update={"cash_flow_sigma": cash_flow_sigma}
        )
        max_weeks = max(request.horizons_days) // 7
        if request.policy_action_path:
            actions: list[SimulationAction] = []
            path_state = state
            for raw_action in request.policy_action_path:
                next_action = self._simulation_action(raw_action, path_state)
                actions.append(next_action)
                # Expiry fields use absolute simulator weeks; advance just the
                # clock for adapting the next historical action.  The real state
                # evolution happens independently inside every rollout below.
                path_state = path_state.model_copy(update={"week": path_state.week + 1})
            rollouts = simulate_action_path(
                state=state,
                actions=tuple(actions),
                world_model=world_model,
                n_rollouts=request.n_rollouts,
                seed=request.seed,
                process_noise=process_noise,
                assembly=self._transition_assembly(request.fitted_model),
            )
        else:
            rollouts = simulate(
                state=state,
                action=action,
                world_model=world_model,
                horizon_weeks=max_weeks,
                n_rollouts=request.n_rollouts,
                seed=request.seed,
                process_noise=process_noise,
                assembly=self._transition_assembly(request.fitted_model),
            )
        samples = tuple(
            ModelOutcomeSample(
                rollout_index=rollout.rollout_index,
                horizon_days=horizon,
                cash=rollout.states[horizon // 7].cash,
                revenue_weekly=rollout.states[horizon // 7].revenue_weekly,
                customers=rollout.states[horizon // 7].customers,
                churn_rate=rollout.states[horizon // 7].churn_rate,
                weekly_leads=rollout.states[horizon // 7].weekly_leads,
                weekly_conversions=rollout.states[horizon // 7].weekly_conversions,
                accounting=self._accounting_period(
                    initial_state=state,
                    rollout_states=rollout.states,
                    horizon_days=horizon,
                ),
            )
            for rollout in rollouts
            for horizon in request.horizons_days
        )
        distribution = ModelOutcomeDistribution(
            artifact_id=self.artifact.id,
            artifact_hash=self.artifact.content_hash,
            fitted_model_id=request.fitted_model.id,
            horizons_days=request.horizons_days,
            n_rollouts=request.n_rollouts,
            samples=samples,
        )
        invariant_report = evaluate_model_outcomes(distribution)
        if not invariant_report.valid:
            codes = ", ".join(
                sorted({violation.code.value for violation in invariant_report.violations})
            )
            raise ValueError(f"fixed baseline failed economic invariants: {codes}")
        return distribution

    def _transition_assembly(
        self,
        fitted_model: FittedModel,
    ) -> TransitionModelAssembly:
        return BASELINE_TRANSITION_ASSEMBLY

    @staticmethod
    def _simulation_action(
        raw_action: dict[str, JsonValue],
        state: SimulationState,
    ) -> SimulationAction:
        """Adapt the shared executable-model action contract to simulator fields."""

        payload = dict(raw_action)
        experiment_duration = payload.pop("experiment_duration_weeks", None)
        development_duration = payload.pop(
            "development_spend_duration_weeks", None
        )
        promotion_duration = payload.pop("lead_promotion_duration_weeks", None)
        payload.pop("targeted_development_spend_weekly", None)
        targeted_development_duration = payload.pop(
            "targeted_development_duration_weeks", None
        )
        marketing_start_delay = payload.pop("marketing_spend_start_after_weeks", None)
        if (
            targeted_development_duration is not None
            and payload.get("targeted_development_spend_until_week") is None
        ):
            payload["targeted_development_spend_until_week"] = state.week + max(
                1, int(targeted_development_duration)
            )
        if (
            marketing_start_delay is not None
            and float(marketing_start_delay) > 0.0
            and payload.get("marketing_spend_start_week") is None
        ):
            payload["marketing_spend_start_week"] = state.week + int(
                marketing_start_delay
            )
        channel_exposures = {
            name: payload.pop(name)
            for name in ACTION_CHANNEL_EXPOSURE_FIELDS
            if name in payload
        }
        numeric_exposures = tuple(
            float(value)
            for value in channel_exposures.values()
            if isinstance(value, int | float) and not isinstance(value, bool)
        )
        if len(numeric_exposures) != len(channel_exposures) or any(
            value < 0 for value in numeric_exposures
        ):
            raise ValueError("action channel exposures must be non-negative numbers")
        if numeric_exposures:
            marketing_spend = payload.get("marketing_spend")
            if not isinstance(marketing_spend, int | float) or isinstance(
                marketing_spend,
                bool,
            ):
                raise ValueError("action channel exposures require marketing_spend")
            tolerance = max(1e-6, abs(float(marketing_spend)) * 1e-9)
            if abs(sum(numeric_exposures) - float(marketing_spend)) > tolerance:
                raise ValueError("action channel exposures must reconcile to marketing_spend")

        def expiry(duration: JsonValue | None) -> int | None:
            if not isinstance(duration, int | float) or duration <= 0:
                return None
            return state.week + max(1, ceil(float(duration)))

        experiment_expiry = expiry(experiment_duration)
        if (
            experiment_expiry is not None
            and payload.get("marketing_spend_until_week") is None
        ):
            payload["marketing_spend_until_week"] = experiment_expiry
        development_expiry = expiry(development_duration) or experiment_expiry
        if (
            development_expiry is not None
            and payload.get("development_spend_until_week") is None
        ):
            payload["development_spend_until_week"] = development_expiry
        promotion_expiry = expiry(promotion_duration)
        if (
            promotion_expiry is not None
            and payload.get("lead_promotion_until_week") is None
        ):
            payload["lead_promotion_until_week"] = promotion_expiry
        return SimulationAction.model_validate(payload)

    @staticmethod
    def _accounting_period(
        *,
        initial_state: SimulationState,
        rollout_states: tuple[SimulationState, ...],
        horizon_days: int,
    ) -> AccountingPeriod:
        horizon_weeks = horizon_days // 7
        states = rollout_states[1 : horizon_weeks + 1]
        return AccountingPeriod(
            period_days=horizon_days,
            starting_cash=initial_state.cash,
            recognized_revenue=sum(item.revenue_weekly for item in states),
            other_inflows=sum(
                max(item.cash_flow_adjustment_weekly, 0.0) for item in states
            ),
            operating_cost=sum(
                item.customers * item.operating_cost_per_customer_weekly for item in states
            ),
            # Support and reliability work is deducted from cash by the state
            # transition, so the bridge must declare it or the period stops
            # reconciling the moment the control is used at all.
            operations_spend=sum(
                item.operations_spend + item.targeted_ops_spend for item in states
            ),
            capacity_spend=sum(item.capacity_spend_weekly for item in states),
            marketing_spend=sum(item.marketing_spend for item in states),
            development_spend=sum(
                item.development_spend
                + item.targeted_development_spend
                # research_spend is deducted by the transition in the week a
                # programme starts; the bridge declares it with development so
                # the period keeps reconciling when the lever is used at all.
                + item.research_spend_weekly
                for item in states
            ),
            other_outflows=sum(
                max(-item.cash_flow_adjustment_weekly, 0.0) for item in states
            ),
            ending_cash=states[-1].cash,
        )

    @staticmethod
    def _cash_flow_residual_sigma(history: tuple[dict[str, JsonValue], ...]) -> float:
        """Estimate unexplained weekly cash movement from observed cash bridges."""

        residuals: list[float] = []

        def number(row: dict[str, JsonValue], name: str) -> float:
            value = row.get(name, 0.0)
            return float(value) if isinstance(value, int | float) else 0.0

        for previous, current in zip(history, history[1:], strict=False):
            elapsed_weeks = max((number(current, "day") - number(previous, "day")) / 7.0, 1.0)
            observed_delta = (
                number(current, "cash") - number(previous, "cash")
            ) / elapsed_weeks
            expected_delta = (
                number(current, "revenue_weekly")
                - number(current, "customers")
                * number(current, "operating_cost_per_customer_weekly")
                - number(current, "operations_spend")
                - number(current, "targeted_ops_spend")
                - number(current, "capacity_spend_weekly")
                - number(current, "marketing_spend")
                - number(current, "development_spend")
                - number(current, "targeted_development_spend")
            )
            residuals.append(observed_delta - expected_delta)
        if not residuals:
            return 0.0
        return sqrt(fmean(residual * residual for residual in residuals))

    def diagnostics(self, fitted_model: FittedModel) -> dict[str, JsonValue]:
        if fitted_model.artifact_id != self.artifact.id:
            raise ValueError("fitted model does not belong to fixed-baseline-v12")
        raw_model = fitted_model.fitted_state.get("legacy_world_model")
        if not isinstance(raw_model, dict):
            raise ValueError("fixed baseline fitted state is missing legacy_world_model")
        world_model = WorldModelVersion.model_validate(raw_model)
        return {
            "baseline": True,
            "cash_flow_residual_sigma_weekly": float(
                fitted_model.fitted_state.get("cash_flow_residual_sigma_weekly", 0.0)
            ),
            "parameter_count": len(world_model.parameters),
            "relationship_count": len(world_model.relationships),
            "source_observation_day": world_model.source_observation_day,
        }
