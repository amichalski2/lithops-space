"""Rolling-origin evaluation and deterministic champion-challenger selection."""

from __future__ import annotations

import ast
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import fmean, median
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    model_validator,
)

from lithops.domain.executable_model import (
    CompanyModelFitRequest,
    CompanyModelPredictRequest,
    FittedModel,
    ModelArtifact,
    ModelOutcomeDistribution,
    ModelRuntimeKind,
)
from lithops.domain.model_registry import (
    ModelPromotionDecision,
    PromotionDisposition,
    TemporalEvaluationFold,
)
from lithops.domain.ports.executable_model import ExecutableCompanyModel
from lithops.evaluation.interval_math import (
    nearest_rank_quantile,
    weighted_interval_score,
)
from lithops.model_runtime.component_program import InsufficientComponentSupportError

FORECAST_INTERVAL_PROBABILITY = 0.95
TEMPORAL_EVALUATION_HORIZONS_DAYS = (7, 28, 84)
# A distribution whose samples never move cannot be scored as a forecast: its
# interval carries no information and every miss is total. Candidates like that are
# rejected rather than rewarded for a lucky point estimate.
DEGENERACY_RELATIVE_SPREAD = 1e-9


class TemporalObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    observation_id: str = Field(min_length=1, max_length=240)
    day: int = Field(ge=0)
    state: dict[str, JsonValue]
    action_from_previous: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_scored_state(self) -> TemporalObservation:
        for name in ("cash", "revenue_weekly", "customers", "churn_rate"):
            value = self.state.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"temporal observation requires finite {name}")
        return self


class ModelStressCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=120, pattern=r"^[a-z][a-z0-9_]*$")
    state: dict[str, JsonValue]
    action: dict[str, JsonValue]
    horizon_days: int = Field(default=7, ge=1, le=365)


@dataclass(frozen=True)
class TemporalEvaluationPolicy:
    min_training_observations: int = 2
    n_rollouts: int = 50
    complexity_weight: float = 0.02
    runtime_budget_ms: int = 1_000
    runtime_bucket_ms: int = 100
    runtime_weight: float = 0.01
    relative_improvement_required: float = 0.05
    absolute_improvement_required: float = 1e-4

    def __post_init__(self) -> None:
        if self.min_training_observations < 2:
            raise ValueError("temporal evaluation requires at least two training observations")
        if self.n_rollouts < 1:
            raise ValueError("temporal evaluation requires at least one rollout")
        if self.runtime_budget_ms < 1 or self.runtime_bucket_ms < 1:
            raise ValueError("runtime budgets must be positive")
        if not 0 <= self.relative_improvement_required < 1:
            raise ValueError("relative improvement must be in [0, 1)")


@dataclass(frozen=True)
class ArtifactEvaluationResult:
    artifact: ModelArtifact
    fitted_models: tuple[FittedModel, ...]
    folds: tuple[TemporalEvaluationFold, ...]
    stress_cases_passed: tuple[str, ...]
    passed: bool
    failure_codes: tuple[str, ...]

    @property
    def mean_total_score(self) -> float:
        return self._horizon_balanced_mean(
            tuple((self._horizon(fold), fold.total_score) for fold in self.folds)
        )

    @property
    def cash_calibration_by_horizon(self) -> dict[int, dict[str, float]]:
        grouped: dict[int, list[TemporalEvaluationFold]] = {}
        for fold in self.folds:
            grouped.setdefault(self._horizon(fold), []).append(fold)
        result: dict[int, dict[str, float]] = {}
        for horizon, folds in grouped.items():
            scores = [
                float(fold.metrics["cash_weighted_interval_score"])
                for fold in folds
                if isinstance(
                    fold.metrics.get("cash_weighted_interval_score"),
                    int | float,
                )
            ]
            hits = [
                bool(fold.metrics["cash_interval_hit"])
                for fold in folds
                if isinstance(fold.metrics.get("cash_interval_hit"), bool)
            ]
            if len(scores) != len(folds) or len(hits) != len(folds):
                continue
            result[horizon] = {
                "mean_weighted_interval_score": fmean(scores),
                "interval_miss_rate": 1.0 - sum(hits) / len(hits),
                "fold_count": float(len(folds)),
            }
        return result

    @property
    def conversion_local_scores(self) -> dict[tuple[int, str], float]:
        """Proper local scores keyed by comparable horizon/holdout support."""

        result: dict[tuple[int, str], float] = {}
        for fold in self.folds:
            score = fold.metrics.get("conversion_binomial_log_score")
            holdout_id = fold.metrics.get("holdout_observation_id")
            if (
                isinstance(score, int | float)
                and not isinstance(score, bool)
                and isinstance(holdout_id, str)
            ):
                result[(self._horizon(fold), holdout_id)] = float(score)
        return result

    @staticmethod
    def _horizon(fold: TemporalEvaluationFold) -> int:
        raw = fold.metrics.get("horizon_days")
        if isinstance(raw, int | float) and not isinstance(raw, bool):
            return int(raw)
        return fold.holdout_start_day - fold.training_end_day

    @staticmethod
    def _horizon_balanced_mean(values: tuple[tuple[int, float], ...]) -> float:
        if not values:
            return math.inf
        grouped: dict[int, list[float]] = {}
        for horizon, value in values:
            grouped.setdefault(horizon, []).append(value)
        return fmean(fmean(group) for group in grouped.values())

    @property
    def latest_fitted_model(self) -> FittedModel:
        if not self.fitted_models:
            raise ValueError("evaluation result has no fitted model")
        return self.fitted_models[-1]


class TemporalModelEvaluator:
    def __init__(self, policy: TemporalEvaluationPolicy | None = None) -> None:
        self.policy = policy or TemporalEvaluationPolicy()

    def evaluate(
        self,
        *,
        run_id: UUID,
        challenge_id: UUID,
        runtime: ExecutableCompanyModel,
        observations: tuple[TemporalObservation, ...],
        prior: dict[str, JsonValue],
        stress_cases: tuple[ModelStressCase, ...] = (),
        seed: int = 0,
    ) -> ArtifactEvaluationResult:
        failures = list(self._validate_input(observations))
        failures.extend(self._run_artifact_tests(runtime))
        if failures:
            return ArtifactEvaluationResult(
                artifact=runtime.artifact,
                fitted_models=(),
                folds=(),
                stress_cases_passed=(),
                passed=False,
                failure_codes=tuple(failures),
            )

        fitted_models: list[FittedModel] = []
        folds: list[TemporalEvaluationFold] = []
        observation_by_day = {observation.day: observation for observation in observations}
        fold_specs: list[
            tuple[
                tuple[TemporalObservation, ...],
                TemporalObservation,
                TemporalObservation,
                tuple[dict[str, JsonValue], ...],
                int,
            ]
        ] = []
        for origin_index in range(
            self.policy.min_training_observations - 1,
            len(observations) - 1,
        ):
            training = observations[: origin_index + 1]
            origin = observations[origin_index]
            for horizon_days in TEMPORAL_EVALUATION_HORIZONS_DAYS:
                holdout = observation_by_day.get(origin.day + horizon_days)
                if holdout is not None:
                    path = tuple(
                        item.action_from_previous
                        for item in observations[origin_index + 1 :]
                        if origin.day < item.day <= holdout.day
                    )
                    if len(path) != horizon_days // 7:
                        continue
                    fold_specs.append(
                        (
                            training,
                            origin,
                            holdout,
                            path,
                            horizon_days,
                        )
                    )
        for fold_index, (
            training,
            origin,
            holdout,
            policy_action_path,
            horizon_days,
        ) in enumerate(fold_specs):
            try:
                started = time.perf_counter_ns()
                fitted = runtime.fit(
                    CompanyModelFitRequest(
                        observation_ids=tuple(item.observation_id for item in training),
                        training_start_day=training[0].day,
                        training_end_day=training[-1].day,
                        history=tuple({"day": item.day, **item.state} for item in training),
                        prior=prior,
                        seed=seed + fold_index,
                    )
                )
                distribution = runtime.predict(
                    CompanyModelPredictRequest(
                        fitted_model=fitted,
                        state=origin.state,
                        action=policy_action_path[0],
                        policy_action_path=policy_action_path,
                        horizons_days=(horizon_days,),
                        n_rollouts=self.policy.n_rollouts,
                        seed=seed + 10_000 + fold_index,
                    )
                )
                elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
                if self._is_degenerate(distribution):
                    failures.append(f"degenerate_distribution:fold_{fold_index}")
                    break
                predictive_score, interval_metrics = self._prediction_score(
                    distribution,
                    holdout,
                )
                component_metrics = self._component_local_score(
                    distribution,
                    holdout,
                )
                complexity_penalty = self._complexity_penalty(runtime.artifact)
                runtime_penalty = self._runtime_penalty(elapsed_ms)
                total_score = predictive_score + complexity_penalty + runtime_penalty
                fold = TemporalEvaluationFold.create(
                    run_id=run_id,
                    challenge_id=challenge_id,
                    artifact_id=runtime.artifact.id,
                    artifact_hash=runtime.artifact.content_hash,
                    fitted_model_id=fitted.id,
                    fold_index=fold_index,
                    evaluation_seed=seed,
                    training_start_day=training[0].day,
                    training_end_day=training[-1].day,
                    holdout_start_day=holdout.day,
                    holdout_end_day=holdout.day,
                    sample_count=self.policy.n_rollouts,
                    predictive_score=predictive_score,
                    complexity_penalty=complexity_penalty,
                    runtime_penalty=runtime_penalty,
                    total_score=total_score,
                    invariant_gate_passed=True,
                    metrics={
                        "holdout_observation_id": holdout.observation_id,
                        "horizon_days": horizon_days,
                        "forecast_semantics": "observed_policy_action_path",
                        "policy_action_count": len(policy_action_path),
                        **interval_metrics,
                        **component_metrics,
                    },
                )
            except InsufficientComponentSupportError:
                # A scoped structure is evaluated only where its declared parents
                # have overlap. An early unsupported prefix is not evidence that the
                # model is wrong; later supported folds still decide eligibility.
                continue
            except Exception as exc:  # runtime failures must fail closed
                failures.append(self._failure_code(f"fold_{fold_index}", exc))
                break
            fitted_models.append(fitted)
            folds.append(fold)

        passed_stress_cases: list[str] = []
        if not failures and fitted_models:
            latest = fitted_models[-1]
            for index, stress in enumerate(stress_cases):
                try:
                    runtime.predict(
                        CompanyModelPredictRequest(
                            fitted_model=latest,
                            state=stress.state,
                            action=stress.action,
                            horizons_days=(stress.horizon_days,),
                            n_rollouts=min(self.policy.n_rollouts, 10),
                            seed=seed + 20_000 + index,
                        )
                    )
                except Exception as exc:
                    failures.append(self._failure_code(f"stress_{stress.name}", exc))
                else:
                    passed_stress_cases.append(stress.name)

        return ArtifactEvaluationResult(
            artifact=runtime.artifact,
            fitted_models=tuple(fitted_models),
            folds=tuple(folds),
            stress_cases_passed=tuple(passed_stress_cases),
            passed=bool(folds) and not failures,
            failure_codes=tuple(failures),
        )

    @staticmethod
    def _failure_code(stage: str, exc: Exception) -> str:
        code = f"{stage}:{type(exc).__name__}"
        if not isinstance(exc, ValidationError):
            return code
        errors = exc.errors(include_url=False, include_context=False)
        if not errors:
            return code
        first = errors[0]
        location = ".".join(str(part) for part in first.get("loc", ())) or "root"
        error_type = str(first.get("type", "validation_error"))
        return f"{code}:{location}:{error_type}"

    def recommend_promotion(
        self,
        *,
        challenge_id: UUID,
        run_id: UUID,
        decision_day: int,
        champion: ArtifactEvaluationResult,
        candidate: ArtifactEvaluationResult,
        baseline: ArtifactEvaluationResult,
    ) -> ModelPromotionDecision:
        champion_fitted = champion.latest_fitted_model
        if not candidate.passed:
            recovery = self.recommend_baseline_recovery(
                challenge_id=challenge_id,
                run_id=run_id,
                decision_day=decision_day,
                champion=champion,
                baseline=baseline,
                evaluated_candidate=candidate,
            )
            if recovery is not None:
                return recovery
            return ModelPromotionDecision.create(
                challenge_id=challenge_id,
                run_id=run_id,
                decision_day=decision_day,
                champion_artifact_id=champion.artifact.id,
                champion_fitted_model_id=champion_fitted.id,
                candidate_artifact_id=candidate.artifact.id,
                candidate_fitted_model_id=(
                    candidate.latest_fitted_model.id if candidate.fitted_models else None
                ),
                evaluation_fold_ids=tuple(fold.id for fold in candidate.folds),
                disposition=PromotionDisposition.REJECTED,
                reason_code="candidate_failed_evaluation",
                evidence={"failure_codes": list(candidate.failure_codes)},
            )

        candidate_fitted = candidate.latest_fitted_model
        references = tuple(
            result for result in self._unique_results((champion, baseline)) if result.passed
        )
        if not references:
            disposition = PromotionDisposition.REJECTED
            reason_code = "reference_models_failed_evaluation"
            supported = False
        else:
            component_candidate = (
                candidate.artifact.runtime_kind
                is ModelRuntimeKind.TYPED_COMPONENT_ASSEMBLY
            )
            if component_candidate:
                local_references = tuple(
                    reference
                    for reference in references
                    if reference.conversion_local_scores
                )
                local_supported = bool(local_references) and all(
                    self.component_local_supported(candidate, reference)
                    for reference in local_references
                )
                score_supported = all(
                    self._not_materially_worse(
                        candidate.mean_total_score,
                        reference.mean_total_score,
                    )
                    for reference in references
                )
            else:
                local_supported = True
                score_supported = all(
                    self._materially_better(
                        candidate.mean_total_score,
                        reference.mean_total_score,
                    )
                    for reference in references
                )
            cash_supported = all(
                self.cash_calibration_noninferior(candidate, reference)
                for reference in references
            )
            supported = local_supported and score_supported and cash_supported
            disposition = (
                PromotionDisposition.PROMOTED if supported else PromotionDisposition.NO_UPDATE
            )
            reason_code = (
                "candidate_component_locally_better_global_safe"
                if supported and component_candidate
                else "candidate_materially_better"
                if supported
                else "candidate_cash_calibration_inferior"
                if local_supported and score_supported and not cash_supported
                else "candidate_component_global_regression"
                if component_candidate and local_supported and not score_supported
                else "candidate_component_not_locally_better"
                if component_candidate and not local_supported
                else "candidate_not_materially_better"
            )
        if not supported:
            recovery = self.recommend_baseline_recovery(
                challenge_id=challenge_id,
                run_id=run_id,
                decision_day=decision_day,
                champion=champion,
                baseline=baseline,
                evaluated_candidate=candidate,
            )
            if recovery is not None:
                return recovery
        return ModelPromotionDecision.create(
            challenge_id=challenge_id,
            run_id=run_id,
            decision_day=decision_day,
            champion_artifact_id=champion.artifact.id,
            champion_fitted_model_id=champion_fitted.id,
            candidate_artifact_id=candidate.artifact.id,
            candidate_fitted_model_id=candidate_fitted.id,
            evaluation_fold_ids=tuple(fold.id for fold in candidate.folds),
            disposition=disposition,
            reason_code=reason_code,
            evidence={
                "candidate_mean_score": candidate.mean_total_score,
                "champion_mean_score": champion.mean_total_score,
                "baseline_mean_score": baseline.mean_total_score,
                "candidate_cash_calibration": self._cash_calibration_evidence(candidate),
                "champion_cash_calibration": self._cash_calibration_evidence(champion),
                "baseline_cash_calibration": self._cash_calibration_evidence(baseline),
                "candidate_component_local": self._component_local_evidence(candidate),
                "champion_component_local": self._component_local_evidence(champion),
                "baseline_component_local": self._component_local_evidence(baseline),
                "materially_supported": supported,
            },
        )

    def component_local_supported(
        self,
        candidate: ArtifactEvaluationResult,
        reference: ArtifactEvaluationResult,
    ) -> bool:
        candidate_scores = candidate.conversion_local_scores
        reference_scores = reference.conversion_local_scores
        common = tuple(sorted(set(candidate_scores) & set(reference_scores)))
        if len(common) < self.policy.min_training_observations:
            return False
        return self._materially_better(
            fmean(candidate_scores[key] for key in common),
            fmean(reference_scores[key] for key in common),
        )

    def recommend_baseline_recovery(
        self,
        *,
        challenge_id: UUID,
        run_id: UUID,
        decision_day: int,
        champion: ArtifactEvaluationResult,
        baseline: ArtifactEvaluationResult,
        evaluated_candidate: ArtifactEvaluationResult | None = None,
    ) -> ModelPromotionDecision | None:
        """Reactivate the trusted baseline when a sandboxed champion has drifted."""

        if (
            champion.artifact.id == baseline.artifact.id
            or not champion.passed
            or not baseline.passed
            or not self._materially_better(
                baseline.mean_total_score,
                champion.mean_total_score,
            )
            or not self.cash_calibration_noninferior(baseline, champion)
        ):
            return None
        candidate_score = (
            evaluated_candidate.mean_total_score
            if evaluated_candidate is not None and evaluated_candidate.passed
            else None
        )
        return ModelPromotionDecision.create(
            challenge_id=challenge_id,
            run_id=run_id,
            decision_day=decision_day,
            champion_artifact_id=champion.artifact.id,
            champion_fitted_model_id=champion.latest_fitted_model.id,
            candidate_artifact_id=baseline.artifact.id,
            candidate_fitted_model_id=baseline.latest_fitted_model.id,
            evaluation_fold_ids=tuple(fold.id for fold in baseline.folds),
            disposition=PromotionDisposition.PROMOTED,
            reason_code="baseline_materially_better_than_champion",
            evidence={
                "baseline_mean_score": baseline.mean_total_score,
                "champion_mean_score": champion.mean_total_score,
                "baseline_cash_calibration": self._cash_calibration_evidence(baseline),
                "champion_cash_calibration": self._cash_calibration_evidence(champion),
                "evaluated_candidate_artifact_id": (
                    str(evaluated_candidate.artifact.id)
                    if evaluated_candidate is not None
                    else None
                ),
                "evaluated_candidate_mean_score": candidate_score,
                "materially_supported": True,
            },
        )

    @staticmethod
    def cash_calibration_noninferior(
        candidate: ArtifactEvaluationResult,
        reference: ArtifactEvaluationResult,
    ) -> bool:
        candidate_metrics = candidate.cash_calibration_by_horizon
        reference_metrics = reference.cash_calibration_by_horizon
        if not reference_metrics or set(candidate_metrics) != set(reference_metrics):
            return False
        return all(
            candidate_metrics[horizon]["mean_weighted_interval_score"]
            <= reference_metrics[horizon]["mean_weighted_interval_score"]
            and candidate_metrics[horizon]["interval_miss_rate"]
            <= reference_metrics[horizon]["interval_miss_rate"]
            for horizon in reference_metrics
        )

    @staticmethod
    def _cash_calibration_evidence(
        result: ArtifactEvaluationResult,
    ) -> dict[str, JsonValue]:
        return {
            str(horizon): metrics
            for horizon, metrics in result.cash_calibration_by_horizon.items()
        }

    @staticmethod
    def _component_local_evidence(
        result: ArtifactEvaluationResult,
    ) -> dict[str, JsonValue]:
        grouped: dict[int, list[float]] = {}
        for (horizon, _), score in result.conversion_local_scores.items():
            grouped.setdefault(horizon, []).append(score)
        return {
            str(horizon): {
                "mean_binomial_log_score": fmean(scores),
                "fold_count": len(scores),
            }
            for horizon, scores in sorted(grouped.items())
        }

    def _validate_input(
        self,
        observations: tuple[TemporalObservation, ...],
    ) -> Iterable[str]:
        if len(observations) <= self.policy.min_training_observations:
            return ("insufficient_temporal_observations",)
        days = tuple(item.day for item in observations)
        if days != tuple(sorted(set(days))):
            return ("observations_not_strictly_chronological",)
        ids = tuple(item.observation_id for item in observations)
        if len(ids) != len(set(ids)):
            return ("duplicate_observation_id",)
        return ()

    @staticmethod
    def _run_artifact_tests(runtime: ExecutableCompanyModel) -> tuple[str, ...]:
        if not runtime.artifact.tests:
            return ()
        runner = getattr(runtime, "runner", None)
        run_tests = getattr(runner, "run_artifact_tests", None)
        if not callable(run_tests):
            return ("artifact_test_runner_unavailable",)
        results = run_tests(runtime.artifact)
        return tuple(
            f"artifact_test_failed:{result.name}" for result in results if not result.passed
        )

    @staticmethod
    def _is_degenerate(distribution: ModelOutcomeDistribution) -> bool:
        """True when a horizon's cash samples carry no spread at all."""

        if distribution.n_rollouts < 2:
            return False
        for horizon in distribution.horizons_days:
            cash = [
                sample.cash
                for sample in distribution.samples
                if sample.horizon_days == horizon
            ]
            if not cash:
                continue
            scale = max(abs(median(cash)), 1.0)
            if max(cash) - min(cash) <= DEGENERACY_RELATIVE_SPREAD * scale:
                return True
        return False

    @staticmethod
    def _prediction_score(
        distribution: ModelOutcomeDistribution,
        holdout: TemporalObservation,
    ) -> tuple[float, dict[str, JsonValue]]:
        """Score the predictive distribution, not just its centre.

        Each channel is scored with the weighted interval score, so a candidate is
        rewarded for a calibrated interval and penalised for a confident miss. A
        point-mass forecast earns no width credit and pays every miss in full.
        """

        horizon = distribution.horizons_days[0]
        samples = [sample for sample in distribution.samples if sample.horizon_days == horizon]
        actual = holdout.state
        channels = (
            ("cash", [sample.cash for sample in samples], float(actual["cash"]), True),
            (
                "revenue_weekly",
                [sample.revenue_weekly for sample in samples],
                float(actual["revenue_weekly"]),
                True,
            ),
            (
                "customers",
                [sample.customers for sample in samples],
                float(actual["customers"]),
                True,
            ),
            (
                "churn_rate",
                [sample.churn_rate for sample in samples],
                float(actual["churn_rate"]),
                False,
            ),
        )
        scores: list[float] = []
        metrics: dict[str, JsonValue] = {}
        for name, values, observed, normalize in channels:
            point = median(values)
            lower = nearest_rank_quantile(values, 0.025)
            upper = nearest_rank_quantile(values, 0.975)
            score = weighted_interval_score(
                point=point,
                lower=lower,
                upper=upper,
                actual=observed,
                interval_probability=FORECAST_INTERVAL_PROBABILITY,
            )
            if normalize:
                score /= max(abs(observed), 1.0)
            scores.append(score)
            metrics[f"{name}_interval_hit"] = lower <= observed <= upper
            metrics[f"{name}_interval_width"] = upper - lower
            metrics[f"{name}_weighted_interval_score"] = score
        return fmean(scores), metrics

    @staticmethod
    def _component_local_score(
        distribution: ModelOutcomeDistribution,
        holdout: TemporalObservation,
    ) -> dict[str, JsonValue]:
        """Score conversion as a cohort-conditioned binomial likelihood."""

        actual_leads = holdout.state.get("weekly_leads")
        actual_conversions = holdout.state.get("weekly_conversions")
        if (
            not isinstance(actual_leads, int | float)
            or isinstance(actual_leads, bool)
            or not isinstance(actual_conversions, int | float)
            or isinstance(actual_conversions, bool)
            or actual_leads <= 0
        ):
            return {}
        sample_rates = [
            sample.weekly_conversions / sample.weekly_leads
            for sample in distribution.samples
            if sample.weekly_leads is not None
            and sample.weekly_conversions is not None
            and sample.weekly_leads > 0
        ]
        if not sample_rates:
            return {}
        probability = max(1e-9, min(1.0 - 1e-9, median(sample_rates)))
        conversions = max(0.0, min(float(actual_leads), float(actual_conversions)))
        log_score = -(
            conversions * math.log(probability)
            + (float(actual_leads) - conversions) * math.log(1.0 - probability)
        ) / float(actual_leads)
        return {
            "conversion_binomial_log_score": log_score,
            "conversion_probability": probability,
            "conversion_holdout_leads": float(actual_leads),
            "conversion_holdout_conversions": conversions,
        }

    def _complexity_penalty(self, artifact: ModelArtifact) -> float:
        if artifact.source_code is None:
            return 0.0
        tree = ast.parse(artifact.source_code)
        node_ratio = min(sum(1 for _ in ast.walk(tree)) / 2_000, 1.0)
        dependency_ratio = min(len(artifact.dependencies) / 10, 1.0)
        return self.policy.complexity_weight * (0.8 * node_ratio + 0.2 * dependency_ratio)

    def _runtime_penalty(self, runtime_ms: float) -> float:
        bucketed_ms = (
            math.ceil(runtime_ms / self.policy.runtime_bucket_ms) * self.policy.runtime_bucket_ms
        )
        over_budget = max(bucketed_ms - self.policy.runtime_budget_ms, 0)
        return self.policy.runtime_weight * over_budget / self.policy.runtime_budget_ms

    def _materially_better(self, candidate_score: float, reference_score: float) -> bool:
        if reference_score <= self.policy.absolute_improvement_required:
            return False
        required = min(
            reference_score - self.policy.absolute_improvement_required,
            reference_score * (1 - self.policy.relative_improvement_required),
        )
        return candidate_score <= required and candidate_score < reference_score

    def _not_materially_worse(
        self,
        candidate_score: float,
        reference_score: float,
    ) -> bool:
        tolerance = max(
            self.policy.absolute_improvement_required,
            reference_score * self.policy.relative_improvement_required,
        )
        return candidate_score <= reference_score + tolerance

    @staticmethod
    def _unique_results(
        results: tuple[ArtifactEvaluationResult, ...],
    ) -> Iterable[ArtifactEvaluationResult]:
        seen: set[UUID] = set()
        for result in results:
            if result.artifact.id not in seen:
                seen.add(result.artifact.id)
                yield result
