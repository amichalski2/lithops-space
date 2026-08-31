from __future__ import annotations

from uuid import UUID

from lithops.domain.component_program import (
    ConversionComponentProgram,
    ConversionFeature,
    ConversionLink,
)
from lithops.domain.economics import AccountingPeriod
from lithops.domain.executable_model import (
    CompanyModelFitRequest,
    CompanyModelPredictRequest,
    FittedModel,
    ModelArtifact,
    ModelArtifactAssertion,
    ModelArtifactTestCase,
    ModelEntrypoint,
    ModelOutcomeDistribution,
    ModelOutcomeSample,
    ModelRuntimeKind,
)
from lithops.domain.model_registry import PromotionDisposition
from lithops.model_runtime.component_program import InsufficientComponentSupportError
from lithops.model_runtime.temporal_evaluation import (
    ModelStressCase,
    TemporalEvaluationPolicy,
    TemporalModelEvaluator,
    TemporalObservation,
)

RUN_ID = UUID("88888888-8888-8888-8888-888888888888")
CHALLENGE_ID = UUID("99999999-9999-9999-9999-999999999999")


class TrendRuntime:
    # Class-level default so subclasses that override __init__ still resolve it.
    degenerate = False

    def __init__(
        self,
        name: str,
        weekly_cash_delta: float,
        *,
        fail_predict: bool = False,
        declares_tests: bool = False,
        degenerate: bool = False,
        customer_offset: float = 0.0,
    ):
        self.weekly_cash_delta = weekly_cash_delta
        self.fail_predict = fail_predict
        self.degenerate = degenerate
        self.customer_offset = customer_offset
        self._artifact = ModelArtifact.create(
            name=name,
            runtime_kind=ModelRuntimeKind.TRUSTED_BASELINE,
            scope="cash",
            hypothesis=f"{name} cash trend",
            authoring_agent="temporal-test",
            provider="deterministic",
            model_name=name,
            prompt_version="test-v1",
            trusted_entrypoint="tests.temporal:TrendRuntime",
            tests=(
                ModelArtifactTestCase(
                    name="declared_test",
                    entrypoint=ModelEntrypoint.DIAGNOSTICS,
                    arguments={"fitted": {}},
                    assertions=(
                        ModelArtifactAssertion(
                            path="ok",
                            operator="equals",
                            expected=True,
                        ),
                    ),
                ),
            )
            if declares_tests
            else (),
        )

    @property
    def artifact(self) -> ModelArtifact:
        return self._artifact

    def fit(self, request: CompanyModelFitRequest) -> FittedModel:
        return FittedModel.create(
            artifact=self.artifact,
            request=request,
            fitted_state={"weekly_cash_delta": self.weekly_cash_delta},
        )

    def predict(self, request: CompanyModelPredictRequest) -> ModelOutcomeDistribution:
        if self.fail_predict:
            raise RuntimeError("unsupported candidate")
        samples = []
        state = request.state
        revenue = float(state["revenue_weekly"])
        for rollout_index in range(request.n_rollouts):
            for horizon_days in request.horizons_days:
                weeks = horizon_days / 7
                spread = 0.0
                if not self.degenerate:
                    unit = (
                        (request.seed * 9301 + rollout_index * 49297) % 233280
                    ) / 233280.0
                    spread = (unit - 0.5) * 0.04 * max(abs(float(state["cash"])), 1.0)
                ending_cash = (
                    float(state["cash"])
                    + (
                        sum(
                            float(action.get("weekly_cash_delta", self.weekly_cash_delta))
                            for action in request.policy_action_path[: int(weeks)]
                        )
                        if request.policy_action_path
                        else self.weekly_cash_delta * weeks
                    )
                    + spread
                )
                samples.append(
                    ModelOutcomeSample(
                        rollout_index=rollout_index,
                        horizon_days=horizon_days,
                        cash=ending_cash,
                        revenue_weekly=revenue,
                        customers=float(state["customers"]) + self.customer_offset,
                        churn_rate=float(state["churn_rate"]),
                        accounting=AccountingPeriod(
                            period_days=horizon_days,
                            starting_cash=float(state["cash"]),
                            recognized_revenue=revenue * weeks,
                            operating_cost=0,
                            marketing_spend=0,
                            development_spend=0,
                            other_outflows=(
                                (revenue - self.weekly_cash_delta) * weeks - spread
                            ),
                            ending_cash=ending_cash,
                        ),
                    )
                )
        return ModelOutcomeDistribution(
            artifact_id=self.artifact.id,
            artifact_hash=self.artifact.content_hash,
            fitted_model_id=request.fitted_model.id,
            horizons_days=request.horizons_days,
            n_rollouts=request.n_rollouts,
            samples=tuple(samples),
        )

    def diagnostics(self, fitted_model: FittedModel) -> dict:
        return fitted_model.fitted_state


class SupportAwareTrendRuntime(TrendRuntime):
    def fit(self, request: CompanyModelFitRequest) -> FittedModel:
        if len(request.history) < 3:
            raise InsufficientComponentSupportError("two regimes are not observed yet")
        return super().fit(request)


class ConversionTrendRuntime(TrendRuntime):
    """Test runtime exposing a typed conversion component's local predictions."""

    def __init__(
        self,
        name: str,
        weekly_cash_delta: float,
        conversion_probability: float,
    ) -> None:
        super().__init__(name, weekly_cash_delta)
        self.conversion_probability = conversion_probability
        self._artifact = ModelArtifact.create(
            name=name,
            runtime_kind=ModelRuntimeKind.TYPED_COMPONENT_ASSEMBLY,
            scope="conversion",
            hypothesis=f"{name} conversion response",
            authoring_agent="temporal-test",
            provider="deterministic",
            model_name=name,
            prompt_version="test-v1",
            component_program=ConversionComponentProgram(
                name=name.replace("-", "_"),
                link=ConversionLink.LOGISTIC,
                features=(ConversionFeature.PRODUCT_QUALITY,),
                rationale="Exercise two-stage local and global component evaluation.",
                falsifiers=("Observed conversion likelihood is worse than baseline.",),
            ),
        )

    def predict(self, request: CompanyModelPredictRequest) -> ModelOutcomeDistribution:
        distribution = super().predict(request)
        return distribution.model_copy(
            update={
                "samples": tuple(
                    sample.model_copy(
                        update={
                            "weekly_leads": 100.0,
                            "weekly_conversions": 100.0
                            * self.conversion_probability,
                        }
                    )
                    for sample in distribution.samples
                )
            }
        )


def observations() -> tuple[TemporalObservation, ...]:
    def state(cash: float) -> dict:
        return {
            "cash": cash,
            "revenue_weekly": 100.0,
            "customers": 10.0,
            "churn_rate": 0.05,
        }

    return tuple(
        TemporalObservation(
            observation_id=f"obs-{day}",
            day=day,
            state=state(cash),
            action_from_previous={"name": "hold"},
        )
        for day, cash in ((0, 1000.0), (7, 900.0), (14, 800.0), (21, 700.0))
    )


def conversion_observations() -> tuple[TemporalObservation, ...]:
    return tuple(
        observation.model_copy(
            update={
                "state": {
                    **observation.state,
                    "weekly_leads": 100.0,
                    "weekly_conversions": 20.0,
                }
            }
        )
        for observation in observations()
    )


def evaluator() -> TemporalModelEvaluator:
    return TemporalModelEvaluator(
        TemporalEvaluationPolicy(
            min_training_observations=2,
            # Enough rollouts for the 2.5/97.5 nearest-rank quantiles to be meaningful;
            # with three samples the upper bound collapses onto the median.
            n_rollouts=40,
            runtime_budget_ms=10_000,
        )
    )


def test_temporal_evaluation_skips_unsupported_prefixes_but_scores_later_folds() -> None:
    result = evaluator().evaluate(
        run_id=RUN_ID,
        challenge_id=CHALLENGE_ID,
        runtime=SupportAwareTrendRuntime("supported-later", -100),
        observations=observations(),
        prior={},
        seed=2,
    )

    assert result.passed
    assert len(result.folds) == 1
    assert result.folds[0].fold_index == 1
    assert result.failure_codes == ()


def test_rolling_origin_candidate_beats_champion_and_baseline() -> None:
    service = evaluator()
    series = observations()
    stress = ModelStressCase(
        name="low_cash",
        state=series[-1].state,
        action={"name": "hold"},
        horizon_days=7,
    )
    candidate = service.evaluate(
        run_id=RUN_ID,
        challenge_id=CHALLENGE_ID,
        runtime=TrendRuntime("candidate-exact", -100.0),
        observations=series,
        prior={},
        stress_cases=(stress,),
        seed=10,
    )
    champion = service.evaluate(
        run_id=RUN_ID,
        challenge_id=CHALLENGE_ID,
        runtime=TrendRuntime("champion-wrong", -50.0),
        observations=series,
        prior={},
        stress_cases=(stress,),
        seed=10,
    )
    baseline = service.evaluate(
        run_id=RUN_ID,
        challenge_id=CHALLENGE_ID,
        runtime=TrendRuntime("baseline-worse", -25.0),
        observations=series,
        prior={},
        stress_cases=(stress,),
        seed=10,
    )

    decision = service.recommend_promotion(
        challenge_id=RUN_ID,
        run_id=RUN_ID,
        decision_day=21,
        champion=champion,
        candidate=candidate,
        baseline=baseline,
    )

    assert candidate.passed
    assert len(candidate.folds) == 2
    assert candidate.stress_cases_passed == ("low_cash",)
    # The exact-trend candidate no longer scores a perfect zero: interval width is
    # now part of the score, so an honest forecast pays for the uncertainty it admits.
    assert candidate.mean_total_score > 0
    assert candidate.mean_total_score < champion.mean_total_score < baseline.mean_total_score
    assert decision.disposition == PromotionDisposition.PROMOTED
    assert decision.evaluation_fold_ids == tuple(fold.id for fold in candidate.folds)
    assert candidate.folds[0].metrics["cash_interval_hit"] is True
    assert candidate.folds[0].metrics["cash_interval_width"] > 0


def test_typed_component_can_win_locally_without_rewriting_global_score() -> None:
    service = evaluator()
    series = conversion_observations()
    baseline = service.evaluate(
        run_id=RUN_ID,
        challenge_id=CHALLENGE_ID,
        runtime=ConversionTrendRuntime("baseline-conversion", -100.0, 0.05),
        observations=series,
        prior={},
    )
    candidate = service.evaluate(
        run_id=RUN_ID,
        challenge_id=CHALLENGE_ID,
        runtime=ConversionTrendRuntime("candidate-conversion", -100.0, 0.20),
        observations=series,
        prior={},
    )

    decision = service.recommend_promotion(
        challenge_id=CHALLENGE_ID,
        run_id=RUN_ID,
        decision_day=21,
        champion=baseline,
        candidate=candidate,
        baseline=baseline,
    )

    assert candidate.mean_total_score == baseline.mean_total_score
    assert service.component_local_supported(candidate, baseline)
    assert decision.disposition == PromotionDisposition.PROMOTED
    assert decision.reason_code == "candidate_component_locally_better_global_safe"
    assert decision.evidence["candidate_component_local"]


def test_local_component_win_cannot_hide_global_regression() -> None:
    service = evaluator()
    series = conversion_observations()
    baseline = service.evaluate(
        run_id=RUN_ID,
        challenge_id=CHALLENGE_ID,
        runtime=ConversionTrendRuntime("safe-conversion", -100.0, 0.05),
        observations=series,
        prior={},
    )
    candidate = service.evaluate(
        run_id=RUN_ID,
        challenge_id=CHALLENGE_ID,
        runtime=ConversionTrendRuntime("unsafe-conversion", -250.0, 0.20),
        observations=series,
        prior={},
    )

    decision = service.recommend_promotion(
        challenge_id=CHALLENGE_ID,
        run_id=RUN_ID,
        decision_day=21,
        champion=baseline,
        candidate=candidate,
        baseline=baseline,
    )

    assert service.component_local_supported(candidate, baseline)
    assert decision.disposition == PromotionDisposition.NO_UPDATE
    assert decision.reason_code == "candidate_component_global_regression"


def test_degenerate_point_mass_candidate_is_rejected() -> None:
    service = evaluator()
    series = observations()
    champion = service.evaluate(
        run_id=RUN_ID,
        challenge_id=CHALLENGE_ID,
        runtime=TrendRuntime("champion-drift", -140.0),
        observations=series,
        prior={},
    )
    baseline = service.evaluate(
        run_id=RUN_ID,
        challenge_id=CHALLENGE_ID,
        runtime=TrendRuntime("baseline-drift", -200.0),
        observations=series,
        prior={},
    )
    candidate = service.evaluate(
        run_id=RUN_ID,
        challenge_id=CHALLENGE_ID,
        runtime=TrendRuntime("candidate-point-mass", -100.0, degenerate=True),
        observations=series,
        prior={},
    )

    decision = service.recommend_promotion(
        challenge_id=RUN_ID,
        run_id=RUN_ID,
        decision_day=21,
        champion=champion,
        candidate=candidate,
        baseline=baseline,
    )

    assert not candidate.passed
    assert candidate.failure_codes == ("degenerate_distribution:fold_0",)
    assert candidate.folds == ()
    assert decision.disposition == PromotionDisposition.REJECTED
    assert decision.reason_code == "candidate_failed_evaluation"


def test_equal_candidate_produces_no_update() -> None:
    service = evaluator()
    series = observations()
    champion = service.evaluate(
        run_id=RUN_ID,
        challenge_id=CHALLENGE_ID,
        runtime=TrendRuntime("champion-exact", -100.0),
        observations=series,
        prior={},
    )
    candidate = service.evaluate(
        run_id=RUN_ID,
        challenge_id=CHALLENGE_ID,
        runtime=TrendRuntime("candidate-equal", -100.0),
        observations=series,
        prior={},
    )
    baseline = service.evaluate(
        run_id=RUN_ID,
        challenge_id=CHALLENGE_ID,
        runtime=TrendRuntime("baseline-equal", -100.0),
        observations=series,
        prior={},
    )

    decision = service.recommend_promotion(
        challenge_id=RUN_ID,
        run_id=RUN_ID,
        decision_day=21,
        champion=champion,
        candidate=candidate,
        baseline=baseline,
    )

    assert decision.disposition == PromotionDisposition.NO_UPDATE
    assert decision.reason_code == "candidate_not_materially_better"


def test_operational_fit_cannot_hide_inferior_cash_calibration() -> None:
    service = evaluator()
    series = observations()
    champion = service.evaluate(
        run_id=RUN_ID,
        challenge_id=CHALLENGE_ID,
        runtime=TrendRuntime(
            "champion-cash-calibrated",
            -100.0,
            customer_offset=100.0,
        ),
        observations=series,
        prior={},
    )
    candidate = service.evaluate(
        run_id=RUN_ID,
        challenge_id=CHALLENGE_ID,
        runtime=TrendRuntime("candidate-commercially-exact", -150.0),
        observations=series,
        prior={},
    )

    decision = service.recommend_promotion(
        challenge_id=CHALLENGE_ID,
        run_id=RUN_ID,
        decision_day=21,
        champion=champion,
        candidate=candidate,
        baseline=champion,
    )

    assert candidate.mean_total_score < champion.mean_total_score
    assert not service.cash_calibration_noninferior(candidate, champion)
    assert decision.disposition == PromotionDisposition.NO_UPDATE
    assert decision.reason_code == "candidate_cash_calibration_inferior"
    assert decision.evidence["materially_supported"] is False


def test_temporal_evaluation_adds_28_and_84_day_rolling_folds() -> None:
    series = tuple(
        TemporalObservation(
            observation_id=f"long-{day}",
            day=day,
            state={
                "cash": 1_000.0 - 100.0 * (day / 7),
                "revenue_weekly": 100.0,
                "customers": 10.0,
                "churn_rate": 0.05,
            },
            action_from_previous={"name": "hold"},
        )
        for day in range(0, 92, 7)
    )

    result = evaluator().evaluate(
        run_id=RUN_ID,
        challenge_id=CHALLENGE_ID,
        runtime=TrendRuntime("multi-horizon-exact", -100.0),
        observations=series,
        prior={},
    )
    by_horizon = {
        horizon: sum(
            fold.metrics["horizon_days"] == horizon for fold in result.folds
        )
        for horizon in (7, 28, 84)
    }

    assert result.passed
    assert by_horizon == {7: 12, 28: 9, 84: 1}
    assert set(result.cash_calibration_by_horizon) == {7, 28, 84}
    assert all(
        fold.metrics["forecast_semantics"] == "observed_policy_action_path"
        for fold in result.folds
    )
    assert all(
        fold.metrics["policy_action_count"]
        == fold.metrics["horizon_days"] // 7
        for fold in result.folds
    )


def test_long_fold_uses_later_committed_actions_instead_of_freezing_origin_action() -> None:
    deltas = (-100.0, -300.0, 50.0, -25.0, -10.0)
    cash = 1_000.0
    rows = [
        TemporalObservation(
            observation_id="path-0",
            day=0,
            state={
                "cash": cash,
                "revenue_weekly": 100.0,
                "customers": 10.0,
                "churn_rate": 0.05,
            },
            action_from_previous={"name": "initial"},
        )
    ]
    for week, delta in enumerate(deltas, start=1):
        cash += delta
        rows.append(
            TemporalObservation(
                observation_id=f"path-{week}",
                day=week * 7,
                state={
                    "cash": cash,
                    "revenue_weekly": 100.0,
                    "customers": 10.0,
                    "churn_rate": 0.05,
                },
                action_from_previous={
                    "name": f"action-{week}",
                    "weekly_cash_delta": delta,
                },
            )
        )

    result = evaluator().evaluate(
        run_id=RUN_ID,
        challenge_id=CHALLENGE_ID,
        runtime=TrendRuntime("policy-path-exact", -999.0),
        observations=tuple(rows),
        prior={},
    )
    long_fold = next(
        fold
        for fold in result.folds
        if fold.training_end_day == 7 and fold.metrics["horizon_days"] == 28
    )

    assert long_fold.metrics["policy_action_count"] == 4
    assert long_fold.metrics["cash_interval_hit"] is True


def test_materially_better_baseline_replaces_a_drifted_champion() -> None:
    service = evaluator()
    series = observations()
    champion = service.evaluate(
        run_id=RUN_ID,
        challenge_id=CHALLENGE_ID,
        runtime=TrendRuntime("champion-drifted", -350.0),
        observations=series,
        prior={},
    )
    baseline = service.evaluate(
        run_id=RUN_ID,
        challenge_id=CHALLENGE_ID,
        runtime=TrendRuntime("baseline-recovered", -100.0),
        observations=series,
        prior={},
    )
    candidate = service.evaluate(
        run_id=RUN_ID,
        challenge_id=CHALLENGE_ID,
        runtime=TrendRuntime("candidate-worse", -500.0),
        observations=series,
        prior={},
    )

    decision = service.recommend_promotion(
        challenge_id=CHALLENGE_ID,
        run_id=RUN_ID,
        decision_day=105,
        champion=champion,
        candidate=candidate,
        baseline=baseline,
    )

    assert decision.disposition is PromotionDisposition.PROMOTED
    assert decision.reason_code == "baseline_materially_better_than_champion"
    assert decision.candidate_artifact_id == baseline.artifact.id
    assert decision.evaluation_fold_ids == tuple(fold.id for fold in baseline.folds)


def test_runtime_failure_rejects_unsupported_candidate() -> None:
    service = evaluator()
    series = observations()
    champion = service.evaluate(
        run_id=RUN_ID,
        challenge_id=CHALLENGE_ID,
        runtime=TrendRuntime("champion-valid", -50.0),
        observations=series,
        prior={},
    )
    baseline = service.evaluate(
        run_id=RUN_ID,
        challenge_id=CHALLENGE_ID,
        runtime=TrendRuntime("baseline-valid", -25.0),
        observations=series,
        prior={},
    )
    candidate = service.evaluate(
        run_id=RUN_ID,
        challenge_id=CHALLENGE_ID,
        runtime=TrendRuntime("candidate-unsupported", -100.0, fail_predict=True),
        observations=series,
        prior={},
    )

    decision = service.recommend_promotion(
        challenge_id=RUN_ID,
        run_id=RUN_ID,
        decision_day=21,
        champion=champion,
        candidate=candidate,
        baseline=baseline,
    )

    assert not candidate.passed
    assert candidate.failure_codes == ("fold_0:RuntimeError",)
    assert decision.disposition == PromotionDisposition.REJECTED
    assert decision.candidate_fitted_model_id is None


def test_declared_artifact_tests_must_have_a_trusted_test_runner() -> None:
    result = evaluator().evaluate(
        run_id=RUN_ID,
        challenge_id=CHALLENGE_ID,
        runtime=TrendRuntime("untested-candidate", -100.0, declares_tests=True),
        observations=observations(),
        prior={},
    )

    assert not result.passed
    assert result.failure_codes == ("artifact_test_runner_unavailable",)
    assert not result.folds
