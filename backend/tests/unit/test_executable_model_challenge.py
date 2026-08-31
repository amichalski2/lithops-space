from __future__ import annotations

import asyncio
from datetime import timedelta
from uuid import UUID

import pytest
from lithops.application.executable_model_challenge import ExecutableModelChallenge
from lithops.domain.errors import ConflictError
from lithops.domain.executable_model import ModelArtifact, ModelRuntimeKind
from lithops.domain.model_challenge import ModelChallengeStatus
from lithops.domain.model_registry import PromotionDisposition
from lithops.infrastructure.persistence.repositories import InMemoryRunRepository
from lithops.model_runtime import TemporalEvaluationPolicy, TemporalModelEvaluator

from backend.tests.unit.test_hypothesis_backtest import challenge_package
from backend.tests.unit.test_temporal_model_evaluation import TrendRuntime, observations

SOURCE = """
def fit(history, prior, seed):
    return {"weekly_cash_delta": -100.0}

def predict(fitted, state, action, horizons_days, n_samples, seed):
    samples = []
    for rollout_index in range(n_samples):
        for horizon_days in horizons_days:
            weeks = horizon_days / 7
            revenue = state["revenue_weekly"]
            unit = ((seed * 9301 + rollout_index * 49297) % 233280) / 233280.0
            spread = (unit - 0.5) * 0.04 * max(abs(state["cash"]), 1.0)
            ending_cash = state["cash"] + fitted["weekly_cash_delta"] * weeks + spread
            samples.append({
                "rollout_index": rollout_index,
                "horizon_days": horizon_days,
                "cash": ending_cash,
                "revenue_weekly": revenue,
                "customers": state["customers"],
                "churn_rate": state["churn_rate"],
                "accounting": {
                    "period_days": horizon_days,
                    "starting_cash": state["cash"],
                    "recognized_revenue": revenue * weeks,
                    "other_inflows": 0,
                    "operating_cost": 0,
                    "operations_spend": 0,
                    "marketing_spend": 0,
                    "development_spend": 0,
                    "other_outflows": (
                        (revenue - fitted["weekly_cash_delta"]) * weeks - spread
                    ),
                    "ending_cash": ending_cash,
                    "currency": "USD",
                },
            })
    return {"samples": samples}

def diagnostics(fitted):
    return fitted
""".strip()


class ExactTrendAuthor:
    calls = 0

    async def author(self, *, package, parent_artifact):
        self.calls += 1
        return ModelArtifact.create(
            name="generated-exact-trend",
            runtime_kind=ModelRuntimeKind.SANDBOXED_PYTHON,
            scope="cash",
            hypothesis="Observed weekly cash trend persists locally.",
            authoring_agent="exact-trend-test",
            provider="deterministic-test",
            model_name="exact-trend",
            prompt_version="test-v1",
            source_code=SOURCE,
            parent_artifact_id=parent_artifact.id,
        )


class FailingAuthor:
    calls = 0

    async def author(self, *, package, parent_artifact):
        del package, parent_artifact
        self.calls += 1
        raise ValueError("invalid structured output")


class UnsupportedAuthor(FailingAuthor):
    def supports(self, package):
        del package
        return False


class SupportedExactTrendAuthor(ExactTrendAuthor):
    def supports(self, package):
        return "persistent_zero_conversion_funnel" in package.health_signal.trigger_codes


class SlowAuthor:
    async def author(self, *, package, parent_artifact):
        del package, parent_artifact
        await asyncio.sleep(60)
        raise AssertionError("author deadline was not enforced")


class CoordinatedAuthor(ExactTrendAuthor):
    def __init__(self, *, ready: list[str], all_ready: asyncio.Event, name: str) -> None:
        self.ready = ready
        self.all_ready = all_ready
        self.name = name

    async def author(self, *, package, parent_artifact):
        self.ready.append(self.name)
        if len(self.ready) == 2:
            self.all_ready.set()
        await asyncio.wait_for(self.all_ready.wait(), timeout=0.25)
        return await super().author(package=package, parent_artifact=parent_artifact)


class ArtifactTrendRuntime(TrendRuntime):
    def __init__(self, artifact: ModelArtifact, weekly_cash_delta: float = -100.0):
        self.weekly_cash_delta = weekly_cash_delta
        self.fail_predict = False
        self.customer_offset = 0.0
        self._artifact = artifact


class FailOnceActivationRepository(InMemoryRunRepository):
    def __init__(self) -> None:
        super().__init__()
        self.fail_activation = True

    async def activate_model(self, assignment, *, expected_previous_sequence):
        if self.fail_activation:
            self.fail_activation = False
            raise RuntimeError("simulated crash after durable promotion")
        return await super().activate_model(
            assignment,
            expected_previous_sequence=expected_previous_sequence,
        )


class FailOncePromotionRepository(InMemoryRunRepository):
    def __init__(self) -> None:
        super().__init__()
        self.fail_promotion = True

    async def append_model_promotion_decision(self, decision):
        if self.fail_promotion:
            self.fail_promotion = False
            raise RuntimeError("simulated crash after durable authoring")
        return await super().append_model_promotion_decision(decision)


class ForeignKeyEnforcingRepository(InMemoryRunRepository):
    """Expose the challenge-package FK that the in-memory store normally omits."""

    async def append_model_challenge_package(self, package):
        if await self.get_model_challenge(package.challenge_id) is None:
            raise ConflictError("challenge package violates challenge_id foreign key")
        return await super().append_model_challenge_package(package)


@pytest.mark.asyncio
async def test_supported_generated_challenger_is_persisted_and_activated() -> None:
    package = challenge_package()
    repository = ForeignKeyEnforcingRepository()
    evaluator = TemporalModelEvaluator(
        TemporalEvaluationPolicy(
            min_training_observations=2,
            n_rollouts=2,
            runtime_budget_ms=10_000,
        )
    )
    author = ExactTrendAuthor()
    service = ExecutableModelChallenge(
        repository=repository,
        authors=(author,),
        evaluator=evaluator,
        baseline=TrendRuntime("misspecified-baseline", -25.0),
        runtime_factory=ArtifactTrendRuntime,
    )

    result = await service.run(
        package=package,
        observations=observations(),
        world_model=package.active_model,
        seed=7,
    )
    active = await repository.get_active_model(package.run_id)

    assert result.promotion.disposition == PromotionDisposition.PROMOTED
    assert result.activation == active
    assert active is not None
    assert active.artifact_id == result.candidate_evaluation.artifact.id
    assert len(result.candidate_evaluation.folds) == 2
    assert result.candidate_evaluation.stress_cases_passed == ("low_cash_runway",)
    assert {fold.evaluation_seed for fold in result.candidate_evaluation.folds} == {7}
    baseline_folds = await repository.list_temporal_evaluation_folds(
        package.run_id,
        service.baseline.artifact.id,
    )
    assert {fold.evaluation_seed for fold in baseline_folds} == {7}
    lifecycle = await repository.get_model_challenge(package.challenge_id)
    assert lifecycle is not None
    assert lifecycle.status is ModelChallengeStatus.COMPLETED
    assert lifecycle.decision_id == result.promotion.id
    assert len(lifecycle.requested_builders) == 1
    assert lifecycle.requested_builders[0].endswith(".ExactTrendAuthor:0")
    assert (
        len(
            await repository.list_temporal_evaluation_folds(
                package.run_id,
                result.candidate_evaluation.artifact.id,
            )
        )
        == 2
    )

    replay = await service.run(
        package=package,
        observations=observations(),
        world_model=package.active_model,
        seed=999,
    )
    assert replay.promotion == result.promotion
    assert replay.activation == result.activation
    assert replay.candidate_evaluation is None
    assert author.calls == 1


@pytest.mark.asyncio
async def test_structural_trigger_only_invokes_matching_specialist_authors() -> None:
    package = challenge_package()
    package = package.model_copy(
        update={
            "health_signal": package.health_signal.model_copy(
                update={
                    "trigger_codes": ("persistent_zero_conversion_funnel",),
                }
            )
        },
        deep=True,
    )
    unsupported = UnsupportedAuthor()
    supported = SupportedExactTrendAuthor()
    service = ExecutableModelChallenge(
        repository=InMemoryRunRepository(),
        authors=(unsupported, supported),
        evaluator=TemporalModelEvaluator(
            TemporalEvaluationPolicy(
                min_training_observations=2,
                n_rollouts=2,
                runtime_budget_ms=10_000,
            )
        ),
        baseline=TrendRuntime("routed-misspecified-baseline", -25.0),
        runtime_factory=ArtifactTrendRuntime,
    )

    result = await service.run(
        package=package,
        observations=observations(),
        world_model=package.active_model,
        seed=17,
    )

    assert unsupported.calls == 0
    assert supported.calls == 1
    assert result.candidate_evaluation is not None
    lifecycle = await service.repository.get_model_challenge(package.challenge_id)
    assert lifecycle is not None
    assert len(lifecycle.requested_builders) == 1
    assert lifecycle.requested_builders[0].endswith(".SupportedExactTrendAuthor:1")


@pytest.mark.asyncio
async def test_retry_after_promotion_resumes_activation_without_reauthoring() -> None:
    package = challenge_package()
    repository = FailOnceActivationRepository()
    author = ExactTrendAuthor()
    service = ExecutableModelChallenge(
        repository=repository,
        authors=(author,),
        evaluator=TemporalModelEvaluator(
            TemporalEvaluationPolicy(
                min_training_observations=2,
                n_rollouts=2,
                runtime_budget_ms=10_000,
            )
        ),
        baseline=TrendRuntime("crash-baseline", -25.0),
        runtime_factory=ArtifactTrendRuntime,
    )

    with pytest.raises(RuntimeError, match="simulated crash"):
        await service.run(
            package=package,
            observations=observations(),
            world_model=package.active_model,
            seed=7,
        )

    promotion = await repository.get_model_promotion_decision_for_challenge(
        package.run_id,
        package.challenge_id,
    )
    assert promotion is not None
    assert await repository.get_active_model(package.run_id) is None

    recovered = await service.run(
        package=package,
        observations=observations(),
        world_model=package.active_model,
        seed=999,
    )

    assert recovered.promotion == promotion
    assert recovered.activation == await repository.get_active_model(package.run_id)
    assert recovered.candidate_evaluation is None
    assert author.calls == 1


@pytest.mark.asyncio
async def test_retry_after_evaluation_reuses_durable_authored_artifact() -> None:
    package = challenge_package()
    repository = FailOncePromotionRepository()
    author = ExactTrendAuthor()
    service = ExecutableModelChallenge(
        repository=repository,
        authors=(author,),
        evaluator=TemporalModelEvaluator(
            TemporalEvaluationPolicy(
                min_training_observations=2,
                n_rollouts=2,
                runtime_budget_ms=10_000,
            )
        ),
        baseline=TrendRuntime("authoring-crash-baseline", -25.0),
        runtime_factory=ArtifactTrendRuntime,
    )

    with pytest.raises(RuntimeError, match="durable authoring"):
        await service.run(
            package=package,
            observations=observations(),
            world_model=package.active_model,
            seed=7,
        )
    receipts = await repository.list_model_artifact_authoring_receipts(
        package.run_id,
        package.challenge_id,
    )
    assert len(receipts) == 1
    assert author.calls == 1

    # A benchmark re-observation can carry a later wall-clock timestamp while
    # describing the same company day.  The retry must use the first sealed package,
    # otherwise the durable authoring receipt no longer matches its input hash.
    last_observation = package.observations[-1]
    drifted_package = package.model_copy(
        update={
            "observations": (
                *package.observations[:-1],
                last_observation.model_copy(
                    update={
                        "observed_at": last_observation.observed_at
                        + timedelta(seconds=5)
                    }
                ),
            )
        },
        deep=True,
    )
    recovered = await service.run(
        package=drifted_package,
        observations=observations(),
        world_model=package.active_model,
        seed=7,
    )

    assert recovered.promotion.disposition == PromotionDisposition.PROMOTED
    assert recovered.activation is not None
    assert author.calls == 1
    assert await repository.get_model_challenge_package(package.challenge_id) == package


@pytest.mark.asyncio
async def test_author_failure_records_no_update_without_failing_company_week() -> None:
    package = challenge_package()
    repository = InMemoryRunRepository()
    author = FailingAuthor()
    service = ExecutableModelChallenge(
        repository=repository,
        authors=(author,),
        evaluator=TemporalModelEvaluator(
            TemporalEvaluationPolicy(
                min_training_observations=2,
                n_rollouts=2,
                runtime_budget_ms=10_000,
            )
        ),
        baseline=TrendRuntime("failure-containment-baseline", -100.0),
        runtime_factory=ArtifactTrendRuntime,
    )

    result = await service.run(
        package=package,
        observations=observations(),
        world_model=package.active_model,
        seed=7,
    )

    assert result.promotion.disposition == PromotionDisposition.NO_UPDATE
    assert result.promotion.reason_code == "all_candidate_authors_failed"
    assert result.promotion.evidence["author_failures"][0]["error_code"] == "ValueError"
    assert result.activation is None
    assert result.candidate_evaluation is None

    replay = await service.run(
        package=package,
        observations=observations(),
        world_model=package.active_model,
        seed=999,
    )
    assert replay.promotion == result.promotion
    assert author.calls == 1


@pytest.mark.asyncio
async def test_failed_fixed_baseline_evaluation_is_contained_with_evidence() -> None:
    package = challenge_package()
    repository = InMemoryRunRepository()
    author = ExactTrendAuthor()
    baseline = TrendRuntime(
        "temporally-failed-operational-baseline",
        -100.0,
        fail_predict=True,
    )
    service = ExecutableModelChallenge(
        repository=repository,
        authors=(author,),
        evaluator=TemporalModelEvaluator(
            TemporalEvaluationPolicy(
                min_training_observations=2,
                n_rollouts=2,
                runtime_budget_ms=10_000,
            )
        ),
        baseline=baseline,
        runtime_factory=ArtifactTrendRuntime,
    )

    result = await service.run(
        package=package,
        observations=observations(),
        world_model=package.active_model,
        seed=7,
    )

    assert result.promotion is None
    assert result.activation is None
    assert result.failure_reason_code == "active_champion_failed_evaluation"
    assert result.failure_codes == (
        "champion:fold_0:RuntimeError",
        "baseline:fold_0:RuntimeError",
    )
    assert result.operational_fallback_artifact_id == baseline.artifact.id
    assert author.calls == 0


@pytest.mark.asyncio
async def test_author_deadline_records_durable_no_update_with_useful_error() -> None:
    package = challenge_package()
    repository = InMemoryRunRepository()
    service = ExecutableModelChallenge(
        repository=repository,
        authors=(SlowAuthor(),),
        evaluator=TemporalModelEvaluator(
            TemporalEvaluationPolicy(
                min_training_observations=2,
                n_rollouts=2,
                runtime_budget_ms=10_000,
            )
        ),
        baseline=TrendRuntime("timeout-baseline", -100.0),
        runtime_factory=ArtifactTrendRuntime,
        author_timeout_seconds=0.01,
    )

    result = await service.run(
        package=package,
        observations=observations(),
        world_model=package.active_model,
        seed=7,
    )

    failures = result.promotion.evidence["author_failures"]
    assert result.promotion.disposition == PromotionDisposition.NO_UPDATE
    assert failures[0]["error_code"] == "TimeoutError"
    assert failures[0]["message"] == "TimeoutError"


@pytest.mark.asyncio
async def test_missing_authors_are_started_concurrently() -> None:
    package = challenge_package()
    repository = InMemoryRunRepository()
    ready: list[str] = []
    all_ready = asyncio.Event()
    authors = (
        CoordinatedAuthor(ready=ready, all_ready=all_ready, name="pricing"),
        CoordinatedAuthor(ready=ready, all_ready=all_ready, name="retention"),
    )
    service = ExecutableModelChallenge(
        repository=repository,
        authors=authors,
        evaluator=TemporalModelEvaluator(
            TemporalEvaluationPolicy(
                min_training_observations=2,
                n_rollouts=2,
                runtime_budget_ms=10_000,
            )
        ),
        baseline=TrendRuntime("parallel-baseline", -25.0),
        runtime_factory=ArtifactTrendRuntime,
        author_timeout_seconds=1.0,
    )

    result = await service.run(
        package=package,
        observations=observations(),
        world_model=package.active_model,
        seed=7,
    )

    assert set(ready) == {"pricing", "retention"}
    assert result.promotion.disposition == PromotionDisposition.PROMOTED


@pytest.mark.asyncio
async def test_second_challenge_in_same_run_reuses_baseline_without_fold_conflict() -> None:
    """A later challenge re-scores the unchanged baseline over the same early window.

    The live run failed on day 49 exactly here: fold identity did not carry the
    challenge, so the second challenge recomputed the first challenge's fold identity
    with different scores and the immutable registry refused the write.
    """

    repository = InMemoryRunRepository()
    service = ExecutableModelChallenge(
        repository=repository,
        authors=(ExactTrendAuthor(),),
        evaluator=TemporalModelEvaluator(
            TemporalEvaluationPolicy(
                min_training_observations=2,
                n_rollouts=2,
                runtime_budget_ms=10_000,
            )
        ),
        baseline=TrendRuntime("misspecified-baseline", -25.0),
        runtime_factory=ArtifactTrendRuntime,
    )
    first_package = challenge_package()
    second_package = first_package.model_copy(
        update={"challenge_id": UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")}
    )

    first = await service.run(
        package=first_package,
        observations=observations(),
        world_model=first_package.active_model,
        seed=7,
    )
    second = await service.run(
        package=second_package,
        observations=observations(),
        world_model=second_package.active_model,
        seed=70_022,
    )

    folds = await repository.list_temporal_evaluation_folds(first_package.run_id)
    by_challenge: dict[UUID, set[UUID]] = {}
    for fold in folds:
        by_challenge.setdefault(fold.challenge_id, set()).add(fold.id)

    assert first.promotion.challenge_id == first_package.challenge_id
    assert second.promotion.challenge_id == second_package.challenge_id
    assert set(by_challenge) == {
        first_package.challenge_id,
        second_package.challenge_id,
    }
    assert not by_challenge[first_package.challenge_id] & by_challenge[
        second_package.challenge_id
    ], "each challenge must own its own fold identities"


@pytest.mark.asyncio
async def test_later_challenge_reactivates_baseline_after_champion_drift() -> None:
    repository = InMemoryRunRepository()
    evaluator = TemporalModelEvaluator(
        TemporalEvaluationPolicy(
            min_training_observations=2,
            n_rollouts=4,
            runtime_budget_ms=10_000,
        )
    )
    first_package = challenge_package()
    first_service = ExecutableModelChallenge(
        repository=repository,
        authors=(ExactTrendAuthor(),),
        evaluator=evaluator,
        baseline=TrendRuntime("initial-bad-baseline", -500.0),
        runtime_factory=ArtifactTrendRuntime,
    )
    promoted = await first_service.run(
        package=first_package,
        observations=observations(),
        world_model=first_package.active_model,
        seed=7,
    )
    assert promoted.activation is not None
    assert promoted.activation.sequence == 1

    later_package = first_package.model_copy(
        update={"challenge_id": UUID("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee")}
    )
    later_observations = tuple(
        item.model_copy(
            update={"state": {**item.state, "cash": 1_000.0 - index * 25.0}}
        )
        for index, item in enumerate(observations())
    )
    recovery_baseline = TrendRuntime("recovery-baseline", -25.0)
    recovery_service = ExecutableModelChallenge(
        repository=repository,
        authors=(FailingAuthor(),),
        evaluator=evaluator,
        baseline=recovery_baseline,
        runtime_factory=ArtifactTrendRuntime,
    )

    recovered = await recovery_service.run(
        package=later_package,
        observations=later_observations,
        world_model=later_package.active_model,
        seed=70_105,
    )
    active = await repository.get_active_model(later_package.run_id)

    assert recovered.promotion.reason_code == "baseline_materially_better_than_champion"
    assert recovered.activation == active
    assert active is not None
    assert active.sequence == 2
    assert active.artifact_id == recovery_baseline.artifact.id
