from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest
from lithops.domain.evaluation import (
    HorizonPerformance,
    ModelHealthSignal,
    ModelHealthStatus,
)
from lithops.domain.model_challenge import (
    AllowedRelationshipKey,
    ChallengeMetric,
    ChallengeObservation,
    ChallengeParameterSensitivity,
    ChallengeResidual,
    HypothesisEvidenceKind,
    HypothesisEvidenceReference,
    HypothesisFamily,
    ModelBuilderProposal,
    ModelChallengePackage,
    ParameterAdjustmentProposal,
    ParameterDirection,
    ParameterStepSize,
    RelationshipActivationProposal,
    WorldModelHypothesisDiff,
)
from lithops.domain.models import ObservationSnapshot
from lithops.domain.world_model import WorldModelParameterName, WorldModelVersion
from lithops.world_model import backtest_hypothesis, bootstrap_world_model, compile_hypothesis

STARTED_AT = datetime(2026, 8, 25, 9, 0, tzinfo=UTC)
RUN_ID = UUID(int=1)
CHALLENGE_ID = UUID(int=2)


def challenge_package() -> ModelChallengePackage:
    model = bootstrap_world_model(
        RUN_ID,
        ObservationSnapshot(
            day=0,
            cash=1_000_000,
            metrics={"marketing_spend": 10_000, "acquisition": 80},
            observed_at=STARTED_AT,
        ),
    )
    outcome_ids = (UUID(int=101), UUID(int=102), UUID(int=103))
    health = ModelHealthSignal(
        id=UUID(int=10),
        run_id=RUN_ID,
        model_version_id=model.id,
        evaluated_day=21,
        status=ModelHealthStatus.DEGRADED,
        outcome_ids=outcome_ids,
        horizon_performance=(
            HorizonPerformance(
                horizon_days=7,
                outcome_count=3,
                mean_normalized_absolute_error=0.6,
                interval_coverage=0.0,
                mean_weighted_interval_score=500_000,
                signed_bias=-0.5,
            ),
        ),
        interval_miss_count=3,
        directional_bias=-0.5,
        rebuild_recommended=True,
        trigger_codes=("persistent_directional_bias",),
        evaluated_at=STARTED_AT,
    )
    actual_cash = (800_000.0, 650_000.0, 500_000.0)
    residuals = tuple(
        ChallengeResidual(
            outcome_id=outcome_id,
            prediction_id=UUID(int=200 + index),
            target_id=UUID(int=300 + index),
            issued_day=7 * (index - 1),
            horizon_days=7,
            target_day=7 * index,
            observed_day=7 * index,
            predicted_cash=1_000_000,
            lower_cash=900_000,
            upper_cash=1_100_000,
            actual_cash=cash,
            signed_error=cash - 1_000_000,
            normalized_absolute_error=abs(cash - 1_000_000) / cash,
            interval_hit=False,
            parameter_sensitivities=(
                ChallengeParameterSensitivity(
                    parameter_name=WorldModelParameterName.MARKETING_SATURATION,
                    cash_sensitivity_per_unit=4_000_000,
                    evidence_reference=f"finite-difference:{outcome_id}:marketing_saturation",
                ),
            ),
        )
        for index, (outcome_id, cash) in enumerate(
            zip(outcome_ids, actual_cash, strict=True),
            start=1,
        )
    )
    return ModelChallengePackage(
        challenge_id=CHALLENGE_ID,
        run_id=RUN_ID,
        health_signal=health,
        active_model=model,
        observations=(
            ChallengeObservation(
                reference=f"observation:{RUN_ID}:21",
                day=21,
                cash=500_000,
                metrics=(ChallengeMetric(name="weekly_acquisition", value=20),),
                observed_at=STARTED_AT,
            ),
        ),
        residuals=residuals,
        created_at=STARTED_AT,
    )


def proposal(
    *,
    direction: ParameterDirection = ParameterDirection.DECREASE,
    proposal_id: int = 20,
    diff: WorldModelHypothesisDiff | None = None,
) -> ModelBuilderProposal:
    package = challenge_package()
    return ModelBuilderProposal(
        id=UUID(int=proposal_id),
        challenge_id=CHALLENGE_ID,
        builder_name="acquisition_builder",
        builder_version="1.0",
        prompt_version="acquisition-builder-v1",
        provider="deterministic-test",
        model_name="static-builder",
        family=HypothesisFamily.ACQUISITION_EFFICIENCY,
        summary="Acquisition saturates earlier than expected.",
        rationale="Falling acquisition accompanies three directional cash misses.",
        diff=diff
        or WorldModelHypothesisDiff(
            parameter_adjustments=(
                ParameterAdjustmentProposal(
                    parameter_name=WorldModelParameterName.MARKETING_SATURATION,
                    direction=direction,
                    step_size=ParameterStepSize.MEDIUM,
                ),
            ),
        ),
        evidence=(
            HypothesisEvidenceReference(
                kind=HypothesisEvidenceKind.OBSERVATION,
                reference=package.observations[0].reference,
                observed_day=21,
            ),
            HypothesisEvidenceReference(
                kind=HypothesisEvidenceKind.PREDICTION_OUTCOME,
                reference=f"prediction-outcome:{package.residuals[-1].outcome_id}",
                observed_day=21,
            ),
        ),
        created_at=STARTED_AT,
    )


def parameter_estimate(model: WorldModelVersion, name: WorldModelParameterName) -> float:
    return next(item.estimate for item in model.parameters if item.name is name)


def test_compiler_maps_a_step_class_to_one_bounded_concrete_model() -> None:
    package = challenge_package()
    compiled = compile_hypothesis(package, proposal())

    old_estimate = parameter_estimate(
        package.active_model,
        WorldModelParameterName.MARKETING_SATURATION,
    )
    new_estimate = parameter_estimate(
        compiled.candidate_model,
        WorldModelParameterName.MARKETING_SATURATION,
    )
    assert old_estimate == 0.65
    assert new_estimate == pytest.approx(0.615)
    assert compiled.changed_parameters == (
        WorldModelParameterName.MARKETING_SATURATION,
    )
    old_confidence = next(
        item.confidence
        for item in package.active_model.parameters
        if item.name is WorldModelParameterName.MARKETING_SATURATION
    )
    new_confidence = next(
        item.confidence
        for item in compiled.candidate_model.parameters
        if item.name is WorldModelParameterName.MARKETING_SATURATION
    )
    assert new_confidence == old_confidence


def test_backtest_is_reproducible_and_rewards_supported_forecast_improvement() -> None:
    package = challenge_package()
    candidate = proposal()

    first_compiled, first = backtest_hypothesis(package, candidate)
    second_compiled, second = backtest_hypothesis(package, candidate)

    assert first_compiled == second_compiled
    assert first == second
    assert first.baseline_score > first.candidate_score
    assert first.raw_improvement > first.complexity_penalty
    assert first.supported is True
    assert all(
        fold.candidate_normalized_error < fold.baseline_normalized_error
        for fold in first.folds
    )


def test_harmful_or_uninformative_hypotheses_do_not_beat_the_baseline() -> None:
    package = challenge_package()
    _, harmful = backtest_hypothesis(
        package,
        proposal(direction=ParameterDirection.INCREASE),
    )
    assert harmful.candidate_score > harmful.baseline_score
    assert harmful.supported is False

    missing_sensitivity = package.model_copy(
        update={
            "residuals": tuple(
                item.model_copy(update={"parameter_sensitivities": ()})
                for item in package.residuals
            )
        }
    )
    with pytest.raises(ValueError, match="lacks sensitivities"):
        backtest_hypothesis(missing_sensitivity, proposal())


def test_prose_and_proposal_identity_do_not_change_deterministic_scores() -> None:
    package = challenge_package()
    first = proposal(proposal_id=20)
    second = proposal(proposal_id=21).model_copy(
        update={"summary": "Different prose.", "rationale": "Still different prose."}
    )

    _, first_result = backtest_hypothesis(package, first)
    _, second_result = backtest_hypothesis(package, second)

    assert first_result.baseline_score == second_result.baseline_score
    assert first_result.candidate_score == second_result.candidate_score
    assert first_result.penalized_improvement == second_result.penalized_improvement


def test_compiler_rejects_external_evidence_and_active_relationship_activation() -> None:
    package = challenge_package()
    external = proposal().model_copy(
        update={
            "evidence": (
                proposal().evidence[0],
                HypothesisEvidenceReference(
                    kind=HypothesisEvidenceKind.PREDICTION_OUTCOME,
                    reference="prediction-outcome:not-in-package",
                    observed_day=21,
                ),
            )
        }
    )
    with pytest.raises(ValueError, match="outside the challenge package"):
        compile_hypothesis(package, external)

    active_relationship = WorldModelHypothesisDiff(
        relationship_activations=(
            RelationshipActivationProposal(
                relationship_key=AllowedRelationshipKey.MARKETING_SPEND_TO_ACQUISITION,
            ),
        )
    )
    with pytest.raises(ValueError, match="cannot activate an active relationship"):
        compile_hypothesis(package, proposal(diff=active_relationship))


def test_compiler_can_activate_only_a_missing_allowlisted_relationship() -> None:
    package = challenge_package()
    reduced_model = package.active_model.model_copy(
        update={
            "relationships": tuple(
                item
                for item in package.active_model.relationships
                if item.key != AllowedRelationshipKey.SEGMENT_TO_CONVERSION.value
            )
        }
    )
    reduced_model = WorldModelVersion.model_validate(reduced_model.model_dump())
    reduced_package = package.model_copy(update={"active_model": reduced_model})
    relationship_diff = WorldModelHypothesisDiff(
        relationship_activations=(
            RelationshipActivationProposal(
                relationship_key=AllowedRelationshipKey.SEGMENT_TO_CONVERSION,
            ),
        )
    )

    compiled, result = backtest_hypothesis(
        reduced_package,
        proposal(diff=relationship_diff),
    )

    assert compiled.activated_relationships == (
        AllowedRelationshipKey.SEGMENT_TO_CONVERSION,
    )
    assert any(
        item.key == AllowedRelationshipKey.SEGMENT_TO_CONVERSION.value
        for item in compiled.candidate_model.relationships
    )
    assert result.raw_improvement == 0
    assert result.supported is False
