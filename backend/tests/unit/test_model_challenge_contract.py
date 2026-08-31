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
    ChallengeResidual,
    HypothesisBacktestFold,
    HypothesisBacktestResult,
    HypothesisEvidenceKind,
    HypothesisEvidenceReference,
    HypothesisFamily,
    ModelBuilderProposal,
    ModelChallengeDecision,
    ModelChallengePackage,
    ModelChallengeRecord,
    ModelChallengeResolution,
    ModelChallengeStatus,
    ParameterAdjustmentProposal,
    ParameterDirection,
    ParameterStepSize,
    RelationshipActivationProposal,
    WorldModelHypothesisDiff,
)
from lithops.domain.models import ObservationSnapshot
from lithops.domain.world_model import WorldModelParameterName
from lithops.world_model.bootstrap import bootstrap_world_model
from pydantic import ValidationError

STARTED_AT = datetime(2026, 8, 25, 9, 0, tzinfo=UTC)


def identifier(value: int) -> UUID:
    return UUID(int=value)


def challenge_package() -> ModelChallengePackage:
    run_id = identifier(1)
    active_model = bootstrap_world_model(
        run_id,
        ObservationSnapshot(
            day=0,
            cash=1_000_000,
            metrics={"marketing_spend": 10_000, "acquisition": 80},
            observed_at=STARTED_AT,
        ),
    )
    outcome_ids = (identifier(101), identifier(102), identifier(103))
    health = ModelHealthSignal(
        id=identifier(10),
        run_id=run_id,
        model_version_id=active_model.id,
        evaluated_day=21,
        status=ModelHealthStatus.DEGRADED,
        outcome_ids=outcome_ids,
        horizon_performance=(
            HorizonPerformance(
                horizon_days=7,
                outcome_count=3,
                mean_normalized_absolute_error=0.3,
                interval_coverage=0.0,
                mean_weighted_interval_score=120_000,
                signed_bias=-0.3,
            ),
        ),
        interval_miss_count=3,
        directional_bias=-0.3,
        rebuild_recommended=True,
        trigger_codes=(
            "two_of_last_three_interval_misses",
            "persistent_directional_bias",
        ),
        evaluated_at=STARTED_AT,
    )
    observations = (
        ChallengeObservation(
            reference="observation:run-1:0",
            day=0,
            cash=1_000_000,
            metrics=(ChallengeMetric(name="marketing_spend", value=10_000),),
            observed_at=STARTED_AT,
        ),
        ChallengeObservation(
            reference="observation:run-1:21",
            day=21,
            cash=700_000,
            metrics=(ChallengeMetric(name="weekly_acquisition", value=25),),
            observed_at=STARTED_AT,
        ),
    )
    residuals = tuple(
        ChallengeResidual(
            outcome_id=outcome_id,
            prediction_id=identifier(200 + index),
            target_id=identifier(300 + index),
            issued_day=7 * (index - 1),
            horizon_days=7,
            target_day=7 * index,
            observed_day=7 * index,
            predicted_cash=1_000_000 - 50_000 * index,
            lower_cash=900_000 - 50_000 * index,
            upper_cash=1_100_000 - 50_000 * index,
            actual_cash=850_000 - 50_000 * index,
            signed_error=-150_000,
            normalized_absolute_error=0.2,
            interval_hit=False,
        )
        for index, outcome_id in enumerate(outcome_ids, start=1)
    )
    return ModelChallengePackage(
        challenge_id=identifier(20),
        run_id=run_id,
        health_signal=health,
        active_model=active_model,
        observations=observations,
        residuals=residuals,
        created_at=STARTED_AT,
    )


def hypothesis_diff() -> WorldModelHypothesisDiff:
    return WorldModelHypothesisDiff(
        parameter_adjustments=(
            ParameterAdjustmentProposal(
                parameter_name=WorldModelParameterName.MARKETING_SATURATION,
                direction=ParameterDirection.DECREASE,
                step_size=ParameterStepSize.MEDIUM,
            ),
        ),
        relationship_activations=(
            RelationshipActivationProposal(
                relationship_key=AllowedRelationshipKey.MARKETING_SPEND_TO_ACQUISITION,
            ),
        ),
    )


def proposal() -> ModelBuilderProposal:
    return ModelBuilderProposal(
        id=identifier(30),
        challenge_id=identifier(20),
        builder_name="acquisition_builder",
        builder_version="1.0",
        prompt_version="acquisition-builder-v1",
        provider="deterministic-test",
        model_name="static-builder",
        family=HypothesisFamily.ACQUISITION_EFFICIENCY,
        summary="Paid acquisition is saturating earlier than the active model expects.",
        rationale="Three directional cash misses coincide with falling weekly acquisition.",
        diff=hypothesis_diff(),
        evidence=(
            HypothesisEvidenceReference(
                kind=HypothesisEvidenceKind.OBSERVATION,
                reference="observation:run-1:21",
                observed_day=21,
            ),
            HypothesisEvidenceReference(
                kind=HypothesisEvidenceKind.PREDICTION_OUTCOME,
                reference="prediction-outcome:103",
                observed_day=21,
            ),
        ),
        created_at=STARTED_AT,
    )


def backtest() -> HypothesisBacktestResult:
    return HypothesisBacktestResult(
        id=identifier(40),
        challenge_id=identifier(20),
        proposal_id=identifier(30),
        scorer_version="rolling-one-step-v1",
        folds=(
            HypothesisBacktestFold(
                outcome_id=identifier(101),
                observed_day=7,
                baseline_normalized_error=0.30,
                candidate_normalized_error=0.20,
                baseline_interval_score=100_000,
                candidate_interval_score=80_000,
            ),
            HypothesisBacktestFold(
                outcome_id=identifier(102),
                observed_day=14,
                baseline_normalized_error=0.28,
                candidate_normalized_error=0.18,
                baseline_interval_score=95_000,
                candidate_interval_score=75_000,
            ),
        ),
        baseline_score=0.29,
        candidate_score=0.19,
        raw_improvement=0.10,
        complexity_penalty=0.02,
        penalized_improvement=0.08,
        minimum_required_improvement=0.05,
        supported=True,
        evaluated_at=STARTED_AT,
    )


def test_challenge_package_is_deeply_immutable_and_complete() -> None:
    package = challenge_package()

    assert package.health_signal.rebuild_recommended is True
    assert len(package.residuals) == 3
    with pytest.raises(ValidationError, match="frozen"):
        package.observations[0].cash = 0  # type: ignore[misc]
    with pytest.raises(ValidationError, match="frozen"):
        package.active_model.parameters[0].confidence = 1  # type: ignore[misc]


def test_challenge_package_requires_one_triggered_model_and_complete_history() -> None:
    package = challenge_package()
    watching = package.health_signal.model_copy(
        update={
            "status": ModelHealthStatus.WATCHING,
            "rebuild_recommended": False,
            "trigger_codes": (),
        }
    )
    with pytest.raises(ValidationError, match="persistent degraded-model trigger"):
        ModelChallengePackage.model_validate(
            package.model_copy(update={"health_signal": watching}).model_dump()
        )

    with pytest.raises(ValidationError, match="include every health-signal outcome"):
        ModelChallengePackage.model_validate(
            package.model_copy(update={"residuals": package.residuals[:-1]}).model_dump()
        )

    with pytest.raises(ValidationError, match="belong to one run"):
        ModelChallengePackage.model_validate(
            package.model_copy(update={"run_id": identifier(999)}).model_dump()
        )


def test_residual_contract_rejects_rewritten_reality() -> None:
    residual = challenge_package().residuals[0]

    with pytest.raises(ValidationError, match="signed error is inconsistent"):
        ChallengeResidual.model_validate(
            residual.model_copy(update={"signed_error": 1}).model_dump()
        )
    with pytest.raises(ValidationError, match="interval result is inconsistent"):
        ChallengeResidual.model_validate(
            residual.model_copy(update={"interval_hit": True}).model_dump()
        )


def test_builder_proposal_accepts_only_bounded_world_model_changes() -> None:
    candidate = proposal()

    assert candidate.diff.parameter_adjustments[0].parameter_name is (
        WorldModelParameterName.MARKETING_SATURATION
    )
    assert "confidence" not in candidate.diff.parameter_adjustments[0].model_dump()

    payload = candidate.model_dump()
    payload["commands"] = [{"tool": "set_prices"}]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ModelBuilderProposal.model_validate(payload)

    invalid_diff = candidate.diff.model_dump()
    invalid_diff["parameter_adjustments"][0]["parameter_name"] = "invented_parameter"
    with pytest.raises(ValidationError, match="Input should be"):
        WorldModelHypothesisDiff.model_validate(invalid_diff)

    invalid_relationship = candidate.diff.model_dump()
    invalid_relationship["relationship_activations"][0]["relationship_key"] = (
        "social_sentiment_to_revenue"
    )
    with pytest.raises(ValidationError, match="Input should be"):
        WorldModelHypothesisDiff.model_validate(invalid_relationship)


def test_hypothesis_diff_and_evidence_must_be_nonempty_and_unique() -> None:
    with pytest.raises(ValidationError, match="at least one allowed change"):
        WorldModelHypothesisDiff()

    adjustment = hypothesis_diff().parameter_adjustments[0]
    with pytest.raises(ValidationError, match="adjust one parameter twice"):
        WorldModelHypothesisDiff(parameter_adjustments=(adjustment, adjustment))

    candidate = proposal()
    with pytest.raises(ValidationError, match="evidence references must be unique"):
        ModelBuilderProposal.model_validate(
            candidate.model_copy(
                update={"evidence": (candidate.evidence[0], candidate.evidence[0])}
            ).model_dump()
        )


def test_backtest_result_requires_a_coherent_deterministic_score() -> None:
    result = backtest()

    assert result.supported is True
    with pytest.raises(ValidationError, match="raw improvement is inconsistent"):
        HypothesisBacktestResult.model_validate(
            result.model_copy(update={"raw_improvement": 0.9}).model_dump()
        )
    with pytest.raises(ValidationError, match="support decision is inconsistent"):
        HypothesisBacktestResult.model_validate(
            result.model_copy(update={"supported": False}).model_dump()
        )
    with pytest.raises(ValidationError, match="folds must be chronological"):
        HypothesisBacktestResult.model_validate(
            result.model_copy(update={"folds": tuple(reversed(result.folds))}).model_dump()
        )


def test_challenge_decision_cannot_claim_an_inconsistent_activation() -> None:
    accepted = ModelChallengeDecision(
        id=identifier(50),
        challenge_id=identifier(20),
        resolution=ModelChallengeResolution.ACCEPTED,
        selected_proposal_ids=(identifier(30),),
        supporting_backtest_ids=(identifier(40),),
        activated_model_version_id=identifier(60),
        authority_name="executive",
        authority_version="static-executive-v1",
        reason_code="best_supported_hypothesis",
        decided_at=STARTED_AT,
    )
    assert accepted.activated_model_version_id == identifier(60)

    with pytest.raises(ValidationError, match="cannot activate proposals"):
        ModelChallengeDecision.model_validate(
            accepted.model_copy(
                update={"resolution": ModelChallengeResolution.NO_SUPPORTED_WINNER}
            ).model_dump()
        )
    with pytest.raises(ValidationError, match="exactly one proposal"):
        ModelChallengeDecision.model_validate(
            accepted.model_copy(
                update={"selected_proposal_ids": (identifier(30), identifier(31))}
            ).model_dump()
        )
    with pytest.raises(ValidationError, match="at least two proposals"):
        ModelChallengeDecision.model_validate(
            accepted.model_copy(
                update={"resolution": ModelChallengeResolution.MERGED}
            ).model_dump()
        )


def test_challenge_lifecycle_requires_unique_builders_and_terminal_evidence() -> None:
    record = ModelChallengeRecord(
        id=identifier(20),
        run_id=identifier(1),
        health_signal_id=identifier(10),
        base_model_version_id=identifier(11),
        requested_builders=("pricing_builder", "acquisition_builder"),
        created_at=STARTED_AT,
        updated_at=STARTED_AT,
    )
    assert record.status is ModelChallengeStatus.TRIGGERED

    specialist = record.model_copy(update={"requested_builders": ("acquisition_builder",)})
    assert specialist.requested_builders == ("acquisition_builder",)

    with pytest.raises(ValidationError, match="builders must be unique"):
        ModelChallengeRecord.model_validate(
            record.model_copy(
                update={"requested_builders": ("pricing_builder", "pricing_builder")}
            ).model_dump()
        )
    with pytest.raises(ValidationError, match="completion timestamp"):
        ModelChallengeRecord.model_validate(
            record.model_copy(update={"status": ModelChallengeStatus.COMPLETED}).model_dump()
        )
    with pytest.raises(ValidationError, match="failure reason"):
        ModelChallengeRecord.model_validate(
            record.model_copy(
                update={"status": ModelChallengeStatus.FAILED, "completed_at": STARTED_AT}
            ).model_dump()
        )
