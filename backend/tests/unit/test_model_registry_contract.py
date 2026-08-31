from uuid import uuid4

import pytest
from lithops.domain.model_registry import (
    ActiveModelAssignment,
    ModelPromotionDecision,
    PromotionDisposition,
    SandboxExecutionRecord,
    SandboxExecutionStatus,
    SandboxOperation,
    TemporalEvaluationFold,
)
from pydantic import ValidationError


def test_denied_sandbox_execution_requires_a_policy_reason() -> None:
    with pytest.raises(ValidationError, match="policy denial codes"):
        SandboxExecutionRecord.create(
            run_id=uuid4(),
            idempotency_key="validate-1",
            artifact_id=uuid4(),
            artifact_hash="a" * 64,
            operation=SandboxOperation.VALIDATE,
            status=SandboxExecutionStatus.DENIED,
            input_hash="b" * 64,
            runtime_ms=1,
        )


def test_temporal_fold_rejects_leakage_and_unreconciled_score() -> None:
    values = {
        "run_id": uuid4(),
        "challenge_id": uuid4(),
        "artifact_id": uuid4(),
        "artifact_hash": "a" * 64,
        "fitted_model_id": uuid4(),
        "fold_index": 0,
        "evaluation_seed": 7,
        "training_start_day": 0,
        "training_end_day": 14,
        "holdout_start_day": 14,
        "holdout_end_day": 21,
        "sample_count": 2,
        "predictive_score": 0.1,
        "complexity_penalty": 0.01,
        "runtime_penalty": 0.01,
        "total_score": 0.12,
        "invariant_gate_passed": True,
    }
    with pytest.raises(ValidationError, match="holdout must start after"):
        TemporalEvaluationFold.create(**values)
    values["holdout_start_day"] = 15
    values["total_score"] = 0.5
    with pytest.raises(ValidationError, match="total score does not reconcile"):
        TemporalEvaluationFold.create(**values)
    values["total_score"] = 0.12
    fold = TemporalEvaluationFold.create(**values)
    for field, tampered in (("challenge_id", str(uuid4())), ("evaluation_seed", 8)):
        payload = fold.model_dump(mode="json")
        payload[field] = tampered
        with pytest.raises(ValidationError, match="does not match its lineage"):
            TemporalEvaluationFold.model_validate(payload)


def test_promotion_requires_candidate_and_temporal_evidence() -> None:
    run_id = uuid4()
    champion_artifact_id = uuid4()
    champion_fitted_model_id = uuid4()
    with pytest.raises(ValidationError, match="requires a candidate"):
        ModelPromotionDecision.create(
            challenge_id=uuid4(),
            run_id=run_id,
            decision_day=14,
            champion_artifact_id=champion_artifact_id,
            champion_fitted_model_id=champion_fitted_model_id,
            candidate_artifact_id=None,
            candidate_fitted_model_id=None,
            evaluation_fold_ids=(),
            disposition=PromotionDisposition.REJECTED,
            reason_code="invalid_candidate",
        )
    with pytest.raises(ValidationError, match="temporal evaluation evidence"):
        ModelPromotionDecision.create(
            challenge_id=uuid4(),
            run_id=run_id,
            decision_day=14,
            champion_artifact_id=champion_artifact_id,
            champion_fitted_model_id=champion_fitted_model_id,
            candidate_artifact_id=uuid4(),
            candidate_fitted_model_id=uuid4(),
            evaluation_fold_ids=(),
            disposition=PromotionDisposition.PROMOTED,
            reason_code="candidate_better",
        )


def test_active_assignment_identity_cannot_be_tampered() -> None:
    assignment = ActiveModelAssignment.create(
        run_id=uuid4(),
        sequence=1,
        artifact_id=uuid4(),
        artifact_hash="a" * 64,
        fitted_model_id=uuid4(),
        fitted_state_hash="b" * 64,
        promotion_decision_id=uuid4(),
    )
    payload = assignment.model_dump(mode="json")
    payload["sequence"] = 2
    with pytest.raises(ValidationError, match="does not match its lineage"):
        ActiveModelAssignment.model_validate(payload)
