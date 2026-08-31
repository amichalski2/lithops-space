"""Immutable lineage records for executable company-model artifacts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from lithops.domain.models import utc_now


def _promotion_identity(
    *,
    challenge_id: UUID,
    run_id: UUID,
    decision_day: int,
    champion_artifact_id: UUID,
    champion_fitted_model_id: UUID,
    candidate_artifact_id: UUID | None,
    candidate_fitted_model_id: UUID | None,
    evaluation_fold_ids: tuple[UUID, ...],
    disposition: PromotionDisposition,
    reason_code: str,
) -> UUID:
    identity = ":".join(
        (
            str(challenge_id),
            str(run_id),
            str(decision_day),
            str(champion_artifact_id),
            str(champion_fitted_model_id),
            str(candidate_artifact_id),
            str(candidate_fitted_model_id),
            ",".join(str(item) for item in evaluation_fold_ids),
            disposition.value,
            reason_code,
        )
    )
    return uuid5(NAMESPACE_URL, f"lithops:model-promotion:{identity}")


class SandboxOperation(StrEnum):
    VALIDATE = "validate"
    TEST = "test"
    FIT = "fit"
    PREDICT = "predict"
    DIAGNOSTICS = "diagnostics"


class SandboxExecutionStatus(StrEnum):
    COMPLETED = "completed"
    DENIED = "denied"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class SandboxExecutionRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    run_id: UUID
    idempotency_key: str = Field(min_length=1, max_length=240)
    artifact_id: UUID
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fitted_model_id: UUID | None = None
    operation: SandboxOperation
    status: SandboxExecutionStatus
    input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    policy_denial_codes: tuple[str, ...] = Field(default=(), max_length=40)
    runtime_ms: int = Field(ge=0)
    error_code: str | None = Field(default=None, max_length=120)
    created_at: datetime = Field(default_factory=utc_now)

    @classmethod
    def create(
        cls,
        *,
        run_id: UUID,
        idempotency_key: str,
        **values: object,
    ) -> SandboxExecutionRecord:
        return cls(
            id=uuid5(
                NAMESPACE_URL,
                f"lithops:sandbox-execution:{run_id}:{idempotency_key}",
            ),
            run_id=run_id,
            idempotency_key=idempotency_key,
            **values,
        )

    @model_validator(mode="after")
    def validate_status_payload(self) -> SandboxExecutionRecord:
        expected_id = uuid5(
            NAMESPACE_URL,
            f"lithops:sandbox-execution:{self.run_id}:{self.idempotency_key}",
        )
        if self.id != expected_id:
            raise ValueError("sandbox execution ID does not match its idempotency key")
        if len(self.policy_denial_codes) != len(set(self.policy_denial_codes)):
            raise ValueError("sandbox policy denial codes must be unique")
        if self.status == SandboxExecutionStatus.DENIED:
            if not self.policy_denial_codes:
                raise ValueError("denied sandbox execution requires policy denial codes")
            if self.output_hash is not None:
                raise ValueError("denied sandbox execution cannot have an output hash")
        elif self.policy_denial_codes:
            raise ValueError("policy denial codes require denied sandbox status")
        if self.status == SandboxExecutionStatus.COMPLETED and self.output_hash is None:
            raise ValueError("completed sandbox execution requires an output hash")
        return self


def _fold_identity(
    *,
    run_id: UUID,
    challenge_id: UUID,
    artifact_id: UUID,
    fitted_model_id: UUID,
    fold_index: int,
    evaluation_seed: int,
) -> str:
    """Identity of one evaluated fold.

    The challenge and the evaluation seed both belong here. Without the challenge, a
    later challenge re-scoring the same unchanged baseline collides with the earlier
    one; without the seed, two candidates evaluated side by side inside one challenge
    collide, because they share an artifact and fitted state but are deliberately
    scored on different draws.
    """

    return (
        "lithops:model-evaluation-fold:"
        f"{run_id}:{challenge_id}:{artifact_id}:{fitted_model_id}"
        f":{fold_index}:{evaluation_seed}"
    )


class TemporalEvaluationFold(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    run_id: UUID
    challenge_id: UUID
    artifact_id: UUID
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fitted_model_id: UUID
    fold_index: int = Field(ge=0)
    evaluation_seed: int
    training_start_day: int = Field(ge=0)
    training_end_day: int = Field(ge=0)
    holdout_start_day: int = Field(ge=0)
    holdout_end_day: int = Field(ge=0)
    sample_count: int = Field(ge=1)
    predictive_score: float = Field(ge=0.0)
    complexity_penalty: float = Field(ge=0.0)
    runtime_penalty: float = Field(ge=0.0)
    total_score: float = Field(ge=0.0)
    invariant_gate_passed: bool
    metrics: dict[str, JsonValue] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @classmethod
    def create(
        cls,
        *,
        run_id: UUID,
        challenge_id: UUID,
        artifact_id: UUID,
        fitted_model_id: UUID,
        fold_index: int,
        evaluation_seed: int,
        **values: object,
    ) -> TemporalEvaluationFold:
        return cls(
            id=uuid5(NAMESPACE_URL, _fold_identity(
                run_id=run_id,
                challenge_id=challenge_id,
                artifact_id=artifact_id,
                fitted_model_id=fitted_model_id,
                fold_index=fold_index,
                evaluation_seed=evaluation_seed,
            )),
            run_id=run_id,
            challenge_id=challenge_id,
            artifact_id=artifact_id,
            fitted_model_id=fitted_model_id,
            fold_index=fold_index,
            evaluation_seed=evaluation_seed,
            **values,
        )

    @model_validator(mode="after")
    def validate_fold(self) -> TemporalEvaluationFold:
        expected_id = uuid5(NAMESPACE_URL, _fold_identity(
            run_id=self.run_id,
            challenge_id=self.challenge_id,
            artifact_id=self.artifact_id,
            fitted_model_id=self.fitted_model_id,
            fold_index=self.fold_index,
            evaluation_seed=self.evaluation_seed,
        ))
        if self.id != expected_id:
            raise ValueError("evaluation fold ID does not match its lineage")
        if self.training_end_day < self.training_start_day:
            raise ValueError("evaluation training window is reversed")
        if self.holdout_end_day < self.holdout_start_day:
            raise ValueError("evaluation holdout window is reversed")
        if self.holdout_start_day <= self.training_end_day:
            raise ValueError("evaluation holdout must start after the training window")
        expected_total = self.predictive_score + self.complexity_penalty + self.runtime_penalty
        if abs(self.total_score - expected_total) > 1e-9:
            raise ValueError("evaluation total score does not reconcile")
        return self


class PromotionDisposition(StrEnum):
    PROMOTED = "promoted"
    REJECTED = "rejected"
    NO_UPDATE = "no_update"


class ModelArtifactAuthoringReceipt(BaseModel):
    """Durable boundary between one coding-agent call and temporal evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    challenge_id: UUID
    run_id: UUID
    author_key: str = Field(min_length=1, max_length=160)
    artifact_id: UUID
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime = Field(default_factory=utc_now)

    @classmethod
    def create(
        cls,
        *,
        challenge_id: UUID,
        run_id: UUID,
        author_key: str,
        **values: object,
    ) -> ModelArtifactAuthoringReceipt:
        return cls(
            id=uuid5(
                NAMESPACE_URL,
                f"lithops:model-artifact-authoring:{run_id}:{challenge_id}:{author_key}",
            ),
            challenge_id=challenge_id,
            run_id=run_id,
            author_key=author_key,
            **values,
        )

    @model_validator(mode="after")
    def validate_identity(self) -> ModelArtifactAuthoringReceipt:
        expected_id = uuid5(
            NAMESPACE_URL,
            f"lithops:model-artifact-authoring:{self.run_id}:{self.challenge_id}:{self.author_key}",
        )
        if self.id != expected_id:
            raise ValueError("artifact-authoring receipt ID does not match its identity")
        return self


class ModelPromotionDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    challenge_id: UUID
    run_id: UUID
    decision_day: int = Field(ge=0)
    champion_artifact_id: UUID
    champion_fitted_model_id: UUID
    candidate_artifact_id: UUID | None = None
    candidate_fitted_model_id: UUID | None = None
    evaluation_fold_ids: tuple[UUID, ...] = ()
    disposition: PromotionDisposition
    reason_code: str = Field(min_length=1, max_length=120)
    evidence: dict[str, JsonValue] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @classmethod
    def create(
        cls,
        *,
        challenge_id: UUID,
        run_id: UUID,
        decision_day: int,
        champion_artifact_id: UUID,
        champion_fitted_model_id: UUID,
        candidate_artifact_id: UUID | None,
        candidate_fitted_model_id: UUID | None,
        evaluation_fold_ids: tuple[UUID, ...],
        disposition: PromotionDisposition,
        reason_code: str,
        **values: object,
    ) -> ModelPromotionDecision:
        return cls(
            id=_promotion_identity(
                challenge_id=challenge_id,
                run_id=run_id,
                decision_day=decision_day,
                champion_artifact_id=champion_artifact_id,
                champion_fitted_model_id=champion_fitted_model_id,
                candidate_artifact_id=candidate_artifact_id,
                candidate_fitted_model_id=candidate_fitted_model_id,
                evaluation_fold_ids=evaluation_fold_ids,
                disposition=disposition,
                reason_code=reason_code,
            ),
            challenge_id=challenge_id,
            run_id=run_id,
            decision_day=decision_day,
            champion_artifact_id=champion_artifact_id,
            champion_fitted_model_id=champion_fitted_model_id,
            candidate_artifact_id=candidate_artifact_id,
            candidate_fitted_model_id=candidate_fitted_model_id,
            evaluation_fold_ids=evaluation_fold_ids,
            disposition=disposition,
            reason_code=reason_code,
            **values,
        )

    @model_validator(mode="after")
    def validate_decision(self) -> ModelPromotionDecision:
        has_artifact = self.candidate_artifact_id is not None
        has_fitted = self.candidate_fitted_model_id is not None
        if has_fitted and not has_artifact:
            raise ValueError("candidate fitted model requires an artifact")
        if self.disposition == PromotionDisposition.REJECTED and not has_artifact:
            raise ValueError("rejection requires a candidate artifact")
        if self.disposition == PromotionDisposition.NO_UPDATE and has_artifact != has_fitted:
            raise ValueError("evaluated no-update candidate requires its fitted model")
        if self.disposition == PromotionDisposition.PROMOTED:
            if not has_artifact or not has_fitted:
                raise ValueError("promotion requires a fitted candidate")
            if not self.evaluation_fold_ids:
                raise ValueError("promotion requires temporal evaluation evidence")
        if len(self.evaluation_fold_ids) != len(set(self.evaluation_fold_ids)):
            raise ValueError("promotion evaluation fold IDs must be unique")
        expected_id = _promotion_identity(
            challenge_id=self.challenge_id,
            run_id=self.run_id,
            decision_day=self.decision_day,
            champion_artifact_id=self.champion_artifact_id,
            champion_fitted_model_id=self.champion_fitted_model_id,
            candidate_artifact_id=self.candidate_artifact_id,
            candidate_fitted_model_id=self.candidate_fitted_model_id,
            evaluation_fold_ids=self.evaluation_fold_ids,
            disposition=self.disposition,
            reason_code=self.reason_code,
        )
        if self.id != expected_id:
            raise ValueError("promotion decision ID does not match its evidence identity")
        return self


class ActiveModelAssignment(BaseModel):
    """Append-only active pointer; latest sequence is the active model for a run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    run_id: UUID
    sequence: int = Field(ge=1)
    artifact_id: UUID
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fitted_model_id: UUID
    fitted_state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    promotion_decision_id: UUID
    activated_at: datetime = Field(default_factory=utc_now)

    @classmethod
    def create(
        cls,
        *,
        run_id: UUID,
        sequence: int,
        promotion_decision_id: UUID,
        **values: object,
    ) -> ActiveModelAssignment:
        return cls(
            id=uuid5(
                NAMESPACE_URL,
                f"lithops:active-model:{run_id}:{sequence}:{promotion_decision_id}",
            ),
            run_id=run_id,
            sequence=sequence,
            promotion_decision_id=promotion_decision_id,
            **values,
        )

    @model_validator(mode="after")
    def validate_identity(self) -> ActiveModelAssignment:
        expected_id = uuid5(
            NAMESPACE_URL,
            f"lithops:active-model:{self.run_id}:{self.sequence}:{self.promotion_decision_id}",
        )
        if self.id != expected_id:
            raise ValueError("active-model assignment ID does not match its lineage")
        return self
