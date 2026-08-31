"""Provider-neutral contracts for dynamic world-model challenges."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from math import isclose
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lithops.domain.evaluation import ModelHealthSignal, ModelHealthStatus
from lithops.domain.models import utc_now
from lithops.domain.world_model import WorldModelParameterName, WorldModelVersion


class ModelChallengeStatus(StrEnum):
    TRIGGERED = "triggered"
    BUILDING = "building"
    BACKTESTING = "backtesting"
    AWAITING_EXECUTIVE = "awaiting_executive"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelBuilderCallStatus(StrEnum):
    COMPLETED = "completed"
    TIMED_OUT = "timed_out"
    INVALID_OUTPUT = "invalid_output"
    FAILED = "failed"


class HypothesisFamily(StrEnum):
    PRICING_RESPONSE = "pricing_response"
    ACQUISITION_EFFICIENCY = "acquisition_efficiency"
    RETENTION_QUALITY = "retention_quality"
    CAPACITY_PRESSURE = "capacity_pressure"
    SEGMENT_MIX = "segment_mix"


class ParameterDirection(StrEnum):
    INCREASE = "increase"
    DECREASE = "decrease"


class ParameterStepSize(StrEnum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class AllowedRelationshipKey(StrEnum):
    """The fixed P0 relationship library; builders cannot invent graph edges."""

    PRICE_TO_CONVERSION = "price_to_conversion"
    PRICE_TO_CHURN = "price_to_churn"
    MARKETING_SPEND_TO_ACQUISITION = "marketing_spend_to_acquisition"
    DEVELOPMENT_SPEND_TO_QUALITY = "development_spend_to_quality"
    QUALITY_TO_CHURN = "quality_to_churn"
    SEGMENT_TO_CONVERSION = "segment_to_conversion"


class HypothesisEvidenceKind(StrEnum):
    OBSERVATION = "observation"
    PREDICTION_OUTCOME = "prediction_outcome"
    MODEL_PARAMETER = "model_parameter"
    MODEL_RELATIONSHIP = "model_relationship"


class HypothesisEvidenceReference(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: HypothesisEvidenceKind
    reference: str = Field(min_length=1, max_length=240)
    observed_day: int | None = Field(default=None, ge=0)


class ChallengeMetric(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=120, pattern=r"^[a-z][a-z0-9_]*$")
    value: float | int | str | bool | None


class ChallengeObservation(BaseModel):
    """An immutable normalized observation safe to place in a builder package."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reference: str = Field(min_length=1, max_length=240)
    day: int = Field(ge=0)
    cash: float
    metrics: tuple[ChallengeMetric, ...] = ()
    observed_at: datetime

    @model_validator(mode="after")
    def validate_metric_names(self) -> ChallengeObservation:
        names = [metric.name for metric in self.metrics]
        if len(names) != len(set(names)):
            raise ValueError("challenge observation metric names must be unique")
        return self


class ChallengeParameterSensitivity(BaseModel):
    """A sensitivity committed before reality, safe for deterministic backtesting."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    parameter_name: WorldModelParameterName
    cash_sensitivity_per_unit: float
    evidence_reference: str = Field(min_length=1, max_length=240)


class ChallengeResidual(BaseModel):
    """One immutable prediction-versus-actual fact supplied to every builder."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome_id: UUID
    prediction_id: UUID
    target_id: UUID
    issued_day: int = Field(ge=0)
    horizon_days: int
    target_day: int = Field(ge=0)
    observed_day: int = Field(ge=0)
    predicted_cash: float
    lower_cash: float
    upper_cash: float
    actual_cash: float
    signed_error: float
    normalized_absolute_error: float = Field(ge=0.0)
    interval_hit: bool
    parameter_sensitivities: tuple[ChallengeParameterSensitivity, ...] = ()

    @model_validator(mode="after")
    def validate_prediction_fact(self) -> ChallengeResidual:
        if self.horizon_days not in {7, 28, 84, 182}:
            raise ValueError("challenge residual horizon must be 7, 28, 84, or 182")
        if self.target_day != self.issued_day + self.horizon_days:
            raise ValueError("challenge residual target day must match its horizon")
        if self.observed_day != self.target_day:
            raise ValueError("challenge residual must use the exact target-day observation")
        if not self.lower_cash <= self.predicted_cash <= self.upper_cash:
            raise ValueError("challenge residual forecast interval is invalid")
        expected_error = self.actual_cash - self.predicted_cash
        if not isclose(self.signed_error, expected_error, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("challenge residual signed error is inconsistent")
        expected_hit = self.lower_cash <= self.actual_cash <= self.upper_cash
        if self.interval_hit is not expected_hit:
            raise ValueError("challenge residual interval result is inconsistent")
        parameter_names = [item.parameter_name for item in self.parameter_sensitivities]
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("challenge residual parameter sensitivities must be unique")
        return self


class ModelChallengePackage(BaseModel):
    """The same immutable evidence surface is provided to every candidate builder."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    challenge_id: UUID
    run_id: UUID
    health_signal: ModelHealthSignal
    active_model: WorldModelVersion
    observations: tuple[ChallengeObservation, ...] = Field(min_length=1)
    residuals: tuple[ChallengeResidual, ...] = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    schema_version: str = Field(default="1.0", pattern=r"^\d+\.\d+$")

    @model_validator(mode="after")
    def validate_package_identity_and_history(self) -> ModelChallengePackage:
        if self.health_signal.run_id != self.run_id or self.active_model.run_id != self.run_id:
            raise ValueError("challenge package records must belong to one run")
        if self.health_signal.model_version_id != self.active_model.id:
            raise ValueError("challenge package health must evaluate the active model")
        if (
            self.health_signal.status is not ModelHealthStatus.DEGRADED
            or not self.health_signal.rebuild_recommended
            or not self.health_signal.trigger_codes
        ):
            raise ValueError("challenge package requires a persistent degraded-model trigger")

        observation_keys = [(item.day, item.reference) for item in self.observations]
        if len(observation_keys) != len(set(observation_keys)):
            raise ValueError("challenge package observations must be unique")
        if tuple(sorted(observation_keys)) != tuple(observation_keys):
            raise ValueError("challenge package observations must be chronological")

        outcome_ids = [item.outcome_id for item in self.residuals]
        if len(outcome_ids) != len(set(outcome_ids)):
            raise ValueError("challenge package residual outcomes must be unique")
        if set(outcome_ids) != set(self.health_signal.outcome_ids):
            raise ValueError("challenge package must include every health-signal outcome")
        if tuple(
            sorted(self.residuals, key=lambda item: (item.observed_day, str(item.outcome_id)))
        ) != self.residuals:
            raise ValueError("challenge package residuals must be chronological")
        if self.observations[-1].day < self.health_signal.evaluated_day:
            raise ValueError("challenge package must include the trigger-day observation")
        return self


class ParameterAdjustmentProposal(BaseModel):
    """A bounded nomination; deterministic code chooses the concrete value."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    parameter_name: WorldModelParameterName
    direction: ParameterDirection
    step_size: ParameterStepSize


class RelationshipActivationProposal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    relationship_key: AllowedRelationshipKey


class WorldModelHypothesisDiff(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    parameter_adjustments: tuple[ParameterAdjustmentProposal, ...] = ()
    relationship_activations: tuple[RelationshipActivationProposal, ...] = ()

    @model_validator(mode="after")
    def validate_nonempty_unique_diff(self) -> WorldModelHypothesisDiff:
        if not self.parameter_adjustments and not self.relationship_activations:
            raise ValueError("model hypothesis diff must propose at least one allowed change")
        parameter_names = [item.parameter_name for item in self.parameter_adjustments]
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("model hypothesis cannot adjust one parameter twice")
        relationship_keys = [item.relationship_key for item in self.relationship_activations]
        if len(relationship_keys) != len(set(relationship_keys)):
            raise ValueError("model hypothesis cannot activate one relationship twice")
        return self


class ModelBuilderProposal(BaseModel):
    """Strict builder output with no company-action or persistence capability."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    challenge_id: UUID
    builder_name: str = Field(min_length=1, max_length=120, pattern=r"^[a-z][a-z0-9_-]*$")
    builder_version: str = Field(min_length=1, max_length=80)
    prompt_version: str = Field(min_length=1, max_length=120)
    provider: str = Field(min_length=1, max_length=80)
    model_name: str = Field(min_length=1, max_length=160)
    family: HypothesisFamily
    summary: str = Field(min_length=1, max_length=500)
    rationale: str = Field(min_length=1, max_length=2_000)
    diff: WorldModelHypothesisDiff
    evidence: tuple[HypothesisEvidenceReference, ...] = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_evidence_is_unique(self) -> ModelBuilderProposal:
        evidence_keys = [(item.kind, item.reference) for item in self.evidence]
        if len(evidence_keys) != len(set(evidence_keys)):
            raise ValueError("model-builder evidence references must be unique")
        evidence_kinds = {item.kind for item in self.evidence}
        required = {
            HypothesisEvidenceKind.OBSERVATION,
            HypothesisEvidenceKind.PREDICTION_OUTCOME,
        }
        if not required.issubset(evidence_kinds):
            raise ValueError("model-builder proposal must cite observations and residuals")
        return self


class ModelBuilderCallReceipt(BaseModel):
    """Sanitized terminal receipt for one provider invocation attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    challenge_id: UUID
    builder_name: str = Field(min_length=1, max_length=120, pattern=r"^[a-z][a-z0-9_-]*$")
    builder_version: str = Field(min_length=1, max_length=80)
    prompt_version: str = Field(min_length=1, max_length=120)
    provider: str = Field(min_length=1, max_length=80)
    model_name: str = Field(min_length=1, max_length=160)
    attempt: int = Field(ge=1, le=2)
    status: ModelBuilderCallStatus
    input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    proposal_id: UUID | None = None
    error_code: str | None = Field(
        default=None,
        min_length=1,
        max_length=120,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    completed_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_terminal_result(self) -> ModelBuilderCallReceipt:
        completed = self.status is ModelBuilderCallStatus.COMPLETED
        if completed and (
            self.output_hash is None or self.proposal_id is None or self.error_code is not None
        ):
            raise ValueError("completed builder call must reference its validated proposal")
        if not completed and (
            self.output_hash is not None or self.proposal_id is not None or self.error_code is None
        ):
            raise ValueError("failed builder call must include only a sanitized error code")
        return self


class HypothesisBacktestFold(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome_id: UUID
    observed_day: int = Field(ge=0)
    baseline_normalized_error: float = Field(ge=0.0)
    candidate_normalized_error: float = Field(ge=0.0)
    baseline_interval_score: float = Field(ge=0.0)
    candidate_interval_score: float = Field(ge=0.0)


class HypothesisBacktestResult(BaseModel):
    """Self-consistent deterministic score; lower forecast score is better."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    challenge_id: UUID
    proposal_id: UUID
    scorer_version: str = Field(min_length=1, max_length=120)
    folds: tuple[HypothesisBacktestFold, ...] = Field(min_length=1)
    baseline_score: float = Field(ge=0.0)
    candidate_score: float = Field(ge=0.0)
    raw_improvement: float
    complexity_penalty: float = Field(ge=0.0)
    penalized_improvement: float
    minimum_required_improvement: float = Field(ge=0.0)
    supported: bool
    evaluated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_deterministic_score(self) -> HypothesisBacktestResult:
        fold_keys = [(fold.observed_day, fold.outcome_id) for fold in self.folds]
        if len(fold_keys) != len(set(fold_keys)):
            raise ValueError("hypothesis backtest folds must be unique")
        if tuple(sorted(fold_keys, key=lambda item: (item[0], str(item[1])))) != tuple(
            fold_keys
        ):
            raise ValueError("hypothesis backtest folds must be chronological")
        if not isclose(
            self.raw_improvement,
            self.baseline_score - self.candidate_score,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("hypothesis raw improvement is inconsistent")
        expected_penalized = self.raw_improvement - self.complexity_penalty
        if not isclose(
            self.penalized_improvement,
            expected_penalized,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("hypothesis penalized improvement is inconsistent")
        expected_supported = self.penalized_improvement >= self.minimum_required_improvement
        if self.supported is not expected_supported:
            raise ValueError("hypothesis support decision is inconsistent")
        return self


class ModelChallengeResolution(StrEnum):
    ACCEPTED = "accepted"
    MERGED = "merged"
    EXECUTIVE_REJECTED = "executive_rejected"
    NO_SUPPORTED_WINNER = "no_supported_winner"


class ModelChallengeDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    challenge_id: UUID
    resolution: ModelChallengeResolution
    selected_proposal_ids: tuple[UUID, ...] = ()
    supporting_backtest_ids: tuple[UUID, ...] = Field(min_length=1)
    activated_model_version_id: UUID | None = None
    authority_name: str = Field(min_length=1, max_length=120)
    authority_version: str = Field(min_length=1, max_length=120)
    reason_code: str = Field(min_length=1, max_length=120, pattern=r"^[a-z][a-z0-9_]*$")
    decided_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_resolution(self) -> ModelChallengeDecision:
        if len(self.selected_proposal_ids) != len(set(self.selected_proposal_ids)):
            raise ValueError("challenge decision proposal IDs must be unique")
        if len(self.supporting_backtest_ids) != len(set(self.supporting_backtest_ids)):
            raise ValueError("challenge decision backtest IDs must be unique")

        accepted = self.resolution in {
            ModelChallengeResolution.ACCEPTED,
            ModelChallengeResolution.MERGED,
        }
        if accepted and (
            not self.selected_proposal_ids or self.activated_model_version_id is None
        ):
            raise ValueError("accepted challenge decision must activate selected proposals")
        if not accepted and (
            self.selected_proposal_ids or self.activated_model_version_id is not None
        ):
            raise ValueError("rejected challenge decision cannot activate proposals")
        if (
            self.resolution is ModelChallengeResolution.ACCEPTED
            and len(self.selected_proposal_ids) != 1
        ):
            raise ValueError("accepted challenge decision requires exactly one proposal")
        if (
            self.resolution is ModelChallengeResolution.MERGED
            and len(self.selected_proposal_ids) < 2
        ):
            raise ValueError("merged challenge decision requires at least two proposals")
        return self


class ModelChallengeRecord(BaseModel):
    """Persistable lifecycle state without provider-specific session objects."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    health_signal_id: UUID
    base_model_version_id: UUID
    status: ModelChallengeStatus = ModelChallengeStatus.TRIGGERED
    # The legacy parameter-build fleet uses two or three builders, while an
    # executable challenge may deliberately route a structural signal to one
    # specialist or fan out to the four supported model families.  The durable
    # lifecycle record describes the authors actually requested; it must not
    # impose the legacy fleet topology on the executable loop.
    requested_builders: tuple[str, ...] = Field(min_length=1, max_length=8)
    decision_id: UUID | None = None
    failure_reason: str | None = Field(default=None, min_length=1, max_length=1_000)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_lifecycle(self) -> ModelChallengeRecord:
        if len(self.requested_builders) != len(set(self.requested_builders)):
            raise ValueError("model challenge builders must be unique")
        terminal = self.status in {
            ModelChallengeStatus.COMPLETED,
            ModelChallengeStatus.FAILED,
        }
        if terminal is not (self.completed_at is not None):
            raise ValueError("terminal model challenge must have a completion timestamp")
        if self.status is ModelChallengeStatus.COMPLETED and self.decision_id is None:
            raise ValueError("completed model challenge must reference its decision")
        if self.status is ModelChallengeStatus.FAILED and self.failure_reason is None:
            raise ValueError("failed model challenge must include a failure reason")
        if self.status is not ModelChallengeStatus.FAILED and self.failure_reason is not None:
            raise ValueError("only failed model challenges may include a failure reason")
        if self.updated_at < self.created_at:
            raise ValueError("model challenge update cannot predate creation")
        if self.completed_at is not None and self.completed_at < self.created_at:
            raise ValueError("model challenge completion cannot predate creation")
        return self
