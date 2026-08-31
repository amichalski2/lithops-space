from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ValidationError, model_validator

from lithops.domain.evidence import WeeklyEvidencePacket


def utc_now() -> datetime:
    return datetime.now(UTC)


class RunStatus(StrEnum):
    CREATED = "created"
    BOOTSTRAPPING = "bootstrapping"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    COMPLETED = "completed"
    BANKRUPT = "bankrupt"
    FAILED = "failed"


class WorkflowStep(StrEnum):
    READY = "ready"
    CHECKPOINT = "checkpoint"


class DecisionStatus(StrEnum):
    PREPARED = "prepared"
    COMMITTED = "committed"
    FAILED = "failed"


class ReceiptStatus(StrEnum):
    EXECUTED = "executed"
    REPLAYED = "replayed"
    REJECTED = "rejected"


class OperationStatus(StrEnum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


class CashForecast(BaseModel):
    horizon_days: int
    point: float
    lower: float
    upper: float

    @model_validator(mode="after")
    def validate_interval(self) -> CashForecast:
        if self.horizon_days not in {7, 28, 84, 182}:
            raise ValueError("cash forecast horizon must be one of 7, 28, 84, or 182")
        if not self.lower <= self.point <= self.upper:
            raise ValueError("cash forecast must satisfy lower <= point <= upper")
        return self


class CashForecasts(BaseModel):
    items: list[CashForecast] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def validate_horizons(self) -> CashForecasts:
        horizons = {item.horizon_days for item in self.items}
        if horizons != {7, 28, 84, 182}:
            raise ValueError("cash forecasts must contain exactly 7, 28, 84, and 182 days")
        return self

    def ordered(self) -> list[CashForecast]:
        return sorted(self.items, key=lambda item: item.horizon_days)


class ObservationSnapshot(BaseModel):
    day: int = Field(ge=0)
    cash: float
    metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    evidence: WeeklyEvidencePacket | None = None
    observed_at: datetime = Field(default_factory=utc_now)


class ActionCommand(BaseModel):
    tool: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    arguments: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str = Field(min_length=1, max_length=200)

    @property
    def semantic_hash(self) -> str:
        """Identity of what is executed, independent of retry metadata."""

        payload = {"tool": self.tool, "arguments": self.arguments}
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ExperimentMeasurement(BaseModel):
    """One observable used to evaluate a treatment without confusing it with the action."""

    source: str = Field(pattern=r"^(?:acquisition|cohort|quality|ledger|configuration)$")
    metric: str = Field(min_length=1, max_length=80)
    target_segment: str | None = Field(
        default=None, pattern=r"^(?:S[1-3]|E[1-3]|D_[SE]\d{2})$"
    )
    target_channel: str | None = Field(
        default=None,
        pattern=r"^(?:social_media|search_ads|linkedin|content_marketing|referral_program)$",
    )
    minimum_exposure: int = Field(default=0, ge=0)
    attribution_window_weeks: int = Field(default=1, ge=1, le=8)
    decision_grade: bool = True


class ExperimentProgram(BaseModel):
    """A bounded causal intervention whose outcome may mature after one week."""

    commitment_id: str = Field(
        min_length=3,
        max_length=80,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    control: str = Field(
        # `strategy` is a standing commitment rather than a controlled
        # intervention: a direction the company holds for a declared window. It
        # earns the same weekly review and downside cap as an experiment, and
        # differs in what it owes — no control arm, and no reversion by default,
        # because a strategy that ends is simply how the company now operates.
        pattern=(
            r"^(?:price|tier|quota|promotion|ads_strength|marketing|development"
            r"|targeted_development|lead_promotion|strategy)$"
        )
    )
    protocol_version: str = Field(default="experiment-program-v1")
    started_week: int = Field(ge=0)
    minimum_maturity_week: int = Field(ge=1)
    maximum_end_week: int = Field(ge=1)
    baseline_value: float = Field(ge=0.0)
    treatment_value: float = Field(ge=0.0)
    maximum_cumulative_downside: float = Field(gt=0.0)
    expected_observation: str = Field(min_length=1, max_length=1_000)
    falsification_condition: str = Field(min_length=1, max_length=1_000)
    # Pre-registered readout: the observation to read at maturity, the
    # threshold that splits it, and what each side commits the company to.
    # Written before the evidence exists, so the week that reads the result
    # inherits a decision instead of re-deriving an intention from history.
    decision_rule: str = Field(default="", max_length=500)
    target_segment: str | None = Field(
        default=None, pattern=r"^(?:S[1-3]|E[1-3]|D_[SE]\d{2})$"
    )
    target_channel: str | None = Field(
        default=None,
        pattern=r"^(?:social_media|search_ads|linkedin|content_marketing|referral_program)$",
    )
    acquisition_probe_weekly_spend: float = Field(default=0.0, ge=0.0, le=70_000.0)
    baseline_targeted_development: dict[str, float] = Field(default_factory=dict)
    treatment_targeted_development: dict[str, float] = Field(default_factory=dict)
    baseline_targeted_ad_spend: dict[str, dict[str, float]] = Field(default_factory=dict)
    baseline_configuration: dict[str, Any] = Field(default_factory=dict)
    treatment_configuration: dict[str, Any] = Field(default_factory=dict)
    measurement_plan: tuple[ExperimentMeasurement, ...] = ()

    @property
    def is_standing_commitment(self) -> bool:
        """Whether this is a held direction rather than a measured intervention."""

        return self.control == "strategy"

    @model_validator(mode="after")
    def validate_timing(self) -> ExperimentProgram:
        if self.is_standing_commitment:
            # It owes a stop condition and a spending limit: durability without
            # either is immunity, which is what this was built to avoid.
            if not self.falsification_condition.strip():
                raise ValueError("a standing commitment requires a stop condition")
        elif self.protocol_version == "experiment-program-v2":
            if not self.baseline_configuration or not self.treatment_configuration:
                raise ValueError("v2 experiments require baseline and treatment configuration")
            if not self.measurement_plan:
                raise ValueError("v2 experiments require an explicit measurement plan")
        if self.minimum_maturity_week <= self.started_week:
            raise ValueError(
                "experiment maturity must follow its start: "
                f"started_week={self.started_week}, "
                f"minimum_maturity_week={self.minimum_maturity_week}, "
                f"maximum_end_week={self.maximum_end_week}, "
                f"commitment_id={self.commitment_id}"
            )
        if self.maximum_end_week < self.minimum_maturity_week:
            raise ValueError(
                "experiment end cannot precede maturity: "
                f"started_week={self.started_week}, "
                f"minimum_maturity_week={self.minimum_maturity_week}, "
                f"maximum_end_week={self.maximum_end_week}, "
                f"commitment_id={self.commitment_id}"
            )
        if self.control in {"development", "targeted_development"} and (
            self.acquisition_probe_weekly_spend > 0.0
        ):
            if self.target_segment is None or self.target_channel is None:
                raise ValueError(
                    "development programs with an acquisition probe require a segment "
                    "and probe channel"
                )
            if self.maximum_end_week != self.minimum_maturity_week + 1:
                raise ValueError(
                    "development programs require one acquisition-probe week"
                )
        if self.control == "targeted_development" and (
            self.acquisition_probe_weekly_spend <= 0.0
        ):
            raise ValueError(
                "targeted development programs require acquisition probe spend"
            )
        return self


class ActionPlan(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    strategy_family: str = Field(min_length=1, max_length=80)
    rationale: str = Field(min_length=1, max_length=4_000)
    commands: list[ActionCommand] = Field(min_length=1)
    proposal_kind: str | None = Field(default=None, pattern=r"^(?:operating|experiment)$")
    hypothesis_id: str | None = Field(
        default=None, max_length=40, pattern=r"^[a-z][a-z0-9_]*$"
    )
    experiment_control: str | None = Field(
        default=None,
        pattern=(
            r"^(?:price|tier|quota|promotion|ads_strength|marketing|development"
            r"|targeted_development|lead_promotion)$"
        ),
    )
    evidence_regime: str | None = Field(default=None, max_length=120)
    experiment_expires_week: int | None = Field(default=None, ge=1)
    # Authorized envelope for this week's enterprise negotiation. Deals are struck
    # per thread after the plan executes, but never outside what was simulated.
    enterprise_engage: bool = False
    enterprise_target_price_per_seat: float | None = Field(default=None, gt=0.0)
    enterprise_floor_price_per_seat: float | None = Field(default=None, gt=0.0)
    enterprise_max_new_seats: float | None = Field(default=None, ge=0.0)
    experiment_program: ExperimentProgram | None = None

    @model_validator(mode="after")
    def validate_command_keys(self) -> ActionPlan:
        keys = [command.idempotency_key for command in self.commands]
        if len(keys) != len(set(keys)):
            raise ValueError("action command idempotency keys must be unique within a plan")
        if self.proposal_kind == "experiment":
            if not all(
                (
                    self.hypothesis_id,
                    self.experiment_control,
                    self.evidence_regime,
                    self.experiment_expires_week,
                )
            ):
                raise ValueError("experiment plans require complete hypothesis lineage")
            if self.experiment_program is not None:
                if self.experiment_program.control != self.experiment_control:
                    raise ValueError("experiment program control must match plan lineage")
                if self.experiment_program.maximum_end_week != self.experiment_expires_week:
                    raise ValueError("experiment program end must match plan expiry")
        elif self.experiment_control is not None or self.experiment_expires_week is not None:
            raise ValueError("only experiment plans may carry experiment controls or expiry")
        elif self.experiment_program is not None:
            # An operating plan may carry a program only when it is a standing
            # commitment: a direction held for a declared window. Refusing every
            # program here is what left strategy with no way to last longer than
            # a week except by pretending to be an experiment.
            if not self.experiment_program.is_standing_commitment:
                raise ValueError(
                    "only experiment plans may carry an experiment program"
                )
        return self

    @property
    def semantic_hash(self) -> str:
        """Ordered identity of the complete action committed for one period."""

        canonical = json.dumps(
            [command.semantic_hash for command in self.commands],
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ProposalRejection(BaseModel):
    """One proposal refused before it could become a candidate.

    A refusal is a veto, not a substitution: it names why the proposal cannot be
    honoured and carries the reason verbatim, so the author sees its own gap
    instead of a value we chose on its behalf.
    """

    week: int = Field(ge=0)
    candidate_index: int = Field(ge=0)
    name: str = Field(min_length=1, max_length=120)
    hypothesis_id: str | None = Field(default=None, max_length=40)
    stage: str = Field(min_length=1, max_length=40)
    veto_codes: tuple[str, ...] = Field(min_length=1)
    detail: str = Field(default="", max_length=500)


_CONSTRUCTION_VETO_PATTERNS: tuple[tuple[str, str], ...] = (
    ("acquisition probe spend", "acquisition_probe_missing"),
    ("segment and probe channel", "measurement_target_missing"),
    ("acquisition-probe week", "experiment_timing_invalid"),
    ("experiment maturity must follow", "experiment_timing_invalid"),
    ("end cannot precede maturity", "experiment_timing_invalid"),
    ("hypothesis lineage", "experiment_lineage_incomplete"),
    ("experiment program control must match", "experiment_lineage_incomplete"),
    ("experiment program end must match", "experiment_lineage_incomplete"),
    ("measurement plan", "configuration_measurement_missing"),
    ("baseline and treatment configuration", "configuration_measurement_missing"),
)


def construction_veto_codes(error: Exception) -> tuple[str, ...]:
    """Name a construction failure in the vocabulary the evaluation cards use.

    The codes are the same ones a simulated candidate would be vetoed under, so a
    refusal reads identically whether it happened before or after simulation. The
    exact message is kept separately as ``detail``; this only classifies it.
    """

    message = str(error)
    codes = tuple(
        code for needle, code in _CONSTRUCTION_VETO_PATTERNS if needle in message
    )
    if codes:
        return tuple(dict.fromkeys(codes))
    if isinstance(error, ValidationError):
        return ("proposal_constraint_violation",)
    return ("proposal_construction_invalid",)


class ProposalBatch(BaseModel):
    """What the Executive proposed, split into what survived and what was refused.

    A batch with no plans is not a failure: the deterministic pool still carries
    the week, and the refusals are what the author is told about next week.
    """

    plans: tuple[ActionPlan, ...] = ()
    rejections: tuple[ProposalRejection, ...] = ()


class CandidateEvaluationRecord(BaseModel):
    strategy: str = Field(min_length=1, max_length=120)
    action_parameters: dict[str, Any] = Field(default_factory=dict)
    expected_ending_cash: float
    downside_ending_cash: float
    bankruptcy_probability: float = Field(ge=0.0, le=1.0)
    going_concern_failure_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    expected_customer_growth: float
    robustness: str = Field(pattern=r"^(?:high|medium|low)$")
    robust_utility: float
    rollout_count: int = Field(ge=1)
    proposal_rationale: str | None = Field(default=None, max_length=4_000)
    proposal_source: str | None = Field(default=None, max_length=120)
    hypothesis_id: str | None = Field(default=None, max_length=40)
    evidence_regime: str | None = Field(default=None, max_length=120)


class ActionReceipt(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    decision_id: UUID
    idempotency_key: str
    tool: str
    semantic_command_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    status: ReceiptStatus
    external_reference: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class DecisionRecord(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    week: int = Field(ge=0)
    status: DecisionStatus = DecisionStatus.PREPARED
    observation: ObservationSnapshot
    action_plan: ActionPlan
    forecasts: CashForecasts
    world_model_version_id: UUID | None = None
    model_artifact_id: UUID | None = None
    model_artifact_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    fitted_model_id: UUID | None = None
    fitted_state_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    prediction_id: UUID | None = None
    prompt_version: str = "unknown"
    assumptions: list[str] = Field(default_factory=list)
    evidence_references: list[str] = Field(default_factory=list)
    candidate_evaluations: list[CandidateEvaluationRecord] = Field(default_factory=list)
    selection_reason_code: str | None = None
    selection_reason: str | None = None
    actual_outcome: ObservationSnapshot | None = None
    created_at: datetime = Field(default_factory=utc_now)
    committed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_executable_model_lineage(self) -> DecisionRecord:
        lineage = (
            self.model_artifact_id,
            self.model_artifact_hash,
            self.fitted_model_id,
            self.fitted_state_hash,
        )
        if any(value is not None for value in lineage) and not all(
            value is not None for value in lineage
        ):
            raise ValueError("executable-model decision lineage must be complete")
        return self


class RunRecord(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    status: RunStatus = RunStatus.CREATED
    workflow_step: WorkflowStep = WorkflowStep.READY
    benchmark_session_id: str | None = None
    current_day: int = Field(default=0, ge=0)
    horizon_days: int = Field(default=500, ge=7)
    version: int = Field(default=0, ge=0)
    last_decision_id: UUID | None = None
    failure_reason: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class EventRecord(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    type: str = Field(min_length=1, max_length=120)
    payload: dict[str, Any] = Field(default_factory=dict)
    sequence: int | None = Field(default=None, ge=1)
    created_at: datetime = Field(default_factory=utc_now)


class StepResult(BaseModel):
    run: RunRecord
    decision: DecisionRecord
    receipts: list[ActionReceipt]
    replayed: bool = False


class StepOperation(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    request_id: str = Field(min_length=1, max_length=200)
    status: OperationStatus = OperationStatus.STARTED
    result: dict[str, Any] | None = None
    error: str | None = None
    attempts: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class RunLease(BaseModel):
    """Exclusive, expiring ownership of one autonomous run."""

    run_id: UUID
    owner_id: str = Field(min_length=1, max_length=200)
    token: UUID = Field(default_factory=uuid4)
    acquired_at: datetime = Field(default_factory=utc_now)
    heartbeat_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime

    @model_validator(mode="after")
    def validate_expiration(self) -> RunLease:
        if self.expires_at <= self.heartbeat_at:
            raise ValueError("run lease must expire after its heartbeat")
        if self.heartbeat_at < self.acquired_at:
            raise ValueError("run lease heartbeat cannot precede acquisition")
        return self
