"""Immutable prediction-ledger records and append-only evaluation outcomes."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lithops.domain.evaluation import CashSensitivityEstimate


class PredictionStatus(StrEnum):
    PENDING = "pending"
    DUE = "due"
    MATURED = "matured"
    INVALIDATED = "invalidated"


class PredictionOutcomeAttribution(StrEnum):
    """Whether a matured residual tests the model or a subsequently changed policy."""

    MODEL_PERFORMANCE = "model_performance"
    POLICY_PATH_DIVERGED = "policy_path_diverged"


class CashPredictionTarget(BaseModel):
    """One committed point and central interval forecast for an exact future day."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    horizon_days: int
    target_day: int = Field(ge=0)
    point: float
    lower: float
    upper: float
    interval_probability: float = Field(default=0.95, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_forecast(self) -> CashPredictionTarget:
        if self.horizon_days not in {7, 28, 84, 182}:
            raise ValueError("prediction horizon must be one of 7, 28, 84, or 182 days")
        if not self.lower <= self.point <= self.upper:
            raise ValueError("prediction interval must satisfy lower <= point <= upper")
        if self.interval_probability != 0.95:
            raise ValueError("P0 cash predictions require a 95% interval")
        return self


class PredictionLedgerEntry(BaseModel):
    """The immutable prediction commitment made before a benchmark advance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    run_id: UUID
    decision_id: UUID
    decision_week: int = Field(ge=0)
    issued_day: int = Field(ge=0)
    model_version_id: UUID
    model_artifact_id: UUID | None = None
    model_artifact_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    fitted_model_id: UUID | None = None
    fitted_state_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    prompt_version: str = Field(min_length=1, max_length=120)
    observation_reference: str = Field(min_length=1, max_length=240)
    assumptions: tuple[str, ...] = Field(min_length=1)
    evidence_references: tuple[str, ...] = Field(min_length=1)
    uncertainty_source: str = Field(min_length=1, max_length=240)
    confidence: float = Field(ge=0.0, le=1.0)
    cash_sensitivities: tuple[CashSensitivityEstimate, ...] = ()
    targets: tuple[CashPredictionTarget, ...] = Field(min_length=4, max_length=4)
    committed_at: datetime

    @model_validator(mode="after")
    def validate_targets(self) -> PredictionLedgerEntry:
        lineage = (
            self.model_artifact_id,
            self.model_artifact_hash,
            self.fitted_model_id,
            self.fitted_state_hash,
        )
        if any(value is not None for value in lineage) and not all(
            value is not None for value in lineage
        ):
            raise ValueError("executable-model prediction lineage must be complete")
        horizons = [target.horizon_days for target in self.targets]
        if set(horizons) != {7, 28, 84, 182} or len(horizons) != len(set(horizons)):
            raise ValueError("ledger entry requires exactly one target for every cash horizon")
        for target in self.targets:
            if target.target_day != self.issued_day + target.horizon_days:
                raise ValueError("prediction target_day must equal issued_day + horizon_days")
        valid_horizons = set(horizons)
        if any(
            sensitivity.horizon_days not in valid_horizons
            for sensitivity in self.cash_sensitivities
        ):
            raise ValueError("cash sensitivity must reference a prediction horizon")
        sensitivity_keys = [
            (sensitivity.parameter_name, sensitivity.horizon_days)
            for sensitivity in self.cash_sensitivities
        ]
        if len(sensitivity_keys) != len(set(sensitivity_keys)):
            raise ValueError("cash sensitivities must be unique by parameter and horizon")
        return self


class PredictionActual(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    target_id: UUID
    observed_day: int = Field(ge=0)
    cash: float
    observation_reference: str = Field(min_length=1, max_length=240)
    observed_at: datetime


class PredictionScore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    target_id: UUID
    signed_error: float
    absolute_error: float = Field(ge=0.0)
    absolute_percentage_error: float | None = Field(default=None, ge=0.0)
    normalized_absolute_error: float = Field(ge=0.0)
    interval_hit: bool
    interval_width: float = Field(ge=0.0)
    interval_score: float = Field(ge=0.0)
    weighted_interval_score: float = Field(ge=0.0)
    scored_at: datetime


class PredictionOutcome(BaseModel):
    """An append-only actual and score for one immutable prediction target."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    run_id: UUID
    ledger_entry_id: UUID
    target_id: UUID
    actual: PredictionActual
    score: PredictionScore
    attribution: PredictionOutcomeAttribution = (
        PredictionOutcomeAttribution.MODEL_PERFORMANCE
    )
    policy_divergence_week: int | None = Field(default=None, ge=0)
    recorded_at: datetime

    @model_validator(mode="after")
    def validate_target_identity(self) -> PredictionOutcome:
        if self.actual.target_id != self.target_id or self.score.target_id != self.target_id:
            raise ValueError("outcome, actual, and score must reference the same target")
        if (
            self.attribution is PredictionOutcomeAttribution.POLICY_PATH_DIVERGED
        ) != (self.policy_divergence_week is not None):
            raise ValueError(
                "policy-path attribution and divergence week must be declared together"
            )
        return self
