"""Provider-neutral model-health and deterministic recalibration contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from lithops.domain.world_model import WorldModelParameterName


class ModelHealthStatus(StrEnum):
    HEALTHY = "healthy"
    WATCHING = "watching"
    DEGRADED = "degraded"


class HorizonPerformance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    horizon_days: int
    outcome_count: int = Field(ge=1)
    mean_normalized_absolute_error: float = Field(ge=0.0)
    interval_coverage: float = Field(ge=0.0, le=1.0)
    mean_weighted_interval_score: float = Field(ge=0.0)
    signed_bias: float


class ModelHealthSignal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    run_id: UUID
    model_version_id: UUID
    evaluated_day: int = Field(ge=0)
    status: ModelHealthStatus
    outcome_ids: tuple[UUID, ...] = Field(min_length=1)
    horizon_performance: tuple[HorizonPerformance, ...] = Field(min_length=1)
    interval_miss_count: int = Field(ge=0)
    directional_bias: float
    rebuild_recommended: bool
    trigger_codes: tuple[str, ...]
    evaluated_at: datetime


class ParameterResidualAttribution(BaseModel):
    """A simulator-derived local sensitivity, never an unconstrained LLM update."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome_id: UUID
    parameter_name: WorldModelParameterName
    cash_sensitivity_per_unit: float
    weight: float = Field(default=1.0, gt=0.0, le=1.0)
    evidence_reference: str = Field(min_length=1, max_length=240)


class CashSensitivityEstimate(BaseModel):
    """Finite-difference cash sensitivity stored when a prediction is committed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    parameter_name: WorldModelParameterName
    horizon_days: int
    cash_sensitivity_per_unit: float
    evidence_reference: str = Field(min_length=1, max_length=240)

    @property
    def is_informative(self) -> bool:
        return self.cash_sensitivity_per_unit != 0.0
