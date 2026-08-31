"""Typed, point-in-time evidence used by weekly strategic decisions."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from lithops.domain.experiment_contracts import OBSERVATION_CONTRACT_VERSION


class AcquisitionEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    segment: str
    channel: str
    leads: int = Field(ge=0)
    spend: float = Field(ge=0.0)


class CohortEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    segment: str
    channel: str
    leads: int = Field(ge=0)
    conversions: int = Field(ge=0)
    losses: int = Field(ge=0)
    pending: int = Field(ge=0)


class QualityEvidence(BaseModel):
    """A transparent proxy; never presented as simulator ground truth."""

    model_config = ConfigDict(frozen=True)

    segment: str
    plan: str = Field(pattern=r"^[ABC]$")
    base_quality_proxy: float = Field(ge=0.0)
    model_tier: int = Field(ge=1, le=5)
    tier_multiplier: float = Field(gt=0.0)
    delivered_quality_proxy: float = Field(ge=0.0)
    targeted_development_daily: float = Field(ge=0.0)
    decision_grade: bool = False
    provenance: str = "derived:ledger-plus-config:segment-quality-proxy-v2"


class LedgerEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    category: str
    weekly_amount: float
    cumulative_amount: float


class ConfigurationEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    prices: dict[str, float]
    model_tiers: dict[str, int]
    usage_quotas: dict[str, float] = Field(default_factory=dict)
    daily_channel_spend: dict[str, float]
    daily_operations_spend: float = Field(ge=0.0)
    daily_development_spend: float = Field(ge=0.0)
    capacity_tier: int = Field(ge=0)
    lead_promotion_json: str = "{}"
    recurring_promotion_json: str = "{}"
    ads_strength_json: str = "{}"
    targeted_ads_json: str = "{}"
    targeted_development_json: str = "{}"
    targeted_ops_json: str = "{}"


class WeeklyEvidencePacket(BaseModel):
    """Deterministic evidence collected at a committed simulation day."""

    model_config = ConfigDict(frozen=True)

    contract_version: str = OBSERVATION_CONTRACT_VERSION
    day: int = Field(ge=0)
    window_start_day_exclusive: int = Field(ge=-7)
    window_end_day_inclusive: int = Field(ge=0)
    acquisition: tuple[AcquisitionEvidence, ...] = ()
    cohorts: tuple[CohortEvidence, ...] = ()
    quality: tuple[QualityEvidence, ...] = ()
    ledger: tuple[LedgerEvidence, ...] = ()
    configuration: ConfigurationEvidence
