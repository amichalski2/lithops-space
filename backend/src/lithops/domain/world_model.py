"""Provider-neutral contracts for Lithops' versioned company world model."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lithops.domain.models import utc_now


class WorldModelParameterName(StrEnum):
    """The deliberately small set of learned parameters supported in P0."""

    PRICE_ELASTICITY = "price_elasticity"
    MARKETING_SATURATION = "marketing_saturation"
    CHURN_SENSITIVITY = "churn_sensitivity"
    QUALITY_LAG_WEEKS = "quality_lag_weeks"
    SEGMENT_RESPONSE = "segment_response"
    # Service allowances ration delivered value: a plan that serves less than a
    # customer demands is worth less to that customer. The demand reference is a
    # scale in usage units per customer per day; observed usage is censored at the
    # configured allowance, so the reference stays uncertain until an allowance
    # above demand reveals it.
    QUOTA_SATURATION = "quota_saturation"
    QUOTA_DEMAND_REFERENCE = "quota_demand_reference"
    # Service capacity is bought in discrete tiers; the step ratio between two
    # neighbouring tiers is learned from observed capacity and its billed cost.
    CAPACITY_TIER_STEP = "capacity_tier_step"
    # In-product advertising trades revenue per customer against perceived quality.
    ADS_REVENUE_RATE = "ads_revenue_rate"
    ADS_QUALITY_TRADEOFF = "ads_quality_tradeoff"
    # Support and reliability work directed at retention.
    OPS_RELIABILITY_RESPONSE = "ops_reliability_response"
    # Owned-channel publishing as a bounded multiplier on lead arrival.
    SOCIAL_LEAD_RESPONSE = "social_lead_response"
    # Negotiated seat demand: how sharply acceptance falls as the offered price
    # rises above the reference. What such buyers require of delivered quality is
    # not a learned parameter but a purchased measurement, carried on the
    # simulation state (`measured_quality_floor_enterprise`).
    ENTERPRISE_PRICE_SENSITIVITY = "enterprise_price_sensitivity"
    # Participation past a purchased floor: the share of arriving leads that
    # convert once delivered quality clears the most accessible group's measured
    # floor, and how wide the transition around that floor is. Both are learned;
    # the floor itself is always bought, never assumed.
    PARTICIPATION_CONVERSION_RATE = "participation_conversion_rate"
    PARTICIPATION_SOFTNESS = "participation_softness"
    # How demanding the dearer half of the catalog is: the delivered quality at
    # which a company can hold a meaningful share of its customers on premium
    # plans rather than the entry one. Fitted from the mix the run actually
    # realises; a fixed blended price hid this relationship entirely.
    PREMIUM_QUALITY_SCALE = "premium_quality_scale"
    # Development spend buys quality with diminishing returns: the response is the
    # size of the effect and the scale is the spend at which returns start to
    # flatten. Both are fitted; a linear form with a ceiling once made every
    # spend above that ceiling look like pure burn, which is the shape of a
    # decision the model can never take rather than a fact about the world.
    DEVELOPMENT_QUALITY_RESPONSE = "development_quality_response"
    # How that response scales with the size of the bet: 1.0 is constant
    # returns, below 1.0 is diminishing. Asked of the data rather than assumed,
    # because our own history cannot separate the two shapes.
    DEVELOPMENT_QUALITY_EXPONENT = "development_quality_exponent"
    # Research programmes: how long a tier takes to land and how much quality it
    # returns per tier. Both are fitted from the run's own completed projects.
    RESEARCH_LAG_WEEKS_PER_TIER = "research_lag_weeks_per_tier"
    RESEARCH_QUALITY_PER_TIER = "research_quality_per_tier"


class EvidenceKind(StrEnum):
    GENERIC_PRIOR = "generic_prior"
    OBSERVATION = "observation"
    PREDICTION_RESIDUAL = "prediction_residual"
    MODEL_BUILDER = "model_builder"


class RelationshipShape(StrEnum):
    LINEAR = "linear"
    SATURATING = "saturating"
    LAGGED = "lagged"
    SEGMENTED = "segmented"


class EvidenceReference(BaseModel):
    """An auditable pointer to evidence, not an unstructured model assertion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: EvidenceKind
    reference: str = Field(min_length=1, max_length=240)
    observed_day: int | None = Field(default=None, ge=0)
    note: str | None = Field(default=None, max_length=1_000)


class WorldModelParameter(BaseModel):
    """One learned scalar with explicit epistemic uncertainty."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: WorldModelParameterName
    estimate: float
    lower_bound: float
    upper_bound: float
    confidence: float = Field(ge=0.0, le=1.0)
    unit: str = Field(min_length=1, max_length=40)
    lag_weeks: int = Field(default=0, ge=0, le=52)
    evidence: tuple[EvidenceReference, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_uncertainty_interval(self) -> WorldModelParameter:
        if not self.lower_bound <= self.estimate <= self.upper_bound:
            raise ValueError(
                "world-model parameter must satisfy lower_bound <= estimate <= upper_bound"
            )
        return self


class WorldModelRelationship(BaseModel):
    """A fixed causal edge whose strength is controlled by learned parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(min_length=1, max_length=120, pattern=r"^[a-z][a-z0-9_]*$")
    cause: str = Field(min_length=1, max_length=120)
    effect: str = Field(min_length=1, max_length=120)
    shape: RelationshipShape
    parameter_names: tuple[WorldModelParameterName, ...] = Field(min_length=1)
    lag_weeks: int = Field(default=0, ge=0, le=52)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: tuple[EvidenceReference, ...] = Field(min_length=1)


class WorldModelParameterChange(BaseModel):
    """Auditable structured diff applied by a deterministic estimator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    parameter_name: WorldModelParameterName
    previous_estimate: float
    new_estimate: float
    previous_confidence: float = Field(ge=0.0, le=1.0)
    new_confidence: float = Field(ge=0.0, le=1.0)
    update_method: str = Field(min_length=1, max_length=120)
    evidence: tuple[EvidenceReference, ...] = Field(min_length=1)


class WorldModelVersion(BaseModel):
    """An immutable snapshot activated for one run at a known observation day."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    version: int = Field(ge=1)
    source_observation_day: int = Field(ge=0)
    based_on_version_id: UUID | None = None
    parameters: tuple[WorldModelParameter, ...] = Field(min_length=1)
    relationships: tuple[WorldModelRelationship, ...] = Field(min_length=1)
    changes: tuple[WorldModelParameterChange, ...] = ()
    update_method: str = Field(default="bootstrap_v1", min_length=1, max_length=120)
    created_at: datetime = Field(default_factory=utc_now)
    schema_version: str = Field(default="1.0", pattern=r"^\d+\.\d+$")

    @model_validator(mode="after")
    def validate_model_graph(self) -> WorldModelVersion:
        parameter_names = [parameter.name for parameter in self.parameters]
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("world-model parameter names must be unique within a version")

        relationship_keys = [relationship.key for relationship in self.relationships]
        if len(relationship_keys) != len(set(relationship_keys)):
            raise ValueError("world-model relationship keys must be unique within a version")

        known_parameters = set(parameter_names)
        unknown_references = {
            parameter_name
            for relationship in self.relationships
            for parameter_name in relationship.parameter_names
            if parameter_name not in known_parameters
        }
        if unknown_references:
            unknown = ", ".join(sorted(name.value for name in unknown_references))
            raise ValueError(f"relationships reference missing parameters: {unknown}")

        if self.version == 1 and self.based_on_version_id is not None:
            raise ValueError("world-model version 1 cannot reference a previous version")
        if self.version == 1 and self.changes:
            raise ValueError("world-model version 1 cannot contain parameter changes")
        if self.version > 1 and self.based_on_version_id is None:
            raise ValueError("world-model versions after 1 must reference a previous version")
        if self.version > 1 and not self.changes:
            raise ValueError("world-model versions after 1 must record parameter changes")
        changed_names = [change.parameter_name for change in self.changes]
        if len(changed_names) != len(set(changed_names)):
            raise ValueError("world-model changes must name each parameter at most once")
        if not set(changed_names).issubset(known_parameters):
            raise ValueError("world-model changes must reference parameters in the version")
        return self
