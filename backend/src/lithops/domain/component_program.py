"""Typed causal programs that models may author without emitting executable Python."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lithops.domain.model_assembly import ModelComponentScope


class ConversionLink(StrEnum):
    LOGISTIC = "logistic"
    THRESHOLD_LOGISTIC = "threshold_logistic"


class ConversionFeature(StrEnum):
    PRODUCT_QUALITY = "product_quality"
    NET_ENTRY_PRICE_MONTHLY = "net_entry_price_monthly"
    REPUTATION = "reputation"
    MARKETING_SPEND_WEEKLY = "marketing_spend_weekly"
    SOCIAL_MEDIA_SPEND_WEEKLY = "social_media_spend_weekly"
    SEARCH_ADS_SPEND_WEEKLY = "search_ads_spend_weekly"
    LINKEDIN_SPEND_WEEKLY = "linkedin_spend_weekly"
    CONTENT_MARKETING_SPEND_WEEKLY = "content_marketing_spend_weekly"
    REFERRAL_PROGRAM_SPEND_WEEKLY = "referral_program_spend_weekly"


class ConversionComponentProgram(BaseModel):
    """Structure-only conversion hypothesis; parameters are fitted from evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=120, pattern=r"^[a-z][a-z0-9_]*$")
    scope: ModelComponentScope = ModelComponentScope.CONVERSION
    link: ConversionLink
    features: tuple[ConversionFeature, ...] = Field(min_length=1, max_length=8)
    threshold_feature: ConversionFeature | None = None
    rationale: str = Field(min_length=1, max_length=2_000)
    falsifiers: tuple[str, ...] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def validate_structure(self) -> ConversionComponentProgram:
        if self.scope is not ModelComponentScope.CONVERSION:
            raise ValueError("conversion program must target the conversion component")
        if len(self.features) != len(set(self.features)):
            raise ValueError("conversion program features must be unique")
        if self.link is ConversionLink.THRESHOLD_LOGISTIC:
            if self.threshold_feature is None:
                raise ValueError("threshold link requires a threshold feature")
            if self.threshold_feature not in self.features:
                raise ValueError("threshold feature must be one of the declared features")
        elif self.threshold_feature is not None:
            raise ValueError("smooth logistic link cannot declare a threshold feature")
        return self


class ConversionEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    observation_id: str = Field(min_length=1, max_length=240)
    day: int = Field(ge=0)
    leads: float = Field(gt=0.0)
    conversions: float = Field(ge=0.0)
    features: dict[ConversionFeature, float]

    @model_validator(mode="after")
    def validate_counts(self) -> ConversionEvidence:
        if self.conversions > self.leads:
            raise ValueError("conversion evidence cannot contain more conversions than leads")
        return self


class FittedConversionProgram(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    program: ConversionComponentProgram
    observation_ids: tuple[str, ...] = Field(min_length=1)
    feature_means: dict[ConversionFeature, float]
    feature_scales: dict[ConversionFeature, float]
    intercept: float
    coefficients: dict[ConversionFeature, float]
    threshold: float | None = None
    fit_log_loss: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_fit(self) -> FittedConversionProgram:
        feature_set = set(self.program.features)
        if set(self.feature_means) != feature_set:
            raise ValueError("fitted means do not match program features")
        if set(self.feature_scales) != feature_set:
            raise ValueError("fitted scales do not match program features")
        if set(self.coefficients) != feature_set:
            raise ValueError("fitted coefficients do not match program features")
        if any(value <= 0 for value in self.feature_scales.values()):
            raise ValueError("fitted feature scales must be positive")
        requires_threshold = self.program.link is ConversionLink.THRESHOLD_LOGISTIC
        if requires_threshold != (self.threshold is not None):
            raise ValueError("fitted threshold does not match program structure")
        return self
