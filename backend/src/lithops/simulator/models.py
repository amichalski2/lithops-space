"""Immutable simulator inputs and outputs."""

from __future__ import annotations

from math import exp
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MarketingChannel = Literal[
    "social_media",
    "search_ads",
    "linkedin",
    "content_marketing",
    "referral_program",
]


class TargetedAdAllocation(BaseModel):
    """One exact daily CEO-Bench advertising allocation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    channel: MarketingChannel
    segment: str = Field(pattern=r"^(?:S[1-3]|E[1-3]|D_[SE]\d{2})$")
    daily_spend: float = Field(ge=0.0, le=10_000.0)


class TargetedDevelopmentAllocation(BaseModel):
    """One exact daily segment-specific product-development allocation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    segment: str = Field(pattern=r"^(?:S[1-3]|E[1-3]|D_[SE]\d{2})$")
    daily_spend: float = Field(ge=0.0, le=10_000.0)


class PendingQualityEffect(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    weeks_remaining: int = Field(ge=1, le=52)
    # Quality itself lives in [0, 1], so that is the only meaningful bound on a
    # single improvement. A tighter one silently truncated large development
    # bets, which is how a ceiling becomes a claim about the world.
    improvement: float = Field(gt=0.0, le=1.0)


class ResearchTierFacts(BaseModel):
    """One R&D tier as the environment's own read-only listing reports it.

    Observed in-run, never assumed: the cost is exact at listing time; quality
    and duration are the published sampling ranges. ``None`` on a bound means
    the listing did not quantify that side, which is not a zero.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tier: int = Field(ge=1, le=20)
    cost: float = Field(ge=0.0)
    # The listing reports sampled means, not ranges; a mean is an observed
    # statement about the distribution's centre and is carried as exactly that.
    mean_weeks: int | None = Field(default=None, ge=0, le=104)
    mean_quality_boost: float | None = Field(default=None, ge=0.0)


class PendingResearch(BaseModel):
    """A research tier whose sampled outcome has not landed yet.

    The environment allows one in-progress invocation per tier, so a candidate
    action repeated across a rollout horizon must not stack a new programme —
    or a new charge — on a tier that is still maturing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tier: int = Field(ge=1, le=20)
    weeks_remaining: int = Field(ge=1, le=52)


class ProcessNoise(BaseModel):
    """Scale of the weekly innovations a rollout applies on top of parameter draws.

    Parameter uncertainty alone collapses as recalibration tightens the bounds, which
    leaves long-horizon forecasts falsely certain. Process noise keeps an irreducible
    floor of week-to-week surprise that no amount of fitting can explain away.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    acquisition_sigma: float = Field(default=0.0, ge=0.0, le=2.0)
    churn_sigma: float = Field(default=0.0, ge=0.0, le=1.0)
    revenue_sigma: float = Field(default=0.0, ge=0.0, le=2.0)
    cash_flow_sigma: float = Field(default=0.0, ge=0.0, le=1_000_000.0)

    @property
    def active(self) -> bool:
        return bool(
            self.acquisition_sigma
            or self.churn_sigma
            or self.revenue_sigma
            or self.cash_flow_sigma
        )


class WeeklyShock(BaseModel):
    """One realized weekly innovation applied inside the state transition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    acquisition_multiplier: float = Field(default=1.0, ge=0.0)
    churn_delta: float = Field(default=0.0)
    revenue_multiplier: float = Field(default=1.0, ge=0.0)
    cash_flow_delta: float = 0.0


class SimulationState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    week: int = Field(default=0, ge=0)
    cash: float
    revenue_weekly: float = Field(ge=0.0)
    customers: float = Field(ge=0.0)
    churn_rate: float = Field(ge=0.0, le=1.0)
    # Realized weekly revenue per active customer. This intentionally keeps the
    # historical field name because persisted executable artifacts already consume
    # it with ARPU semantics.
    price_per_customer_weekly: float = Field(gt=0.0)
    # Weekly equivalent of the current catalog tier average. Catalog price governs
    # the bounded price-change envelope; realized ARPU governs unit economics.
    # ``None`` preserves legacy fixtures and callers, where the old field represented
    # both concepts.
    catalog_price_per_customer_weekly: float | None = Field(default=None, gt=0.0)
    # Lowest catalog tier in its native monthly unit. Lead promotions are monthly
    # first-period deductions, so keeping this separate avoids another period mix-up.
    entry_price_monthly: float | None = Field(default=None, gt=0.0)
    lead_promotion_monthly: float = Field(default=0.0, ge=0.0)
    # Latent, persistent conversion from catalog price to expected realized ARPU.
    # Keeping it separate prevents a one-week revenue shock from becoming a random
    # walk in every later week.
    price_realization_ratio: float | None = Field(default=None, ge=0.0)
    # The catalog's cheapest and dearest plans, weekly. A single blended price
    # cannot express the fact that better delivery is worth more per customer,
    # only that it attracts more of them — which is why every quality bet used
    # to forecast as volume at an unchanged low price, and never paid back.
    catalog_price_entry_weekly: float | None = Field(default=None, gt=0.0)
    catalog_price_premium_weekly: float | None = Field(default=None, gt=0.0)
    weekly_acquisition: float = Field(ge=0.0)
    weekly_leads: float = Field(default=0.0, ge=0.0)
    weekly_conversions: float = Field(default=0.0, ge=0.0)
    weekly_lost_leads: float = Field(default=0.0, ge=0.0)
    total_leads: float = Field(default=0.0, ge=0.0)
    total_conversions: float = Field(default=0.0, ge=0.0)
    total_lost_leads: float = Field(default=0.0, ge=0.0)
    marketing_spend: float = Field(ge=0.0)
    marketing_spend_social_media_weekly: float = Field(default=0.0, ge=0.0)
    marketing_spend_search_ads_weekly: float = Field(default=0.0, ge=0.0)
    marketing_spend_linkedin_weekly: float = Field(default=0.0, ge=0.0)
    marketing_spend_content_marketing_weekly: float = Field(default=0.0, ge=0.0)
    marketing_spend_referral_program_weekly: float = Field(default=0.0, ge=0.0)
    development_spend: float = Field(ge=0.0)
    targeted_development_spend: float = Field(default=0.0, ge=0.0)
    operations_spend: float = Field(default=0.0, ge=0.0)
    capacity_spend_weekly: float = Field(default=0.0, ge=0.0)
    product_quality: float = Field(ge=0.0, le=1.0)
    capacity: float = Field(gt=0.0)
    reputation: float = Field(ge=0.0, le=1.0)
    operating_cost_per_customer_weekly: float = Field(default=0.0, ge=0.0)
    model_tier_a: int = Field(default=1, ge=1, le=5)
    model_tier_b: int = Field(default=1, ge=1, le=5)
    model_tier_c: int = Field(default=1, ge=1, le=5)
    # Daily service allowance per customer on each plan. An allowance below what a
    # customer demands rations the value actually delivered to them. ``None`` means
    # the state carries no allowance information, which is not the same as a zero
    # allowance: no rationing is modelled for it.
    usage_quota_a: float | None = Field(default=None, ge=0.0)
    usage_quota_b: float | None = Field(default=None, ge=0.0)
    usage_quota_c: float | None = Field(default=None, ge=0.0)
    # Observed daily usage per customer. It is censored by the active allowance,
    # so it is a lower bound on demand rather than demand itself.
    daily_usage_per_customer: float = Field(default=0.0, ge=0.0)
    # Purchased estimate of demanded usage, which the observed figure cannot show
    # while the allowance censors it. ``None`` means nothing has been bought yet.
    estimated_usage_demand_per_day: float | None = Field(default=None, ge=0.0)
    # Purchased participation floors: the delivered quality below which the most
    # accessible known group reports it will not participate at all. Measured
    # facts bought in-run (research_group), never priors — ``None`` means no
    # floor has been bought, which is not a floor of zero.
    measured_quality_floor_individual: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    measured_quality_floor_enterprise: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    # The bar the run's own churn has revealed: mass cancellation at a given
    # delivered quality is a failed retention re-test, so the effective bar sits
    # above that level regardless of what any purchased floor said at purchase
    # time. ``None`` means no such week has occurred — nothing revealed.
    revealed_quality_bar_lower_bound: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    capacity_tier: int = Field(default=0, ge=0)
    # Recurring discount off the listed price, unlike the first-invoice promotion.
    recurring_promotion_monthly: float = Field(default=0.0, ge=0.0)
    # In-product advertising: revenue per customer traded against perceived quality.
    ads_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    ads_revenue_weekly: float = Field(default=0.0, ge=0.0)
    # Support and reliability work directed at existing customers.
    targeted_ops_spend: float = Field(default=0.0, ge=0.0)
    social_posts_weekly: float = Field(default=0.0, ge=0.0)
    # Enterprise demand arrives as negotiable threads billed per seat, so it is
    # tracked in seats rather than customer counts.
    active_seats: float = Field(default=0.0, ge=0.0)
    open_enterprise_threads: float = Field(default=0.0, ge=0.0)
    open_enterprise_seats: float = Field(default=0.0, ge=0.0)
    enterprise_revenue_weekly: float = Field(default=0.0, ge=0.0)
    pending_quality_effects: tuple[PendingQualityEffect, ...] = ()
    # The environment's own published R&D price list, observed read-only in-run.
    # Empty means unread — the levers exist but their prices are unquantified.
    research_catalog: tuple[ResearchTierFacts, ...] = ()
    # Tiers still maturing, so a repeated action neither restacks nor recharges.
    research_tiers_in_progress: tuple[PendingResearch, ...] = ()
    # Tiers this rollout has already started. One candidate action is ONE
    # decision: without this memory a rollout re-bought the programme every
    # time it landed — twenty charges on a 62-week horizon priced a $167k
    # decision at -$3.3M and the bankruptcy gate vetoed the lever forever.
    research_tiers_started: tuple[int, ...] = ()
    # The research charge this week's transition deducted, carried on the state
    # so the accounting bridge can declare the same term the cash flow does.
    research_spend_weekly: float = Field(default=0.0, ge=0.0)
    cash_flow_adjustment_weekly: float = 0.0

    @property
    def effective_catalog_price_per_customer_weekly(self) -> float:
        return self.catalog_price_per_customer_weekly or self.price_per_customer_weekly

    @property
    def average_usage_quota(self) -> float | None:
        """Blended plan allowance, mirroring how price and tier are blended."""

        values = (self.usage_quota_a, self.usage_quota_b, self.usage_quota_c)
        if all(value is None for value in values):
            return None
        return sum(value or 0.0 for value in values) / len(values)

    def projected_arpu(
        self,
        catalog_price_per_customer_weekly: float,
        *,
        delivered_quality: float | None = None,
        premium_quality_scale: float | None = None,
    ) -> float:
        """Carry the observed price-realization ratio into a candidate forecast.

        When the catalog's range and a delivered quality are known, the price a
        customer will carry is not fixed: dearer plans ask more of the product,
        so better delivery moves the mix up the catalog. Without that, quality
        buys volume at the entry price and a quality investment can never repay
        itself however large it is.
        """

        ratio = (
            self.price_realization_ratio
            if self.price_realization_ratio is not None
            else self.price_per_customer_weekly
            / self.effective_catalog_price_per_customer_weekly
        )
        blended = catalog_price_per_customer_weekly * ratio
        entry = self.catalog_price_entry_weekly
        premium = self.catalog_price_premium_weekly
        if (
            delivered_quality is None
            or premium_quality_scale is None
            or entry is None
            or premium is None
            or premium <= entry
        ):
            return blended
        # A saturating share: at no delivered quality the company holds only the
        # entry plan; as delivery improves it holds more of the dearer ones. The
        # scale is learned, so how demanding the upper catalog is stays an open
        # question rather than an assumption.
        premium_share = 1.0 - exp(
            -max(0.0, delivered_quality) / max(1e-6, premium_quality_scale)
        )
        mix_price = entry + premium_share * (premium - entry)
        return mix_price * ratio


class SimulationAction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    # Weekly equivalent of the proposed catalog tier average, not realized ARPU.
    price_per_customer_weekly: float = Field(gt=0.0)
    marketing_spend: float = Field(ge=0.0, le=1_000_000.0)
    targeted_ad_allocations: tuple[TargetedAdAllocation, ...] = ()
    development_spend: float = Field(ge=0.0, le=1_000_000.0)
    targeted_development_allocations: tuple[TargetedDevelopmentAllocation, ...] = ()
    operations_spend: float | None = Field(default=None, ge=0.0, le=1_000_000.0)
    model_tier_a: int | None = Field(default=None, ge=1, le=5)
    model_tier_b: int | None = Field(default=None, ge=1, le=5)
    model_tier_c: int | None = Field(default=None, ge=1, le=5)
    # Per-plan daily service allowance. Like price and tier it is a configuration
    # vector: a bounded probe reverts it through the program's baseline record
    # rather than through a per-field experiment window.
    usage_quota_a: float | None = Field(default=None, ge=0.0)
    usage_quota_b: float | None = Field(default=None, ge=0.0)
    usage_quota_c: float | None = Field(default=None, ge=0.0)
    capacity_tier: int | None = Field(default=None, ge=0)
    recurring_promotion_monthly: float | None = Field(default=None, ge=0.0)
    ads_strength: float | None = Field(default=None, ge=0.0, le=1.0)
    targeted_ops_spend: float | None = Field(default=None, ge=0.0, le=1_000_000.0)
    social_posts: int | None = Field(default=None, ge=0, le=7)
    # The authorized envelope for enterprise negotiation. Execution may move only
    # inside it, so what is negotiated can never exceed what was simulated.
    enterprise_engage: bool = False
    enterprise_target_price_per_seat: float | None = Field(default=None, gt=0.0)
    enterprise_floor_price_per_seat: float | None = Field(default=None, gt=0.0)
    enterprise_max_new_seats: float | None = Field(default=None, ge=0.0)
    # A research programme is irreversible spend with a delayed, uncertain quality
    # return; the tier is its size.
    research_project_tier: int | None = Field(default=None, ge=1, le=20)
    segment_focus: float = Field(default=1.0, ge=0.5, le=1.5)
    marketing_spend_until_week: int | None = Field(default=None, ge=1)
    development_spend_until_week: int | None = Field(default=None, ge=1)
    marketing_spend_after_experiment: float | None = Field(
        default=None,
        ge=0.0,
        le=1_000_000.0,
    )
    development_spend_after_experiment: float | None = Field(
        default=None,
        ge=0.0,
        le=1_000_000.0,
    )
    targeted_development_spend_until_week: int | None = Field(default=None, ge=1)
    targeted_development_spend_after_experiment: float | None = Field(
        default=None,
        ge=0.0,
        le=1_000_000.0,
    )
    # A staged quality experiment can defer acquisition until the quality treatment
    # has matured. Before this week the observed acquisition baseline is retained.
    marketing_spend_start_week: int | None = Field(default=None, ge=1)
    # Global first-billing-period discount offered only to new leads.
    lead_promotion_monthly: float | None = Field(default=None, ge=0.0, le=1_000_000.0)
    lead_promotion_until_week: int | None = Field(default=None, ge=1)
    lead_promotion_after_experiment: float | None = Field(
        default=None,
        ge=0.0,
        le=1_000_000.0,
    )

    @model_validator(mode="after")
    def validate_targeted_ad_allocations(self) -> SimulationAction:
        identities = [
            (allocation.channel, allocation.segment)
            for allocation in self.targeted_ad_allocations
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("targeted ad channel/segment allocations must be unique")
        if self.targeted_ad_allocations:
            allocated_weekly = 7.0 * sum(
                allocation.daily_spend
                for allocation in self.targeted_ad_allocations
            )
            tolerance = max(1e-6, abs(self.marketing_spend) * 1e-9)
            if abs(allocated_weekly - self.marketing_spend) > tolerance:
                raise ValueError(
                    "targeted ad allocations must reconcile to weekly marketing spend"
                )
        if (
            self.enterprise_floor_price_per_seat is not None
            and self.enterprise_target_price_per_seat is not None
            and self.enterprise_floor_price_per_seat
            > self.enterprise_target_price_per_seat
        ):
            raise ValueError(
                "enterprise floor price cannot exceed the target price per seat"
            )
        development_segments = [
            allocation.segment for allocation in self.targeted_development_allocations
        ]
        if len(development_segments) != len(set(development_segments)):
            raise ValueError("targeted development segment allocations must be unique")
        if sum(
            allocation.daily_spend
            for allocation in self.targeted_development_allocations
        ) > 10_000.0:
            raise ValueError("targeted development spend cannot exceed 10,000/day")
        return self


class RolloutOutcome(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rollout_index: int = Field(ge=0)
    states: tuple[SimulationState, ...] = Field(min_length=2)
    ending_cash: float
    ending_customers: float = Field(ge=0.0)
    bankrupt: bool
