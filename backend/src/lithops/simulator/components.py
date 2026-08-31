"""Narrow replaceable causal components for the trusted weekly transition shell."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log1p
from typing import Protocol

from lithops.domain.public_instruments import MODEL_TIER_QUALITY_MULTIPLIER
from lithops.simulator.models import (
    PendingQualityEffect,
    SimulationAction,
    SimulationState,
    WeeklyShock,
)

STARTUP_ASSUMED_ACQUISITION_COST = 200.0


def _bounded(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


# The development response as a power law, so the shape is asked of the data
# rather than assumed: an exponent of 1 is constant returns, below 1 is
# diminishing, and the run's own observations decide which. The previous form
# was linear up to a hard ceiling, which asserted that spend past the ceiling
# buys nothing — a claim about the world we had no evidence for, and the reason
# every large quality bet forecast as pure burn. Fitting our own week-to-week
# history could not separate a concave shape from a straight line, so the
# exponent starts mildly diminishing and weakly held.
_DEFAULT_DEVELOPMENT_QUALITY_RESPONSE = 0.01
_DEFAULT_DEVELOPMENT_QUALITY_EXPONENT = 0.55
_REFERENCE_WEEKLY_DEVELOPMENT_SPEND = 10_000.0


@dataclass(frozen=True, slots=True)
class QualityTransition:
    quality: float
    delivered_quality: float
    # Delivered quality per catalog plan (A, B, C): a customer judges the plan
    # they would buy, not a blend of the three.
    delivered_quality_by_plan: tuple[float, float, float]
    conversion_multiplier: float
    pending_effects: tuple[PendingQualityEffect, ...]


@dataclass(frozen=True, slots=True)
class FunnelTransition:
    acquired_customers: float
    predicted_leads: float
    lost_leads: float


@dataclass(frozen=True, slots=True)
class LeadArrivalTransition:
    """Lead exposure and priors, before any conversion response is applied."""

    predicted_leads: float
    acquisition_baseline: float
    marketing_multiplier: float
    segment_multiplier: float


class QualityDynamicsComponent(Protocol):
    def transition(
        self,
        *,
        state: SimulationState,
        action: SimulationAction,
        development_spend: float,
        quality_lag_weeks: int,
        development_quality_response: float,
        development_quality_exponent: float,
    ) -> QualityTransition: ...


class LeadArrivalComponent(Protocol):
    def transition(
        self,
        *,
        state: SimulationState,
        action: SimulationAction,
        marketing_spend: float,
        marketing_saturation: float,
        segment_response: float,
    ) -> LeadArrivalTransition: ...


class ConversionComponent(Protocol):
    def transition(
        self,
        *,
        state: SimulationState,
        action: SimulationAction,
        lead_arrival: LeadArrivalTransition,
        retained_customers: float,
        quality_multiplier: float,
        price_multiplier: float,
        shock: WeeklyShock | None,
        delivered_quality_effective: float,
        participation_floor: float | None,
        participation_rate: float,
        participation_softness: float,
    ) -> FunnelTransition: ...


class BaselineQualityDynamics:
    """Current trusted quality equation, isolated behind a narrow contract."""

    def transition(
        self,
        *,
        state: SimulationState,
        action: SimulationAction,
        development_spend: float,
        quality_lag_weeks: int,
        development_quality_response: float = _DEFAULT_DEVELOPMENT_QUALITY_RESPONSE,
        development_quality_exponent: float = _DEFAULT_DEVELOPMENT_QUALITY_EXPONENT,
    ) -> QualityTransition:
        matured = sum(
            effect.improvement
            for effect in state.pending_quality_effects
            if effect.weeks_remaining == 1
        )
        pending = tuple(
            PendingQualityEffect(
                weeks_remaining=effect.weeks_remaining - 1,
                improvement=effect.improvement,
            )
            for effect in state.pending_quality_effects
            if effect.weeks_remaining > 1
        )
        quality = _bounded(state.product_quality + matured, 0.0, 1.0)
        current_tiers = (state.model_tier_a, state.model_tier_b, state.model_tier_c)
        selected_tiers = (
            state.model_tier_a if action.model_tier_a is None else action.model_tier_a,
            state.model_tier_b if action.model_tier_b is None else action.model_tier_b,
            state.model_tier_c if action.model_tier_c is None else action.model_tier_c,
        )
        # A customer experiences the tier of the plan they buy, so the tier
        # multiplier applies per plan. The plan delivering the most is the one a
        # quality-seeking lead would judge; averaging across plans diluted every
        # single-plan upgrade to a third of its published effect.
        current_multiplier = max(
            MODEL_TIER_QUALITY_MULTIPLIER[tier] for tier in current_tiers
        )
        selected_multiplier = max(
            MODEL_TIER_QUALITY_MULTIPLIER[tier] for tier in selected_tiers
        )
        delivered_by_plan = tuple(
            _bounded(quality * MODEL_TIER_QUALITY_MULTIPLIER[tier], 0.0, 1.0)
            for tier in selected_tiers
        )
        # Response measured against a reference week of spending, so the learned
        # size and the learned shape stay separable: the response is what a
        # reference week buys, the exponent is how that scales with a larger or
        # smaller bet. No level of spend is declared worthless.
        relative_spend = max(0.0, development_spend) / _REFERENCE_WEEKLY_DEVELOPMENT_SPEND
        improvement = development_quality_response * (
            relative_spend ** max(0.0, min(1.0, development_quality_exponent))
        )
        # Quality lives in [0, 1], so a single improvement cannot exceed that
        # range. This is what the model can represent, not a claim that spending
        # more stops working — the difference the old ceiling blurred.
        improvement = min(improvement, 1.0)
        if improvement > 0:
            pending = (
                *pending,
                PendingQualityEffect(
                    weeks_remaining=quality_lag_weeks,
                    improvement=improvement,
                ),
            )
        return QualityTransition(
            quality=quality,
            delivered_quality=_bounded(quality * selected_multiplier, 0.0, 1.0),
            delivered_quality_by_plan=delivered_by_plan,
            conversion_multiplier=max(
                0.05,
                (_bounded(quality * selected_multiplier, 0.0, 1.0) + 0.05)
                / (_bounded(state.product_quality * current_multiplier, 0.0, 1.0) + 0.05),
            ),
            pending_effects=pending,
        )


class BaselineLeadArrival:
    """Observed lead flow and spend response, independent of conversion response."""

    def transition(
        self,
        *,
        state: SimulationState,
        action: SimulationAction,
        marketing_spend: float,
        marketing_saturation: float,
        segment_response: float,
    ) -> LeadArrivalTransition:
        reference_spend = max(state.marketing_spend, 1_000.0)
        marketing_multiplier = 0.5 + marketing_saturation * log1p(
            marketing_spend / reference_spend
        )
        segment_multiplier = max(
            0.1,
            1.0 + (action.segment_focus - 1.0) * segment_response,
        )
        incremental_spend = max(0.0, marketing_spend - state.marketing_spend)
        has_evidence = state.total_leads > 0 or state.weekly_leads > 0
        startup_prior = (
            incremental_spend / STARTUP_ASSUMED_ACQUISITION_COST
            if state.weekly_acquisition == 0 and not has_evidence
            else 0.0
        )
        evidence_lead_flow = max(
            state.weekly_leads,
            state.total_leads / max(state.week, 1),
        )
        evidence_rate = (
            (state.total_conversions + 0.5) / (state.total_leads + 1.0)
            if has_evidence
            else 0.0
        )
        acquisition_baseline = max(
            state.weekly_acquisition,
            startup_prior,
            evidence_lead_flow * evidence_rate,
        )
        return LeadArrivalTransition(
            predicted_leads=evidence_lead_flow * marketing_multiplier,
            acquisition_baseline=acquisition_baseline,
            marketing_multiplier=marketing_multiplier,
            segment_multiplier=segment_multiplier,
        )


class BaselineConversion:
    """Current trusted conversion equation, unable to alter lead arrival or cash."""

    def transition(
        self,
        *,
        state: SimulationState,
        action: SimulationAction,
        lead_arrival: LeadArrivalTransition,
        retained_customers: float,
        quality_multiplier: float,
        price_multiplier: float,
        shock: WeeklyShock | None,
        delivered_quality_effective: float,
        participation_floor: float | None,
        participation_rate: float,
        participation_softness: float,
    ) -> FunnelTransition:
        desired = max(
            0.0,
            lead_arrival.acquisition_baseline
            * price_multiplier
            * lead_arrival.marketing_multiplier
            * lead_arrival.segment_multiplier
            * quality_multiplier
            * (1.0 if shock is None else shock.acquisition_multiplier),
        )
        # Participation is a threshold, not a slope: a purchased floor says the
        # most accessible group converts essentially nothing below it and a
        # learned share of arriving leads above it. The evidence-anchored term
        # cannot price that cliff — anchored at zero conversions it multiplies
        # zero forever, which made every quality lever forecast as pure burn.
        # The unlock only exists once a floor has been *bought*: an unmeasured
        # floor is unmeasured, never assumed to be zero or crossed.
        if participation_floor is not None and lead_arrival.predicted_leads > 0.0:
            softness = max(1e-3, participation_softness)
            headroom = (delivered_quality_effective - participation_floor) / softness
            gate = 1.0 / (1.0 + exp(-_bounded(headroom, -60.0, 60.0)))
            unlock = (
                lead_arrival.predicted_leads
                * max(0.0, participation_rate)
                * gate
                * price_multiplier
                * (1.0 if shock is None else shock.acquisition_multiplier)
            )
            # Both terms convert the same arriving pool, so together they never
            # exceed it; only the unlock is capped, leaving the evidence term
            # exactly as it always was.
            desired += min(
                unlock, max(0.0, lead_arrival.predicted_leads - desired)
            )
        acquired = min(desired, max(0.0, state.capacity - retained_customers))
        predicted_leads = max(lead_arrival.predicted_leads, acquired)
        return FunnelTransition(
            acquired_customers=acquired,
            predicted_leads=predicted_leads,
            lost_leads=max(0.0, predicted_leads - acquired),
        )


@dataclass(frozen=True, slots=True)
class TransitionModelAssembly:
    """Runtime composition; the outer transition retains economic ownership."""

    quality: QualityDynamicsComponent
    lead_arrival: LeadArrivalComponent
    conversion: ConversionComponent


BASELINE_TRANSITION_ASSEMBLY = TransitionModelAssembly(
    quality=BaselineQualityDynamics(),
    lead_arrival=BaselineLeadArrival(),
    conversion=BaselineConversion(),
)
