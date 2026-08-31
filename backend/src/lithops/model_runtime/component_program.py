"""Deterministic fitter/compiler for typed causal component programs."""

from __future__ import annotations

from math import exp, log, sqrt

from lithops.domain.component_program import (
    ConversionComponentProgram,
    ConversionEvidence,
    ConversionFeature,
    ConversionLink,
    FittedConversionProgram,
)
from lithops.simulator.components import FunnelTransition, LeadArrivalTransition
from lithops.simulator.models import SimulationAction, SimulationState, WeeklyShock


class InsufficientComponentSupportError(ValueError):
    """Training prefix cannot identify the proposed component structure yet."""


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = exp(-min(value, 60.0))
        return 1.0 / (1.0 + inverse)
    direct = exp(max(value, -60.0))
    return direct / (1.0 + direct)


def _fit_logistic(
    rows: tuple[ConversionEvidence, ...],
    features: tuple[ConversionFeature, ...],
    means: dict[ConversionFeature, float],
    scales: dict[ConversionFeature, float],
    *,
    threshold_feature: ConversionFeature | None,
    threshold: float | None,
) -> tuple[float, dict[ConversionFeature, float], float]:
    total_leads = sum(row.leads for row in rows)
    overall = (sum(row.conversions for row in rows) + 0.5) / (total_leads + 1.0)
    intercept = log(overall / (1.0 - overall))
    coefficients = {feature: 0.0 for feature in features}
    learning_rate = 0.2
    ridge = 0.01
    for _ in range(500):
        intercept_gradient = 0.0
        gradients = {feature: 0.0 for feature in features}
        for row in rows:
            values = {
                feature: (row.features[feature] - means[feature]) / scales[feature]
                for feature in features
            }
            if threshold_feature is not None and threshold is not None:
                values[threshold_feature] = (
                    1.0 if row.features[threshold_feature] >= threshold else 0.0
                )
            linear = intercept + sum(
                coefficients[feature] * values[feature] for feature in features
            )
            predicted = _sigmoid(linear)
            error = (predicted * row.leads - row.conversions) / total_leads
            intercept_gradient += error
            for feature in features:
                gradients[feature] += error * values[feature]
        intercept -= learning_rate * intercept_gradient
        for feature in features:
            coefficients[feature] -= learning_rate * (
                gradients[feature] + ridge * coefficients[feature]
            )
    loss = 0.0
    for row in rows:
        values = {
            feature: (row.features[feature] - means[feature]) / scales[feature]
            for feature in features
        }
        if threshold_feature is not None and threshold is not None:
            values[threshold_feature] = 1.0 if row.features[threshold_feature] >= threshold else 0.0
        probability = _sigmoid(
            intercept + sum(coefficients[feature] * values[feature] for feature in features)
        )
        probability = max(1e-9, min(1.0 - 1e-9, probability))
        loss -= row.conversions * log(probability)
        loss -= (row.leads - row.conversions) * log(1.0 - probability)
    return intercept, coefficients, loss / total_leads


def fit_conversion_program(
    program: ConversionComponentProgram,
    evidence: tuple[ConversionEvidence, ...],
) -> FittedConversionProgram:
    if not evidence:
        raise ValueError("conversion component fitting requires evidence")
    missing = {
        feature
        for feature in program.features
        if any(feature not in row.features for row in evidence)
    }
    if missing:
        raise ValueError(
            "conversion evidence is missing declared features: "
            + ", ".join(sorted(feature.value for feature in missing))
        )
    means = {
        feature: sum(row.features[feature] * row.leads for row in evidence)
        / sum(row.leads for row in evidence)
        for feature in program.features
    }
    scales = {
        feature: max(
            sqrt(
                sum(row.leads * (row.features[feature] - means[feature]) ** 2 for row in evidence)
                / sum(row.leads for row in evidence)
            ),
            1e-6,
        )
        for feature in program.features
    }
    candidates: tuple[float | None, ...] = (None,)
    if program.link is ConversionLink.THRESHOLD_LOGISTIC:
        assert program.threshold_feature is not None
        values = sorted({row.features[program.threshold_feature] for row in evidence})
        if len(values) < 2:
            raise InsufficientComponentSupportError(
                "threshold hypothesis requires at least two observed regimes"
            )
        candidates = tuple(
            (left + right) / 2.0 for left, right in zip(values, values[1:], strict=False)
        )
    fits = tuple(
        (
            threshold,
            *_fit_logistic(
                evidence,
                program.features,
                means,
                scales,
                threshold_feature=program.threshold_feature,
                threshold=threshold,
            ),
        )
        for threshold in candidates
    )
    threshold, intercept, coefficients, loss = min(
        fits,
        key=lambda item: (item[3], float("-inf") if item[0] is None else item[0]),
    )
    return FittedConversionProgram(
        program=program,
        observation_ids=tuple(row.observation_id for row in evidence),
        feature_means=means,
        feature_scales=scales,
        intercept=intercept,
        coefficients=coefficients,
        threshold=threshold,
        fit_log_loss=loss,
    )


class CompiledConversionComponent:
    """Narrow runtime produced by the deterministic typed-program compiler."""

    def __init__(self, fitted: FittedConversionProgram) -> None:
        self.fitted = fitted

    def probability(self, features: dict[ConversionFeature, float]) -> float:
        values = {
            feature: (features[feature] - self.fitted.feature_means[feature])
            / self.fitted.feature_scales[feature]
            for feature in self.fitted.program.features
        }
        threshold_feature = self.fitted.program.threshold_feature
        if threshold_feature is not None and self.fitted.threshold is not None:
            values[threshold_feature] = (
                1.0 if features[threshold_feature] >= self.fitted.threshold else 0.0
            )
        return _sigmoid(
            self.fitted.intercept
            + sum(
                self.fitted.coefficients[feature] * values[feature]
                for feature in self.fitted.program.features
            )
        )

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
        delivered_quality_effective: float = 0.0,
        participation_floor: float | None = None,
        participation_rate: float = 0.0,
        participation_softness: float = 0.0,
    ) -> FunnelTransition:
        # The compiled program learns its own conversion structure — including
        # any threshold — from the run's history, so the baseline participation
        # gate parameters are not consulted here. Delivered quality, when the
        # shell supplies it, is the honest input for the quality feature.
        marketing_spend = state.marketing_spend
        if (
            action.marketing_spend_until_week is None
            or state.week < action.marketing_spend_until_week
        ):
            marketing_spend = action.marketing_spend
        channel_spend = {
            "social_media": state.marketing_spend_social_media_weekly,
            "search_ads": state.marketing_spend_search_ads_weekly,
            "linkedin": state.marketing_spend_linkedin_weekly,
            "content_marketing": state.marketing_spend_content_marketing_weekly,
            "referral_program": state.marketing_spend_referral_program_weekly,
        }
        if action.targeted_ad_allocations:
            channel_spend = {name: 0.0 for name in channel_spend}
            for allocation in action.targeted_ad_allocations:
                channel_spend[allocation.channel] += allocation.daily_spend * 7.0
        promotion = (
            state.lead_promotion_monthly
            if action.lead_promotion_monthly is None
            else action.lead_promotion_monthly
        )
        entry_price = state.entry_price_monthly or (
            state.effective_catalog_price_per_customer_weekly * 30.0 / 7.0
        )
        feature_values = {
            ConversionFeature.PRODUCT_QUALITY: max(
                0.0,
                min(
                    1.0,
                    delivered_quality_effective
                    if delivered_quality_effective > 0.0
                    else state.product_quality * quality_multiplier,
                ),
            ),
            ConversionFeature.NET_ENTRY_PRICE_MONTHLY: max(0.01, entry_price - promotion),
            ConversionFeature.REPUTATION: state.reputation,
            ConversionFeature.MARKETING_SPEND_WEEKLY: marketing_spend,
            ConversionFeature.SOCIAL_MEDIA_SPEND_WEEKLY: channel_spend["social_media"],
            ConversionFeature.SEARCH_ADS_SPEND_WEEKLY: channel_spend["search_ads"],
            ConversionFeature.LINKEDIN_SPEND_WEEKLY: channel_spend["linkedin"],
            ConversionFeature.CONTENT_MARKETING_SPEND_WEEKLY: channel_spend["content_marketing"],
            ConversionFeature.REFERRAL_PROGRAM_SPEND_WEEKLY: channel_spend["referral_program"],
        }
        probability = self.probability(feature_values)
        shock_multiplier = 1.0 if shock is None else shock.acquisition_multiplier
        exposed_leads = max(
            lead_arrival.predicted_leads,
            lead_arrival.acquisition_baseline
            * lead_arrival.marketing_multiplier
            * lead_arrival.segment_multiplier
            * shock_multiplier,
        )
        acquired = min(
            exposed_leads * probability,
            max(0.0, state.capacity - retained_customers),
        )
        return FunnelTransition(
            acquired_customers=acquired,
            predicted_leads=exposed_leads,
            lost_leads=max(0.0, exposed_leads - acquired),
        )
