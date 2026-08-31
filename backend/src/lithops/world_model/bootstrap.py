"""Build World Model v0 from conservative priors and one normalized observation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from uuid import NAMESPACE_URL, UUID, uuid5

from lithops.domain.models import ObservationSnapshot
from lithops.domain.world_model import (
    EvidenceKind,
    EvidenceReference,
    RelationshipShape,
    WorldModelParameter,
    WorldModelParameterName,
    WorldModelRelationship,
    WorldModelVersion,
)


@dataclass(frozen=True, slots=True)
class ParameterPrior:
    name: WorldModelParameterName
    estimate: float
    lower_bound: float
    upper_bound: float
    confidence: float
    unit: str
    lag_weeks: int
    supporting_metrics: tuple[str, ...]


GENERIC_PRIOR_REFERENCE = "lithops-generic-business-priors-v1"

P0_PRIORS: tuple[ParameterPrior, ...] = (
    ParameterPrior(
        name=WorldModelParameterName.PRICE_ELASTICITY,
        estimate=0.8,
        lower_bound=0.2,
        upper_bound=1.8,
        confidence=0.20,
        unit="absolute_elasticity",
        lag_weeks=0,
        supporting_metrics=("pricing", "conversion", "churn"),
    ),
    ParameterPrior(
        name=WorldModelParameterName.MARKETING_SATURATION,
        estimate=0.65,
        lower_bound=0.25,
        upper_bound=0.95,
        confidence=0.20,
        unit="saturation_ratio",
        lag_weeks=0,
        supporting_metrics=("marketing_spend", "acquisition"),
    ),
    ParameterPrior(
        name=WorldModelParameterName.CHURN_SENSITIVITY,
        estimate=0.5,
        lower_bound=0.1,
        upper_bound=1.2,
        confidence=0.20,
        unit="response_ratio",
        lag_weeks=1,
        supporting_metrics=("churn", "product_quality", "pricing"),
    ),
    ParameterPrior(
        name=WorldModelParameterName.QUALITY_LAG_WEEKS,
        estimate=4.0,
        lower_bound=2.0,
        upper_bound=8.0,
        confidence=0.25,
        unit="weeks",
        lag_weeks=4,
        supporting_metrics=("development_spend", "product_quality"),
    ),
    ParameterPrior(
        name=WorldModelParameterName.SEGMENT_RESPONSE,
        estimate=1.0,
        lower_bound=0.5,
        upper_bound=1.5,
        confidence=0.15,
        unit="response_multiplier",
        lag_weeks=0,
        supporting_metrics=("segment_count", "segment_conversion"),
    ),
    ParameterPrior(
        name=WorldModelParameterName.QUOTA_SATURATION,
        estimate=1.0,
        lower_bound=0.4,
        upper_bound=2.5,
        confidence=0.10,
        unit="rationing_exponent",
        lag_weeks=0,
        supporting_metrics=("usage_quota_a", "daily_usage_per_customer"),
    ),
    ParameterPrior(
        # A scale, not a known level: observed usage is censored at the configured
        # allowance, so this stays wide until an allowance above demand reveals it.
        name=WorldModelParameterName.QUOTA_DEMAND_REFERENCE,
        estimate=200.0,
        lower_bound=10.0,
        upper_bound=5_000.0,
        confidence=0.05,
        unit="usage_units_per_customer_day",
        lag_weeks=0,
        supporting_metrics=("daily_usage_per_customer", "usage_quota_a"),
    ),
    ParameterPrior(
        name=WorldModelParameterName.CAPACITY_TIER_STEP,
        estimate=2.5,
        lower_bound=1.2,
        upper_bound=6.0,
        confidence=0.10,
        unit="ratio_per_tier_step",
        lag_weeks=0,
        supporting_metrics=("capacity", "capacity_spend_weekly", "capacity_tier"),
    ),
    ParameterPrior(
        name=WorldModelParameterName.ADS_REVENUE_RATE,
        estimate=0.5,
        lower_bound=0.0,
        upper_bound=5.0,
        confidence=0.05,
        unit="weekly_revenue_per_customer",
        lag_weeks=0,
        supporting_metrics=("ads_revenue_weekly", "active_customers"),
    ),
    ParameterPrior(
        name=WorldModelParameterName.ADS_QUALITY_TRADEOFF,
        estimate=0.15,
        lower_bound=0.0,
        upper_bound=0.6,
        confidence=0.05,
        unit="quality_loss_at_full_ads",
        lag_weeks=0,
        supporting_metrics=("ads_revenue_weekly", "churn_rate"),
    ),
    ParameterPrior(
        name=WorldModelParameterName.OPS_RELIABILITY_RESPONSE,
        estimate=0.2,
        lower_bound=0.0,
        upper_bound=1.0,
        confidence=0.10,
        unit="churn_reduction_response",
        lag_weeks=1,
        supporting_metrics=("operations_spend", "churn_rate"),
    ),
    ParameterPrior(
        name=WorldModelParameterName.SOCIAL_LEAD_RESPONSE,
        estimate=0.1,
        lower_bound=0.0,
        upper_bound=0.5,
        confidence=0.05,
        unit="lead_multiplier_at_saturation",
        lag_weeks=0,
        supporting_metrics=("social_posts_weekly", "weekly_leads"),
    ),
    ParameterPrior(
        name=WorldModelParameterName.ENTERPRISE_PRICE_SENSITIVITY,
        estimate=1.0,
        lower_bound=0.2,
        upper_bound=3.0,
        confidence=0.05,
        unit="acceptance_elasticity",
        lag_weeks=0,
        supporting_metrics=("open_enterprise_threads", "active_seats"),
    ),
    ParameterPrior(
        name=WorldModelParameterName.PREMIUM_QUALITY_SCALE,
        estimate=0.60,
        lower_bound=0.10,
        upper_bound=3.0,
        # Weakly held: every run so far has sold almost entirely at the entry
        # price, so the observations say little about what the upper catalog
        # asks for.
        confidence=0.03,
        unit="delivered_quality",
        lag_weeks=0,
        supporting_metrics=("product_quality", "price_per_customer_weekly"),
    ),
    ParameterPrior(
        name=WorldModelParameterName.DEVELOPMENT_QUALITY_RESPONSE,
        # Quality bought by one reference week of development spend. This is a
        # generic prior and must stay one: fitting it to earlier runs of the
        # same benchmark would hand a fresh run an answer it is supposed to
        # discover, and would flatter the system by passing our learning off as
        # its own. Deliberately uninformative — a wide band around a small
        # effect, so the run's own recalibration does the work.
        estimate=0.01,
        lower_bound=0.0005,
        upper_bound=0.20,
        # Deliberately wide and weakly held: every week observed so far sits at the
        # flat end of the curve, where a great many response sizes fit equally
        # well. Only a materially larger development bet can separate them, so the
        # prior must not pretend to know the answer in advance.
        confidence=0.03,
        unit="quality_per_log_spend",
        lag_weeks=1,
        supporting_metrics=("product_quality", "development_spend"),
    ),
    ParameterPrior(
        name=WorldModelParameterName.DEVELOPMENT_QUALITY_EXPONENT,
        estimate=0.55,
        lower_bound=0.10,
        upper_bound=1.00,
        # The whole admissible range is on the table, from strongly diminishing
        # to constant returns, because a fit to our own weeks separated them by
        # less than the noise. The exponent is a question, not a finding.
        confidence=0.02,
        unit="exponent",
        lag_weeks=1,
        supporting_metrics=("product_quality", "development_spend"),
    ),
    ParameterPrior(
        name=WorldModelParameterName.RESEARCH_LAG_WEEKS_PER_TIER,
        estimate=2.0,
        lower_bound=0.5,
        upper_bound=6.0,
        confidence=0.05,
        unit="weeks_per_tier",
        lag_weeks=0,
        supporting_metrics=("product_quality",),
    ),
    ParameterPrior(
        name=WorldModelParameterName.RESEARCH_QUALITY_PER_TIER,
        estimate=0.02,
        lower_bound=0.0,
        # The quality scale itself is the only admissible ceiling. The old 0.10
        # cap was our assumption about the benchmark, and a wrong one: it made
        # the sole step-change lever forecast as a rounding error, so no run
        # ever had a reason to try it and the parameter could never be measured.
        # Wide and weakly held; the environment's own research listing, once
        # read in-run, bounds the draw with observed ranges instead.
        upper_bound=1.0,
        confidence=0.05,
        unit="quality_per_tier",
        lag_weeks=0,
        supporting_metrics=("product_quality", "development_spend"),
    ),
    ParameterPrior(
        # What share of arriving leads converts once delivered quality clears a
        # purchased participation floor. Wide and weakly held: nothing about the
        # level is assumed beyond "a cleared floor converts some of the pool";
        # the run's own weeks above the floor are what pin it down.
        name=WorldModelParameterName.PARTICIPATION_CONVERSION_RATE,
        estimate=0.15,
        lower_bound=0.02,
        upper_bound=0.60,
        confidence=0.05,
        unit="converted_share_of_leads",
        lag_weeks=0,
        supporting_metrics=("weekly_leads", "weekly_conversions", "product_quality"),
    ),
    ParameterPrior(
        # How wide the transition around a purchased floor is, in delivered
        # quality units. The floor is measured with a stated noise band, so the
        # crossing is a slope rather than a step; how steep is learned.
        name=WorldModelParameterName.PARTICIPATION_SOFTNESS,
        estimate=0.05,
        lower_bound=0.01,
        upper_bound=0.25,
        confidence=0.05,
        unit="delivered_quality",
        lag_weeks=0,
        supporting_metrics=("weekly_conversions", "product_quality"),
    ),
)


def _observation_evidence(
    observation: ObservationSnapshot,
    metric_names: Iterable[str],
) -> tuple[EvidenceReference, ...]:
    observed = tuple(sorted(name for name in metric_names if name in observation.metrics))
    if not observed:
        return ()
    return (
        EvidenceReference(
            kind=EvidenceKind.OBSERVATION,
            reference=f"observation-day-{observation.day}",
            observed_day=observation.day,
            note=f"Available signals: {', '.join(observed)}",
        ),
    )


def _build_parameter(
    prior: ParameterPrior,
    observation: ObservationSnapshot,
) -> WorldModelParameter:
    prior_evidence = EvidenceReference(
        kind=EvidenceKind.GENERIC_PRIOR,
        reference=GENERIC_PRIOR_REFERENCE,
    )
    observed_evidence = _observation_evidence(observation, prior.supporting_metrics)
    observed_fraction = sum(
        metric in observation.metrics for metric in prior.supporting_metrics
    ) / len(prior.supporting_metrics)
    # A cross-sectional snapshot is supporting context, not causal identification.
    confidence = min(0.45, prior.confidence + 0.15 * observed_fraction)
    return WorldModelParameter(
        name=prior.name,
        estimate=prior.estimate,
        lower_bound=prior.lower_bound,
        upper_bound=prior.upper_bound,
        confidence=confidence,
        unit=prior.unit,
        lag_weeks=prior.lag_weeks,
        evidence=(prior_evidence, *observed_evidence),
    )


def _relationship(
    *,
    key: str,
    cause: str,
    effect: str,
    shape: RelationshipShape,
    parameter_names: tuple[WorldModelParameterName, ...],
    parameters: dict[WorldModelParameterName, WorldModelParameter],
    lag_weeks: int = 0,
) -> WorldModelRelationship:
    evidence = tuple(
        dict.fromkeys(
            item
            for parameter_name in parameter_names
            for item in parameters[parameter_name].evidence
        )
    )
    return WorldModelRelationship(
        key=key,
        cause=cause,
        effect=effect,
        shape=shape,
        parameter_names=parameter_names,
        lag_weeks=lag_weeks,
        confidence=min(parameters[name].confidence for name in parameter_names),
        evidence=evidence,
    )


def bootstrap_world_model(
    run_id: UUID,
    observation: ObservationSnapshot,
    *,
    priors: tuple[ParameterPrior, ...] = P0_PRIORS,
) -> WorldModelVersion:
    """Return a deterministic, uncertainty-aware first model for one run."""

    parameters = tuple(_build_parameter(prior, observation) for prior in priors)
    by_name = {parameter.name: parameter for parameter in parameters}
    quality_lag = round(by_name[WorldModelParameterName.QUALITY_LAG_WEEKS].estimate)
    relationships = (
        _relationship(
            key="price_to_conversion",
            cause="pricing",
            effect="conversion",
            shape=RelationshipShape.LINEAR,
            parameter_names=(WorldModelParameterName.PRICE_ELASTICITY,),
            parameters=by_name,
        ),
        _relationship(
            key="price_to_churn",
            cause="pricing",
            effect="churn",
            shape=RelationshipShape.LINEAR,
            parameter_names=(
                WorldModelParameterName.PRICE_ELASTICITY,
                WorldModelParameterName.CHURN_SENSITIVITY,
            ),
            parameters=by_name,
            lag_weeks=1,
        ),
        _relationship(
            key="marketing_spend_to_acquisition",
            cause="marketing_spend",
            effect="acquisition",
            shape=RelationshipShape.SATURATING,
            parameter_names=(WorldModelParameterName.MARKETING_SATURATION,),
            parameters=by_name,
        ),
        _relationship(
            key="development_spend_to_quality",
            cause="development_spend",
            effect="product_quality",
            shape=RelationshipShape.LAGGED,
            parameter_names=(WorldModelParameterName.QUALITY_LAG_WEEKS,),
            parameters=by_name,
            lag_weeks=quality_lag,
        ),
        _relationship(
            key="quality_to_churn",
            cause="product_quality",
            effect="churn",
            shape=RelationshipShape.LAGGED,
            parameter_names=(WorldModelParameterName.CHURN_SENSITIVITY,),
            parameters=by_name,
            lag_weeks=1,
        ),
        _relationship(
            key="segment_to_conversion",
            cause="segment_targeting",
            effect="conversion",
            shape=RelationshipShape.SEGMENTED,
            parameter_names=(WorldModelParameterName.SEGMENT_RESPONSE,),
            parameters=by_name,
        ),
    )
    return WorldModelVersion(
        id=uuid5(NAMESPACE_URL, f"lithops:{run_id}:world-model:1"),
        run_id=run_id,
        version=1,
        source_observation_day=observation.day,
        parameters=parameters,
        relationships=relationships,
        created_at=observation.observed_at,
    )
