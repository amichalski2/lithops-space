"""Deterministic reduction of executed experiment programs into typed outcomes.

The reducer derives exposure from committed observations and executed receipts,
never from the proposed plan. It classifies what was actually observed; it does
not interpret causes. In particular NO_EXPOSURE only states that no exposure or
observation occurred — attributing that to channel, segment, budget, timing, or
lead mechanics is the Executive's job.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from uuid import UUID

from lithops.domain.models import (
    ActionReceipt,
    DecisionRecord,
    ExperimentProgram,
    ObservationSnapshot,
    ReceiptStatus,
)
from lithops.domain.strategy import (
    EvidenceEnvelope,
    ExperimentOutcome,
    ExperimentOutcomeStatus,
    experiment_outcome_id,
)

_CHANNEL_SPEND_METRICS = {
    "social_media": "marketing_spend_social_media_weekly",
    "search_ads": "marketing_spend_search_ads_weekly",
    "linkedin": "marketing_spend_linkedin_weekly",
    "content_marketing": "marketing_spend_content_marketing_weekly",
    "referral_program": "marketing_spend_referral_program_weekly",
}


def _metric(observation: ObservationSnapshot, name: str, default: float = 0.0) -> float:
    value = observation.metrics.get(name, default)
    return float(value) if isinstance(value, int | float) else default


def probe_weeks(program: ExperimentProgram) -> tuple[int, ...]:
    """The weeks whose observations measure the program's treatment exposure."""

    if program.control in {"price", "tier", "marketing", "lead_promotion"}:
        first = program.started_week
    else:
        first = program.minimum_maturity_week
    last_exclusive = max(program.maximum_end_week, first + 1)
    return tuple(range(first, last_exclusive))


def evidence_envelope_from_observation(
    observation: ObservationSnapshot,
    *,
    segment: str | None,
    channel: str | None,
) -> EvidenceEnvelope:
    """The active offer configuration under which exposure was observed."""

    quality = _metric(observation, "product_quality", 0.5)
    prices = {
        tier.upper(): round(_metric(observation, f"price_{tier}"), 6)
        for tier in ("a", "b", "c")
        if _metric(observation, f"price_{tier}") > 0.0
    }
    tiers = {
        tier.upper(): str(int(round(_metric(observation, f"model_tier_{tier}", 1.0))))
        for tier in ("a", "b", "c")
    }
    entry_price = prices.get("A", 0.0)
    promotion_monthly = _metric(observation, "lead_promotion_monthly")
    if promotion_monthly <= 0.0:
        promotion = 0.0
    elif entry_price > 0.0:
        promotion = min(1.0, round(promotion_monthly / entry_price, 6))
    else:
        promotion = 1.0
    quality_proxies: dict[str, float] = {}
    targeted_development: dict[str, float] = {}
    quality_provenance = str(
        observation.metrics.get("product_quality_provenance") or "unavailable"
    )
    quality_decision_grade = False
    if observation.evidence is not None:
        quality_proxies = {
            f"{item.segment}:{item.plan}": item.delivered_quality_proxy
            for item in observation.evidence.quality
            if segment is None or item.segment == segment
        }
        targeted_development = {
            item.segment: item.targeted_development_daily
            for item in observation.evidence.quality
            if item.plan == "A" and item.targeted_development_daily > 0.0
        }
        if observation.evidence.quality:
            quality_provenance = observation.evidence.quality[0].provenance
            quality_decision_grade = all(
                item.decision_grade for item in observation.evidence.quality
            )
    return EvidenceEnvelope(
        segment=segment,
        channel=channel,
        quality_band=min(9, max(0, int(quality * 10))),
        catalog_prices=prices,
        model_tiers=tiers,
        promotion=promotion,
        segment_plan_quality_proxies=quality_proxies,
        quality_decision_grade=quality_decision_grade,
        quality_provenance=quality_provenance,
        targeted_development_daily=targeted_development,
    )


def _exposure_spend(
    program: ExperimentProgram,
    observations: Sequence[ObservationSnapshot],
) -> float:
    packet_spend = sum(
        row.spend
        for observation in observations
        if observation.evidence is not None
        for row in observation.evidence.acquisition
        if (program.target_segment is None or row.segment == program.target_segment)
        and (program.target_channel is None or row.channel == program.target_channel)
    )
    if packet_spend > 0.0 or any(
        observation.evidence is not None for observation in observations
    ):
        return packet_spend
    if program.target_channel is not None:
        metric_name = _CHANNEL_SPEND_METRICS[program.target_channel]
        return sum(_metric(observation, metric_name) for observation in observations)
    if program.control == "marketing":
        return sum(_metric(observation, "marketing_spend") for observation in observations)
    if program.control == "lead_promotion":
        return sum(
            _metric(observation, "lead_promotion_monthly") for observation in observations
        )
    if program.control == "targeted_development":
        return sum(
            _metric(observation, "targeted_development_spend")
            for observation in observations
        )
    return sum(_metric(observation, "development_spend") for observation in observations)


def _cohort_counts(
    program: ExperimentProgram,
    observations: Sequence[ObservationSnapshot],
) -> tuple[int, int, int]:
    packet_observations = [
        observation for observation in observations if observation.evidence is not None
    ]
    if packet_observations:
        rows = [
            row
            for observation in packet_observations
            for row in observation.evidence.cohorts  # type: ignore[union-attr]
            if (program.target_segment is None or row.segment == program.target_segment)
            and (program.target_channel is None or row.channel == program.target_channel)
        ]
        leads = sum(row.leads for row in rows)
        conversions = sum(row.conversions for row in rows)
        matured = sum(row.conversions + row.losses for row in rows)
        return leads, min(matured, leads), min(conversions, matured, leads)
    leads = int(
        round(sum(_metric(observation, "weekly_leads") for observation in observations))
    )
    conversions = int(
        round(
            sum(_metric(observation, "weekly_conversions") for observation in observations)
        )
    )
    lost_leads = int(
        round(
            sum(_metric(observation, "weekly_lost_leads") for observation in observations)
        )
    )
    matured = min(leads, conversions + lost_leads)
    return leads, matured, min(conversions, matured)


def reduce_experiment_outcome(
    *,
    run_id: UUID,
    program: ExperimentProgram,
    hypothesis_id: str | None,
    decisions: Sequence[DecisionRecord],
    receipts_by_decision: Mapping[UUID, Sequence[ActionReceipt]] | None = None,
    current_week: int,
    stopped_week: int | None = None,
) -> ExperimentOutcome:
    """Classify what one committed program actually observed.

    Deterministic over persisted decisions and receipts, so replay reduces to
    the exact same record. Immature outcomes carry no measured week and must
    not be persisted as final evidence.
    """

    program_decisions = sorted(
        (
            decision
            for decision in decisions
            if decision.action_plan.experiment_program is not None
            and decision.action_plan.experiment_program.commitment_id
            == program.commitment_id
        ),
        key=lambda decision: decision.week,
    )
    observed_by_week = {
        decision.week: decision
        for decision in program_decisions
        if decision.actual_outcome is not None
    }

    if receipts_by_decision is not None:
        for decision in program_decisions:
            receipts = receipts_by_decision.get(decision.id, ())
            if any(receipt.status is ReceiptStatus.REJECTED for receipt in receipts):
                week = max(decision.week, program.started_week)
                return ExperimentOutcome(
                    id=experiment_outcome_id(run_id, program.commitment_id, week),
                    run_id=run_id,
                    commitment_id=program.commitment_id,
                    hypothesis_id=hypothesis_id,
                    outcome_status=ExperimentOutcomeStatus.INVALID_EXECUTION,
                    envelope=_final_envelope(program, observed_by_week, week),
                    exposure_spend=0.0,
                    leads=0,
                    matured_leads=0,
                    conversions=0,
                    started_week=program.started_week,
                    measured_week=week,
                    evidence_refs=(f"decision:{decision.id}",),
                )

    if stopped_week is not None:
        week = max(stopped_week, program.started_week)
        return ExperimentOutcome(
            id=experiment_outcome_id(run_id, program.commitment_id, week),
            run_id=run_id,
            commitment_id=program.commitment_id,
            hypothesis_id=hypothesis_id,
            outcome_status=ExperimentOutcomeStatus.STOPPED_FOR_SAFETY,
            envelope=_final_envelope(program, observed_by_week, week),
            exposure_spend=_exposure_spend(
                program, _observations_up_to(observed_by_week, week)
            ),
            leads=0,
            matured_leads=0,
            conversions=0,
            started_week=program.started_week,
            measured_week=week,
            evidence_refs=_evidence_refs(observed_by_week, up_to=week),
        )

    measurement_weeks = probe_weeks(program)
    missing = [week for week in measurement_weeks if week not in observed_by_week]
    if missing:
        return ExperimentOutcome(
            id=experiment_outcome_id(run_id, program.commitment_id, current_week),
            run_id=run_id,
            commitment_id=program.commitment_id,
            hypothesis_id=hypothesis_id,
            outcome_status=ExperimentOutcomeStatus.IMMATURE,
            envelope=_final_envelope(program, observed_by_week, current_week),
            exposure_spend=0.0,
            leads=0,
            matured_leads=0,
            conversions=0,
            started_week=program.started_week,
            measured_week=None,
            evidence_refs=_evidence_refs(observed_by_week, up_to=current_week),
        )

    observations = [
        observation
        for week in measurement_weeks
        if (observation := observed_by_week[week].actual_outcome) is not None
    ]
    leads, matured_leads, conversions = _cohort_counts(program, observations)
    measured_week = measurement_weeks[-1]

    if leads <= 0:
        status = ExperimentOutcomeStatus.NO_EXPOSURE
        leads = matured_leads = conversions = 0
    elif conversions > 0:
        status = ExperimentOutcomeStatus.POSITIVE_CONVERSION
    elif matured_leads > 0:
        status = ExperimentOutcomeStatus.EXPOSED_ZERO_CONVERSION
    else:
        status = ExperimentOutcomeStatus.CENSORED

    return ExperimentOutcome(
        id=experiment_outcome_id(run_id, program.commitment_id, measured_week),
        run_id=run_id,
        commitment_id=program.commitment_id,
        hypothesis_id=hypothesis_id,
        outcome_status=status,
        envelope=_final_envelope(program, observed_by_week, measured_week),
        exposure_spend=_exposure_spend(program, observations),
        leads=leads,
        matured_leads=matured_leads,
        conversions=conversions,
        started_week=program.started_week,
        measured_week=measured_week,
        evidence_refs=_evidence_refs(observed_by_week, up_to=measured_week),
    )


def matched_experiment_evidence(
    outcomes: Sequence[ExperimentOutcome],
    envelope: EvidenceEnvelope,
) -> tuple[ExperimentOutcome, ...]:
    """Only evidence gathered under this exact support envelope.

    Replaces pooling by quality band alone: an E1/LinkedIn probe never inherits
    an S1/search conversion history merely because the quality band matches.
    """

    return tuple(
        outcome for outcome in outcomes if outcome.envelope.matches(envelope)
    )


def _observations_up_to(
    observed_by_week: Mapping[int, DecisionRecord],
    week: int,
) -> list[ObservationSnapshot]:
    return [
        observation
        for observed_week in sorted(observed_by_week)
        if observed_week <= week
        and (observation := observed_by_week[observed_week].actual_outcome) is not None
    ]


def _evidence_refs(
    observed_by_week: Mapping[int, DecisionRecord],
    *,
    up_to: int,
) -> tuple[str, ...]:
    """References only up to the classification week, so the record stays
    byte-identical when later weeks (for example the reversion) commit."""

    return tuple(
        f"decision:{observed_by_week[week].id}"
        for week in sorted(observed_by_week)
        if week <= up_to
    )


def _final_envelope(
    program: ExperimentProgram,
    observed_by_week: Mapping[int, DecisionRecord],
    week: int,
) -> EvidenceEnvelope:
    observations = _observations_up_to(observed_by_week, week)
    if not observations:
        return EvidenceEnvelope(
            segment=program.target_segment,
            channel=program.target_channel,
        )
    return evidence_envelope_from_observation(
        observations[-1],
        segment=program.target_segment,
        channel=program.target_channel,
    )
