"""Rolling deterministic model-health signals from matured prediction outcomes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean
from uuid import UUID, uuid5

from lithops.domain.evaluation import (
    HorizonPerformance,
    ModelHealthSignal,
    ModelHealthStatus,
)
from lithops.domain.models import ObservationSnapshot
from lithops.domain.predictions import PredictionLedgerEntry, PredictionOutcome


@dataclass(frozen=True, slots=True)
class ModelHealthThresholds:
    minimum_rebuild_outcomes: int = 3
    normalized_error_threshold: float = 0.25
    directional_bias_threshold: float = 0.15
    required_interval_misses_in_last_three: int = 2


DEFAULT_MODEL_HEALTH_THRESHOLDS = ModelHealthThresholds()

# How many recent weekly observations the funnel diagnosis looks across. Four
# weeks of arriving leads with zero conversions is a structural anomaly whether
# or not some earlier week converted once.
FUNNEL_DIAGNOSIS_WINDOW_OBSERVATIONS = 4


def _resolve_horizons(
    entries: tuple[PredictionLedgerEntry, ...],
) -> dict[UUID, int]:
    return {
        target.id: target.horizon_days
        for entry in entries
        for target in entry.targets
    }


def evaluate_model_health(
    *,
    model_version_id: UUID,
    entries: tuple[PredictionLedgerEntry, ...],
    outcomes: tuple[PredictionOutcome, ...],
    observations: tuple[ObservationSnapshot, ...] = (),
    thresholds: ModelHealthThresholds = DEFAULT_MODEL_HEALTH_THRESHOLDS,
) -> ModelHealthSignal:
    if not outcomes:
        raise ValueError("model health requires at least one matured outcome")

    horizon_by_target = _resolve_horizons(entries)
    unknown_targets = [
        outcome.target_id
        for outcome in outcomes
        if outcome.target_id not in horizon_by_target
    ]
    if unknown_targets:
        raise ValueError("every outcome must resolve to a supplied prediction target")

    model_entries = {entry.id: entry for entry in entries}
    for outcome in outcomes:
        entry = model_entries.get(outcome.ledger_entry_id)
        if entry is None:
            raise ValueError("every outcome must belong to a supplied ledger entry")
        if entry.run_id != outcome.run_id:
            raise ValueError("prediction entry and outcome must belong to the same run")

    by_horizon: dict[int, list[PredictionOutcome]] = defaultdict(list)
    for outcome in outcomes:
        by_horizon[horizon_by_target[outcome.target_id]].append(outcome)

    horizon_performance = tuple(
        HorizonPerformance(
            horizon_days=horizon,
            outcome_count=len(items),
            mean_normalized_absolute_error=fmean(
                item.score.normalized_absolute_error for item in items
            ),
            interval_coverage=fmean(float(item.score.interval_hit) for item in items),
            mean_weighted_interval_score=fmean(
                item.score.weighted_interval_score for item in items
            ),
            signed_bias=fmean(
                item.score.signed_error / max(abs(item.actual.cash), 1.0) for item in items
            ),
        )
        for horizon, items in sorted(by_horizon.items())
    )

    ordered = sorted(outcomes, key=lambda item: (item.actual.observed_day, str(item.id)))
    last_three = ordered[-3:]
    interval_miss_count = sum(not outcome.score.interval_hit for outcome in outcomes)
    recent_misses = sum(not outcome.score.interval_hit for outcome in last_three)
    mean_normalized_error = fmean(
        outcome.score.normalized_absolute_error for outcome in outcomes
    )
    directional_bias = fmean(
        outcome.score.signed_error / max(abs(outcome.actual.cash), 1.0)
        for outcome in outcomes
    )

    enough_evidence = len(outcomes) >= thresholds.minimum_rebuild_outcomes
    trigger_codes: list[str] = []
    if (
        enough_evidence
        and recent_misses >= thresholds.required_interval_misses_in_last_three
    ):
        trigger_codes.append("two_of_last_three_interval_misses")
    if enough_evidence and mean_normalized_error >= thresholds.normalized_error_threshold:
        trigger_codes.append("rolling_normalized_error_high")
    if enough_evidence and abs(directional_bias) >= thresholds.directional_bias_threshold:
        trigger_codes.append("persistent_directional_bias")

    # Cash can look predictable while an entire commercial subsystem is plainly
    # misspecified.  This observable anomaly is a diagnosis trigger, not a claim
    # about the hidden conversion function and not a forced action.  Thirty failed
    # trials put the zero-success Wilson upper bound below roughly 12%, which is
    # enough evidence to ask the acquisition/conversion author for a new structure.
    # Measured over a recent window, not lifetime totals: one conversion in an
    # early week must not silence the diagnosis for the rest of the run while
    # hundreds of later leads convert at zero.
    recent = sorted(observations, key=lambda item: item.day)[
        -FUNNEL_DIAGNOSIS_WINDOW_OBSERVATIONS:
    ]
    windowed_leads = 0.0
    windowed_conversions = 0.0
    for snapshot in recent:
        leads = snapshot.metrics.get("weekly_leads", 0.0)
        conversions = snapshot.metrics.get("weekly_conversions", 0.0)
        if isinstance(leads, int | float):
            windowed_leads += max(0.0, float(leads))
        if isinstance(conversions, int | float):
            windowed_conversions += max(0.0, float(conversions))
    if windowed_leads >= 30.0 and windowed_conversions <= 0.0:
        trigger_codes.append("persistent_zero_conversion_funnel")

    trigger_codes = list(dict.fromkeys(trigger_codes))
    rebuild_recommended = bool(trigger_codes)
    if rebuild_recommended:
        status = ModelHealthStatus.DEGRADED
    elif interval_miss_count or mean_normalized_error >= thresholds.normalized_error_threshold:
        status = ModelHealthStatus.WATCHING
    else:
        status = ModelHealthStatus.HEALTHY

    outcome_ids = tuple(sorted((outcome.id for outcome in outcomes), key=str))
    evaluated_at = max(outcome.recorded_at for outcome in outcomes)
    run_id = outcomes[0].run_id
    signal_key = ":".join(str(outcome_id) for outcome_id in outcome_ids)
    return ModelHealthSignal(
        id=uuid5(model_version_id, f"health:{signal_key}"),
        run_id=run_id,
        model_version_id=model_version_id,
        evaluated_day=max(outcome.actual.observed_day for outcome in outcomes),
        status=status,
        outcome_ids=outcome_ids,
        horizon_performance=horizon_performance,
        interval_miss_count=interval_miss_count,
        directional_bias=directional_bias,
        rebuild_recommended=rebuild_recommended,
        trigger_codes=tuple(trigger_codes),
        evaluated_at=evaluated_at,
    )
