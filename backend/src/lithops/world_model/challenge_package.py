"""Build one canonical read-only package from a persistent model-health trigger."""

from __future__ import annotations

import re
from collections.abc import Iterable
from uuid import uuid5

from lithops.domain.evaluation import ModelHealthSignal
from lithops.domain.model_challenge import (
    ChallengeMetric,
    ChallengeObservation,
    ChallengeParameterSensitivity,
    ChallengeResidual,
    ModelChallengePackage,
)
from lithops.domain.models import ObservationSnapshot
from lithops.domain.predictions import PredictionLedgerEntry, PredictionOutcome
from lithops.domain.world_model import WorldModelVersion


def _normalized_metrics(
    metrics: dict[str, float | int | str | bool | None],
) -> tuple[ChallengeMetric, ...]:
    normalized: dict[str, float | int | str | bool | None] = {}
    for source_name, value in sorted(metrics.items()):
        name = re.sub(r"[^a-z0-9_]+", "_", source_name.lower()).strip("_")
        if not name or not name[0].isalpha():
            name = f"metric_{name}" if name else "metric_unknown"
        if name in normalized and normalized[name] != value:
            raise ValueError(f"challenge metric normalization conflict: {name}")
        normalized[name] = value
    return tuple(
        ChallengeMetric(name=name, value=value)
        for name, value in sorted(normalized.items())
    )

PACKAGE_SCHEMA_VERSION = "1.0"


def assemble_model_challenge_package(
    *,
    health_signal: ModelHealthSignal,
    active_model: WorldModelVersion,
    observations: Iterable[ObservationSnapshot],
    predictions: Iterable[PredictionLedgerEntry],
    outcomes: Iterable[PredictionOutcome],
) -> ModelChallengePackage:
    """Canonicalize persisted history into the exact evidence every builder receives."""

    entries = tuple(predictions)
    entries_by_id = {entry.id: entry for entry in entries}
    if len(entries_by_id) != len(entries):
        raise ValueError("challenge package predictions must be unique")

    supplied_outcomes = tuple(outcomes)
    outcomes_by_id = {outcome.id: outcome for outcome in supplied_outcomes}
    if len(outcomes_by_id) != len(supplied_outcomes):
        raise ValueError("challenge package outcomes must be unique")
    try:
        triggering_outcomes = tuple(
            outcomes_by_id[outcome_id] for outcome_id in health_signal.outcome_ids
        )
    except KeyError as exc:
        raise ValueError("challenge package is missing a health-signal outcome") from exc

    residuals: list[ChallengeResidual] = []
    observation_references_by_day: dict[int, str] = {}
    for outcome in triggering_outcomes:
        entry = entries_by_id.get(outcome.ledger_entry_id)
        if entry is None:
            raise ValueError("challenge outcome cannot resolve its prediction")
        if entry.run_id != health_signal.run_id or outcome.run_id != health_signal.run_id:
            raise ValueError("challenge history must belong to the triggering run")
        try:
            target = next(target for target in entry.targets if target.id == outcome.target_id)
        except StopIteration as exc:
            raise ValueError("challenge outcome cannot resolve its prediction target") from exc

        previous_reference = observation_references_by_day.setdefault(
            outcome.actual.observed_day,
            outcome.actual.observation_reference,
        )
        if previous_reference != outcome.actual.observation_reference:
            raise ValueError("challenge outcomes disagree on their observation reference")
        residuals.append(
            ChallengeResidual(
                outcome_id=outcome.id,
                prediction_id=entry.id,
                target_id=target.id,
                issued_day=entry.issued_day,
                horizon_days=target.horizon_days,
                target_day=target.target_day,
                observed_day=outcome.actual.observed_day,
                predicted_cash=target.point,
                lower_cash=target.lower,
                upper_cash=target.upper,
                actual_cash=outcome.actual.cash,
                signed_error=outcome.score.signed_error,
                normalized_absolute_error=outcome.score.normalized_absolute_error,
                interval_hit=outcome.score.interval_hit,
                parameter_sensitivities=tuple(
                    ChallengeParameterSensitivity(
                        parameter_name=item.parameter_name,
                        cash_sensitivity_per_unit=item.cash_sensitivity_per_unit,
                        evidence_reference=item.evidence_reference,
                    )
                    for item in sorted(
                        (
                            sensitivity
                            for sensitivity in entry.cash_sensitivities
                            if sensitivity.horizon_days == target.horizon_days
                        ),
                        key=lambda sensitivity: sensitivity.parameter_name.value,
                    )
                ),
            )
        )

    observation_items = tuple(observations)
    by_day: dict[int, ObservationSnapshot] = {}
    for observation in observation_items:
        existing = by_day.get(observation.day)
        if existing is not None and (
            existing.cash != observation.cash or existing.metrics != observation.metrics
        ):
            raise ValueError("challenge observations conflict for the same day")
        if existing is None or observation.observed_at < existing.observed_at:
            by_day[observation.day] = observation

    normalized_observations = tuple(
        ChallengeObservation(
            reference=observation_references_by_day.get(
                observation.day,
                f"observation:{health_signal.run_id}:{observation.day}",
            ),
            day=observation.day,
            cash=observation.cash,
            metrics=_normalized_metrics(observation.metrics),
            observed_at=observation.observed_at,
        )
        for observation in sorted(by_day.values(), key=lambda item: item.day)
    )

    challenge_id = uuid5(health_signal.id, f"model-challenge:{PACKAGE_SCHEMA_VERSION}")
    return ModelChallengePackage(
        challenge_id=challenge_id,
        run_id=health_signal.run_id,
        health_signal=health_signal,
        active_model=active_model,
        observations=normalized_observations,
        residuals=tuple(
            sorted(residuals, key=lambda item: (item.observed_day, str(item.outcome_id)))
        ),
        created_at=health_signal.evaluated_at,
        schema_version=PACKAGE_SCHEMA_VERSION,
    )
