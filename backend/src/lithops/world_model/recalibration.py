"""Bounded residual-sensitivity updates that create new immutable model versions."""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from uuid import UUID, uuid5

from lithops.domain.evaluation import ParameterResidualAttribution
from lithops.domain.predictions import PredictionLedgerEntry, PredictionOutcome
from lithops.domain.world_model import (
    EvidenceKind,
    EvidenceReference,
    WorldModelParameter,
    WorldModelParameterChange,
    WorldModelParameterName,
    WorldModelRelationship,
    WorldModelVersion,
)

UPDATE_METHOD = "bounded_residual_sensitivity_v1"


def _target_points(
    entries: tuple[PredictionLedgerEntry, ...],
) -> dict[UUID, float]:
    return {target.id: target.point for entry in entries for target in entry.targets}


def recalibrate_world_model(
    *,
    world_model: WorldModelVersion,
    entries: tuple[PredictionLedgerEntry, ...],
    outcomes: tuple[PredictionOutcome, ...],
    attributions: tuple[ParameterResidualAttribution, ...],
    learning_rate: float = 0.20,
    maximum_step_fraction: float = 0.10,
) -> WorldModelVersion:
    if not 0 < learning_rate <= 1:
        raise ValueError("learning_rate must be in (0, 1]")
    if not 0 < maximum_step_fraction <= 0.5:
        raise ValueError("maximum_step_fraction must be in (0, 0.5]")
    if not outcomes or not attributions:
        raise ValueError("recalibration requires outcomes and parameter attributions")

    outcomes_by_id = {outcome.id: outcome for outcome in outcomes}
    target_points = _target_points(entries)
    if any(outcome.target_id not in target_points for outcome in outcomes):
        raise ValueError("every outcome must resolve to a supplied prediction target")
    if any(attribution.outcome_id not in outcomes_by_id for attribution in attributions):
        raise ValueError("every attribution must reference a supplied outcome")
    if any(attribution.cash_sensitivity_per_unit == 0 for attribution in attributions):
        raise ValueError("cash sensitivity cannot be zero")

    by_parameter: dict[
        WorldModelParameterName,
        list[ParameterResidualAttribution],
    ] = defaultdict(list)
    for attribution in attributions:
        by_parameter[attribution.parameter_name].append(attribution)

    current_by_name = {parameter.name: parameter for parameter in world_model.parameters}
    unknown_parameters = set(by_parameter) - set(current_by_name)
    if unknown_parameters:
        raise ValueError("attributions reference parameters absent from the world model")

    updated_parameters: list[WorldModelParameter] = []
    changes: list[WorldModelParameterChange] = []
    for parameter in world_model.parameters:
        parameter_attributions = by_parameter.get(parameter.name)
        if not parameter_attributions:
            updated_parameters.append(parameter)
            continue

        weighted_adjustments: list[float] = []
        evidence: list[EvidenceReference] = []
        relevant_outcomes: list[PredictionOutcome] = []
        for attribution in parameter_attributions:
            outcome = outcomes_by_id[attribution.outcome_id]
            residual = outcome.actual.cash - target_points[outcome.target_id]
            weighted_adjustments.append(
                residual
                / attribution.cash_sensitivity_per_unit
                * attribution.weight
            )
            relevant_outcomes.append(outcome)
            evidence.append(
                EvidenceReference(
                    kind=EvidenceKind.PREDICTION_RESIDUAL,
                    reference=f"prediction-outcome:{outcome.id}",
                    observed_day=outcome.actual.observed_day,
                    note=(
                        f"Residual {residual:.6g}; simulator sensitivity source "
                        f"{attribution.evidence_reference}."
                    ),
                )
            )

        raw_step = learning_rate * fmean(weighted_adjustments)
        maximum_step = maximum_step_fraction * (
            parameter.upper_bound - parameter.lower_bound
        )
        bounded_step = max(-maximum_step, min(maximum_step, raw_step))
        new_estimate = max(
            parameter.lower_bound,
            min(parameter.upper_bound, parameter.estimate + bounded_step),
        )

        mean_error = fmean(
            outcome.score.normalized_absolute_error for outcome in relevant_outcomes
        )
        if any(not outcome.score.interval_hit for outcome in relevant_outcomes):
            confidence_delta = -min(0.10, 0.03 + 0.10 * mean_error)
        else:
            confidence_delta = min(0.05, 0.01 * len(relevant_outcomes))
        new_confidence = max(0.0, min(1.0, parameter.confidence + confidence_delta))
        unique_evidence = tuple(dict.fromkeys(evidence))
        updated = WorldModelParameter(
            name=parameter.name,
            estimate=new_estimate,
            lower_bound=parameter.lower_bound,
            upper_bound=parameter.upper_bound,
            confidence=new_confidence,
            unit=parameter.unit,
            lag_weeks=parameter.lag_weeks,
            evidence=(*parameter.evidence, *unique_evidence),
        )
        updated_parameters.append(updated)
        changes.append(
            WorldModelParameterChange(
                parameter_name=parameter.name,
                previous_estimate=parameter.estimate,
                new_estimate=updated.estimate,
                previous_confidence=parameter.confidence,
                new_confidence=updated.confidence,
                update_method=UPDATE_METHOD,
                evidence=unique_evidence,
            )
        )

    updated_by_name = {parameter.name: parameter for parameter in updated_parameters}
    updated_relationships = tuple(
        _update_relationship(relationship, updated_by_name, changes)
        for relationship in world_model.relationships
    )
    outcome_ids = sorted((outcome.id for outcome in outcomes), key=str)
    change_names = sorted(change.parameter_name.value for change in changes)
    version_key = ":".join([*(str(item) for item in outcome_ids), *change_names])
    return WorldModelVersion(
        id=uuid5(world_model.id, f"recalibration:{version_key}"),
        run_id=world_model.run_id,
        version=world_model.version + 1,
        source_observation_day=max(
            outcome.actual.observed_day for outcome in outcomes
        ),
        based_on_version_id=world_model.id,
        parameters=tuple(updated_parameters),
        relationships=updated_relationships,
        changes=tuple(changes),
        update_method=UPDATE_METHOD,
        created_at=max(outcome.recorded_at for outcome in outcomes),
        schema_version=world_model.schema_version,
    )


def _update_relationship(
    relationship: WorldModelRelationship,
    parameters: dict[WorldModelParameterName, WorldModelParameter],
    changes: list[WorldModelParameterChange],
) -> WorldModelRelationship:
    relevant_changes = [
        change for change in changes if change.parameter_name in relationship.parameter_names
    ]
    if not relevant_changes:
        return relationship
    new_evidence = tuple(
        dict.fromkeys(
            evidence
            for change in relevant_changes
            for evidence in change.evidence
        )
    )
    return WorldModelRelationship(
        key=relationship.key,
        cause=relationship.cause,
        effect=relationship.effect,
        shape=relationship.shape,
        parameter_names=relationship.parameter_names,
        lag_weeks=relationship.lag_weeks,
        confidence=min(
            parameters[parameter_name].confidence
            for parameter_name in relationship.parameter_names
        ),
        evidence=(*relationship.evidence, *new_evidence),
    )
