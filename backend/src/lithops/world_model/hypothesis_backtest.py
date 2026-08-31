"""Deterministically compile and score structured builder hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from uuid import uuid5

from lithops.domain.model_challenge import (
    AllowedRelationshipKey,
    HypothesisBacktestFold,
    HypothesisBacktestResult,
    HypothesisEvidenceKind,
    ModelBuilderProposal,
    ModelChallengePackage,
    ParameterDirection,
    ParameterStepSize,
)
from lithops.domain.world_model import (
    EvidenceKind,
    EvidenceReference,
    RelationshipShape,
    WorldModelParameter,
    WorldModelParameterName,
    WorldModelRelationship,
    WorldModelVersion,
)

SCORER_VERSION = "rolling-local-sensitivity-v1"
STEP_FRACTIONS = {
    ParameterStepSize.SMALL: 0.025,
    ParameterStepSize.MEDIUM: 0.05,
    ParameterStepSize.LARGE: 0.10,
}

RelationshipBlueprint = tuple[
    str,
    str,
    RelationshipShape,
    tuple[WorldModelParameterName, ...],
    int,
]
RELATIONSHIP_LIBRARY: dict[AllowedRelationshipKey, RelationshipBlueprint] = {
    AllowedRelationshipKey.PRICE_TO_CONVERSION: (
        "pricing",
        "conversion",
        RelationshipShape.LINEAR,
        (WorldModelParameterName.PRICE_ELASTICITY,),
        0,
    ),
    AllowedRelationshipKey.PRICE_TO_CHURN: (
        "pricing",
        "churn",
        RelationshipShape.LINEAR,
        (
            WorldModelParameterName.PRICE_ELASTICITY,
            WorldModelParameterName.CHURN_SENSITIVITY,
        ),
        1,
    ),
    AllowedRelationshipKey.MARKETING_SPEND_TO_ACQUISITION: (
        "marketing_spend",
        "acquisition",
        RelationshipShape.SATURATING,
        (WorldModelParameterName.MARKETING_SATURATION,),
        0,
    ),
    AllowedRelationshipKey.DEVELOPMENT_SPEND_TO_QUALITY: (
        "development_spend",
        "product_quality",
        RelationshipShape.LAGGED,
        (WorldModelParameterName.QUALITY_LAG_WEEKS,),
        4,
    ),
    AllowedRelationshipKey.QUALITY_TO_CHURN: (
        "product_quality",
        "churn",
        RelationshipShape.LAGGED,
        (WorldModelParameterName.CHURN_SENSITIVITY,),
        1,
    ),
    AllowedRelationshipKey.SEGMENT_TO_CONVERSION: (
        "segment_targeting",
        "conversion",
        RelationshipShape.SEGMENTED,
        (WorldModelParameterName.SEGMENT_RESPONSE,),
        0,
    ),
}


@dataclass(frozen=True, slots=True)
class CompiledHypothesis:
    candidate_model: WorldModelVersion
    changed_parameters: tuple[WorldModelParameterName, ...]
    activated_relationships: tuple[AllowedRelationshipKey, ...]
    complexity_units: int


def _allowed_evidence(package: ModelChallengePackage) -> set[tuple[str, str]]:
    model = package.active_model
    allowed = {
        (HypothesisEvidenceKind.OBSERVATION.value, item.reference)
        for item in package.observations
    }
    allowed.update(
        (
            HypothesisEvidenceKind.PREDICTION_OUTCOME.value,
            f"prediction-outcome:{item.outcome_id}",
        )
        for item in package.residuals
    )
    allowed.update(
        (
            HypothesisEvidenceKind.MODEL_PARAMETER.value,
            f"world-model:{model.id}:parameter:{item.name.value}",
        )
        for item in model.parameters
    )
    allowed.update(
        (
            HypothesisEvidenceKind.MODEL_RELATIONSHIP.value,
            f"world-model:{model.id}:relationship:{item.key}",
        )
        for item in model.relationships
    )
    return allowed


def compile_hypothesis(
    package: ModelChallengePackage,
    proposal: ModelBuilderProposal,
) -> CompiledHypothesis:
    """Turn bounded nominations into a temporary valid model; never persist it here."""

    if proposal.challenge_id != package.challenge_id:
        raise ValueError("model-builder proposal belongs to a different challenge")
    allowed_evidence = _allowed_evidence(package)
    supplied_evidence = {(item.kind.value, item.reference) for item in proposal.evidence}
    unknown_evidence = supplied_evidence - allowed_evidence
    if unknown_evidence:
        raise ValueError("model-builder proposal cites evidence outside the challenge package")

    evidence = EvidenceReference(
        kind=EvidenceKind.MODEL_BUILDER,
        reference=f"model-builder-proposal:{proposal.id}",
        observed_day=package.health_signal.evaluated_day,
        note=f"Bounded nomination from {proposal.builder_name}@{proposal.builder_version}.",
    )
    adjustments = {item.parameter_name: item for item in proposal.diff.parameter_adjustments}
    parameters: list[WorldModelParameter] = []
    changed_parameters: list[WorldModelParameterName] = []
    for parameter in package.active_model.parameters:
        adjustment = adjustments.get(parameter.name)
        if adjustment is None:
            parameters.append(parameter)
            continue
        width = parameter.upper_bound - parameter.lower_bound
        signed_fraction = STEP_FRACTIONS[adjustment.step_size]
        if adjustment.direction is ParameterDirection.DECREASE:
            signed_fraction *= -1
        estimate = min(
            parameter.upper_bound,
            max(parameter.lower_bound, parameter.estimate + width * signed_fraction),
        )
        if estimate == parameter.estimate:
            raise ValueError("model-builder parameter adjustment has no effect at its bound")
        parameters.append(
            WorldModelParameter(
                name=parameter.name,
                estimate=estimate,
                lower_bound=parameter.lower_bound,
                upper_bound=parameter.upper_bound,
                confidence=parameter.confidence,
                unit=parameter.unit,
                lag_weeks=parameter.lag_weeks,
                evidence=(*parameter.evidence, evidence),
            )
        )
        changed_parameters.append(parameter.name)
    unknown_parameters = set(adjustments) - {item.name for item in package.active_model.parameters}
    if unknown_parameters:
        raise ValueError("model-builder proposal references a parameter absent from the model")

    parameters_by_name = {item.name: item for item in parameters}
    relationships = list(package.active_model.relationships)
    active_relationships = {item.key for item in relationships}
    activated_relationships: list[AllowedRelationshipKey] = []
    for activation in proposal.diff.relationship_activations:
        key = activation.relationship_key
        if key.value in active_relationships:
            raise ValueError("model-builder proposal cannot activate an active relationship")
        cause, effect, shape, parameter_names, default_lag = RELATIONSHIP_LIBRARY[key]
        if not set(parameter_names).issubset(parameters_by_name):
            raise ValueError("allowlisted relationship requires parameters absent from the model")
        lag_weeks = default_lag
        if key is AllowedRelationshipKey.DEVELOPMENT_SPEND_TO_QUALITY:
            lag_weeks = round(
                parameters_by_name[WorldModelParameterName.QUALITY_LAG_WEEKS].estimate
            )
        relationships.append(
            WorldModelRelationship(
                key=key.value,
                cause=cause,
                effect=effect,
                shape=shape,
                parameter_names=parameter_names,
                lag_weeks=lag_weeks,
                confidence=min(parameters_by_name[name].confidence for name in parameter_names),
                evidence=(evidence,),
            )
        )
        active_relationships.add(key.value)
        activated_relationships.append(key)

    candidate = package.active_model.model_copy(
        update={"parameters": tuple(parameters), "relationships": tuple(relationships)}
    )
    candidate = WorldModelVersion.model_validate(candidate.model_dump())
    return CompiledHypothesis(
        candidate_model=candidate,
        changed_parameters=tuple(changed_parameters),
        activated_relationships=tuple(activated_relationships),
        complexity_units=len(changed_parameters) + len(activated_relationships),
    )


def _interval_score(lower: float, upper: float, actual: float) -> float:
    alpha = 0.05
    below = 2 / alpha * (lower - actual) if actual < lower else 0.0
    above = 2 / alpha * (actual - upper) if actual > upper else 0.0
    return upper - lower + below + above


def _fold_score(normalized_error: float, interval_score: float, actual_cash: float) -> float:
    return normalized_error + interval_score / max(abs(actual_cash), 1.0)


def backtest_hypothesis(
    package: ModelChallengePackage,
    proposal: ModelBuilderProposal,
    *,
    complexity_penalty_per_unit: float = 0.01,
    minimum_required_improvement: float = 0.02,
) -> tuple[CompiledHypothesis, HypothesisBacktestResult]:
    """Score chronological one-step forecasts using only pre-outcome sensitivities."""

    if complexity_penalty_per_unit < 0 or minimum_required_improvement < 0:
        raise ValueError("hypothesis backtest thresholds must be non-negative")
    compiled = compile_hypothesis(package, proposal)
    baseline_parameters = {item.name: item for item in package.active_model.parameters}
    candidate_parameters = {item.name: item for item in compiled.candidate_model.parameters}

    folds: list[HypothesisBacktestFold] = []
    baseline_fold_scores: list[float] = []
    candidate_fold_scores: list[float] = []
    for residual in package.residuals:
        sensitivities = {
            item.parameter_name: item.cash_sensitivity_per_unit
            for item in residual.parameter_sensitivities
        }
        missing = set(compiled.changed_parameters) - set(sensitivities)
        if missing:
            names = ", ".join(sorted(item.value for item in missing))
            raise ValueError(f"challenge residual lacks sensitivities for: {names}")
        forecast_shift = sum(
            sensitivities[name]
            * (candidate_parameters[name].estimate - baseline_parameters[name].estimate)
            for name in compiled.changed_parameters
        )
        candidate_point = residual.predicted_cash + forecast_shift
        candidate_lower = residual.lower_cash + forecast_shift
        candidate_upper = residual.upper_cash + forecast_shift
        candidate_error = abs(residual.actual_cash - candidate_point) / max(
            abs(residual.actual_cash), 1.0
        )
        baseline_interval_score = _interval_score(
            residual.lower_cash,
            residual.upper_cash,
            residual.actual_cash,
        )
        candidate_interval_score = _interval_score(
            candidate_lower,
            candidate_upper,
            residual.actual_cash,
        )
        folds.append(
            HypothesisBacktestFold(
                outcome_id=residual.outcome_id,
                observed_day=residual.observed_day,
                baseline_normalized_error=residual.normalized_absolute_error,
                candidate_normalized_error=candidate_error,
                baseline_interval_score=baseline_interval_score,
                candidate_interval_score=candidate_interval_score,
            )
        )
        baseline_fold_scores.append(
            _fold_score(
                residual.normalized_absolute_error,
                baseline_interval_score,
                residual.actual_cash,
            )
        )
        candidate_fold_scores.append(
            _fold_score(candidate_error, candidate_interval_score, residual.actual_cash)
        )

    baseline_score = fmean(baseline_fold_scores)
    candidate_score = fmean(candidate_fold_scores)
    raw_improvement = baseline_score - candidate_score
    complexity_penalty = compiled.complexity_units * complexity_penalty_per_unit
    penalized_improvement = raw_improvement - complexity_penalty
    supported = penalized_improvement >= minimum_required_improvement
    result = HypothesisBacktestResult(
        id=uuid5(
            proposal.id,
            f"backtest:{SCORER_VERSION}:{package.schema_version}",
        ),
        challenge_id=package.challenge_id,
        proposal_id=proposal.id,
        scorer_version=SCORER_VERSION,
        folds=tuple(folds),
        baseline_score=baseline_score,
        candidate_score=candidate_score,
        raw_improvement=raw_improvement,
        complexity_penalty=complexity_penalty,
        penalized_improvement=penalized_improvement,
        minimum_required_improvement=minimum_required_improvement,
        supported=supported,
        evaluated_at=package.created_at,
    )
    return compiled, result
