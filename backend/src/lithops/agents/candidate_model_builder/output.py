"""Strict provider-facing output for candidate model builders."""

from __future__ import annotations

from uuid import uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lithops.domain.model_challenge import (
    AllowedRelationshipKey,
    HypothesisEvidenceReference,
    HypothesisFamily,
    ModelBuilderProposal,
    ModelChallengePackage,
    ParameterAdjustmentProposal,
    ParameterDirection,
    ParameterStepSize,
    RelationshipActivationProposal,
    WorldModelHypothesisDiff,
)
from lithops.domain.world_model import WorldModelParameterName


class BuilderOutputModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ParameterAdjustmentOutput(BuilderOutputModel):
    parameter_name: WorldModelParameterName
    direction: ParameterDirection
    step_size: ParameterStepSize


class RelationshipActivationOutput(BuilderOutputModel):
    relationship_key: AllowedRelationshipKey


class ModelBuilderOutput(BuilderOutputModel):
    family: HypothesisFamily
    summary: str = Field(min_length=1, max_length=500)
    rationale: str = Field(min_length=1, max_length=2_000)
    parameter_adjustments: list[ParameterAdjustmentOutput] = Field(default_factory=list)
    relationship_activations: list[RelationshipActivationOutput] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_output_shape(self) -> ModelBuilderOutput:
        if not self.parameter_adjustments and not self.relationship_activations:
            raise ValueError("builder output must nominate at least one bounded model change")
        parameter_names = [item.parameter_name for item in self.parameter_adjustments]
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("builder output cannot adjust one parameter twice")
        relationship_keys = [item.relationship_key for item in self.relationship_activations]
        if len(relationship_keys) != len(set(relationship_keys)):
            raise ValueError("builder output cannot activate one relationship twice")
        return self

    def to_domain(
        self,
        *,
        package: ModelChallengePackage,
        builder_name: str,
        builder_version: str,
        prompt_version: str,
        provider: str,
        model_name: str,
    ) -> ModelBuilderProposal:
        proposal_id = uuid5(
            package.challenge_id,
            f"builder-proposal:{builder_name}:{builder_version}",
        )
        return ModelBuilderProposal(
            id=proposal_id,
            challenge_id=package.challenge_id,
            builder_name=builder_name,
            builder_version=builder_version,
            prompt_version=prompt_version,
            provider=provider,
            model_name=model_name,
            family=self.family,
            summary=self.summary,
            rationale=self.rationale,
            diff=WorldModelHypothesisDiff(
                parameter_adjustments=tuple(
                    ParameterAdjustmentProposal(**item.model_dump())
                    for item in self.parameter_adjustments
                ),
                relationship_activations=tuple(
                    RelationshipActivationProposal(**item.model_dump())
                    for item in self.relationship_activations
                ),
            ),
            evidence=_grounded_evidence(package, self),
            created_at=package.created_at,
        )


def _grounded_evidence(
    package: ModelChallengePackage,
    output: ModelBuilderOutput,
) -> tuple[HypothesisEvidenceReference, ...]:
    """Attach only canonical identifiers already present in the immutable package.

    Evidence identity is a trust boundary, not a reasoning task.  The model chooses
    the bounded hypothesis; deterministic code commits the trigger observation,
    recent residual facts, and adjusted parameters that the backtest can verify.
    """

    latest_observation = package.observations[-1]
    evidence = [
        HypothesisEvidenceReference(
            kind="observation",
            reference=latest_observation.reference,
            observed_day=latest_observation.day,
        )
    ]
    evidence.extend(
        HypothesisEvidenceReference(
            kind="prediction_outcome",
            reference=f"prediction-outcome:{residual.outcome_id}",
            observed_day=residual.observed_day,
        )
        for residual in package.residuals[-3:]
    )
    evidence.extend(
        HypothesisEvidenceReference(
            kind="model_parameter",
            reference=(
                f"world-model:{package.active_model.id}:parameter:"
                f"{adjustment.parameter_name.value}"
            ),
            observed_day=package.active_model.source_observation_day,
        )
        for adjustment in output.parameter_adjustments
    )
    return tuple(evidence)
