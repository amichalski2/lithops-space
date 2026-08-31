"""Deterministic specialists for orchestration, recovery, and demo evidence tests."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid5

from lithops.agents.candidate_model_builder.agent import CandidateModelBuilderSpec
from lithops.domain.model_challenge import (
    HypothesisEvidenceKind,
    HypothesisEvidenceReference,
    ModelBuilderProposal,
    ModelChallengePackage,
    ParameterAdjustmentProposal,
    ParameterDirection,
    ParameterStepSize,
    WorldModelHypothesisDiff,
)
from lithops.domain.world_model import WorldModelParameterName


@dataclass(frozen=True, slots=True)
class _DeterministicProviderIdentity:
    model_id: str = "bounded-hypothesis-fixture-v1"


class DeterministicCandidateModelBuilder:
    """Emits one fixed bounded nomination while preserving production metadata."""

    provider_name = "deterministic"
    provider = _DeterministicProviderIdentity()

    def __init__(
        self,
        *,
        spec: CandidateModelBuilderSpec,
        parameter_name: WorldModelParameterName,
        direction: ParameterDirection,
        step_size: ParameterStepSize = ParameterStepSize.MEDIUM,
    ) -> None:
        self.spec = spec
        self.parameter_name = parameter_name
        self.direction = direction
        self.step_size = step_size

    async def propose(self, package: ModelChallengePackage) -> ModelBuilderProposal:
        residual = package.residuals[-1]
        observation = package.observations[-1]
        return ModelBuilderProposal(
            id=uuid5(package.challenge_id, f"proposal:{self.spec.name}"),
            challenge_id=package.challenge_id,
            builder_name=self.spec.name,
            builder_version=self.spec.version,
            prompt_version=self.spec.prompt_version,
            provider=self.provider_name,
            model_name=self.provider.model_id,
            family=self.spec.family,
            summary=f"Bounded {self.spec.family.value} challenge.",
            rationale="The nominated direction is evaluated only by deterministic replay.",
            diff=WorldModelHypothesisDiff(
                parameter_adjustments=(
                    ParameterAdjustmentProposal(
                        parameter_name=self.parameter_name,
                        direction=self.direction,
                        step_size=self.step_size,
                    ),
                ),
            ),
            evidence=(
                HypothesisEvidenceReference(
                    kind=HypothesisEvidenceKind.OBSERVATION,
                    reference=observation.reference,
                    observed_day=observation.day,
                ),
                HypothesisEvidenceReference(
                    kind=HypothesisEvidenceKind.PREDICTION_OUTCOME,
                    reference=f"prediction-outcome:{residual.outcome_id}",
                    observed_day=residual.observed_day,
                ),
            ),
            created_at=package.created_at,
        )
