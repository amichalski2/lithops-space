"""Named candidate model builders behind the shared structured-provider port."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from lithops.agents.candidate_model_builder.output import ModelBuilderOutput
from lithops.domain.model_challenge import (
    HypothesisFamily,
    ModelBuilderProposal,
    ModelChallengePackage,
)
from lithops.domain.ports import StructuredModelProvider
from lithops.domain.world_model import WorldModelParameterName

PROMPT_DIRECTORY = Path(__file__).with_name("prompts")


@dataclass(frozen=True, slots=True)
class CandidateModelBuilderSpec:
    name: str
    version: str
    prompt_version: str
    prompt_file: str
    family: HypothesisFamily
    allowed_parameters: tuple[WorldModelParameterName, ...]


PRICING_BUILDER = CandidateModelBuilderSpec(
    name="pricing_builder",
    version="2.0",
    prompt_version="pricing-builder-v2-grounded-evidence",
    prompt_file="pricing.txt",
    family=HypothesisFamily.PRICING_RESPONSE,
    allowed_parameters=(WorldModelParameterName.PRICE_ELASTICITY,),
)
ACQUISITION_BUILDER = CandidateModelBuilderSpec(
    name="acquisition_builder",
    version="2.0",
    prompt_version="acquisition-builder-v2-grounded-evidence",
    prompt_file="acquisition.txt",
    family=HypothesisFamily.ACQUISITION_EFFICIENCY,
    allowed_parameters=(WorldModelParameterName.MARKETING_SATURATION,),
)
RETENTION_BUILDER = CandidateModelBuilderSpec(
    name="retention_capacity_builder",
    version="2.0",
    prompt_version="retention-capacity-builder-v2-grounded-evidence",
    prompt_file="retention_capacity.txt",
    family=HypothesisFamily.RETENTION_QUALITY,
    allowed_parameters=(
        WorldModelParameterName.CHURN_SENSITIVITY,
        WorldModelParameterName.QUALITY_LAG_WEEKS,
    ),
)


class CandidateModelBuilder:
    def __init__(
        self,
        *,
        spec: CandidateModelBuilderSpec,
        provider: StructuredModelProvider,
        provider_name: str,
    ) -> None:
        self.spec = spec
        self.provider = provider
        self.provider_name = provider_name
        self.system_prompt = (PROMPT_DIRECTORY / spec.prompt_file).read_text(encoding="utf-8")

    async def propose(self, package: ModelChallengePackage) -> ModelBuilderProposal:
        payload = {
            "assigned_family": self.spec.family.value,
            "challenge": package.model_dump(mode="json"),
        }
        output = await self.provider.generate_structured(
            system_prompt=self.system_prompt,
            user_prompt=json.dumps(payload, separators=(",", ":"), sort_keys=True),
            output_schema=ModelBuilderOutput,
        )
        if output.family is not self.spec.family:
            raise ValueError(
                f"{self.spec.name} returned family {output.family.value}; "
                f"expected {self.spec.family.value}"
            )
        invalid_parameters = {
            item.parameter_name for item in output.parameter_adjustments
        } - set(self.spec.allowed_parameters)
        if invalid_parameters:
            raise ValueError(
                f"{self.spec.name} returned parameters outside its family: "
                + ", ".join(sorted(item.value for item in invalid_parameters))
            )
        return output.to_domain(
            package=package,
            builder_name=self.spec.name,
            builder_version=self.spec.version,
            prompt_version=self.spec.prompt_version,
            provider=self.provider_name,
            model_name=self.provider.model_id,
        )
