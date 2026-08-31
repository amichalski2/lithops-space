from __future__ import annotations

import json

import pytest
from lithops.agents.candidate_model_builder import (
    ACQUISITION_BUILDER,
    PRICING_BUILDER,
    RETENTION_BUILDER,
    CandidateModelBuilder,
    ModelBuilderOutput,
)
from lithops.domain.model_challenge import HypothesisFamily
from lithops.domain.world_model import WorldModelParameterName
from lithops.infrastructure.llm.gemini_adk_provider import GeminiAdkProvider
from pydantic import ValidationError

from backend.tests.unit.test_hypothesis_backtest import challenge_package


def valid_builder_output(
    family: HypothesisFamily = HypothesisFamily.ACQUISITION_EFFICIENCY,
) -> dict[str, object]:
    return {
        "family": family.value,
        "summary": "Acquisition saturates earlier than the active model assumes.",
        "rationale": "Three consecutive cash residuals share the same direction.",
        "parameter_adjustments": [
            {
                "parameter_name": WorldModelParameterName.MARKETING_SATURATION.value,
                "direction": "decrease",
                "step_size": "medium",
            }
        ],
        "relationship_activations": [],
    }


class CapturingBuilderProvider:
    model_id = "test/free-builder"

    def __init__(self, family: HypothesisFamily) -> None:
        self.family = family
        self.payload: dict[str, object] | None = None

    async def generate_structured(self, *, system_prompt, user_prompt, output_schema):
        self.payload = json.loads(user_prompt)
        assert "actions" in system_prompt
        assert output_schema is ModelBuilderOutput
        return output_schema.model_validate(valid_builder_output(self.family))


@pytest.mark.asyncio
async def test_builder_receives_only_the_immutable_package_and_adds_trusted_metadata() -> None:
    package = challenge_package()
    provider = CapturingBuilderProvider(HypothesisFamily.ACQUISITION_EFFICIENCY)
    builder = CandidateModelBuilder(
        spec=ACQUISITION_BUILDER,
        provider=provider,
        provider_name="openrouter",
    )

    first = await builder.propose(package)
    second = await builder.propose(package)

    assert provider.payload == {
        "assigned_family": "acquisition_efficiency",
        "challenge": package.model_dump(mode="json"),
    }
    assert first == second
    assert first.builder_name == "acquisition_builder"
    assert first.provider == "openrouter"
    assert first.model_name == "test/free-builder"
    assert first.challenge_id == package.challenge_id
    assert first.evidence[0].reference == package.observations[-1].reference
    assert {
        item.reference for item in first.evidence if item.kind == "prediction_outcome"
    } == {
        f"prediction-outcome:{residual.outcome_id}"
        for residual in package.residuals[-3:]
    }
    assert all(
        item.reference.startswith(("observation:", "prediction-outcome:", "world-model:"))
        for item in first.evidence
    )


@pytest.mark.asyncio
async def test_builder_rejects_a_response_from_an_unassigned_family() -> None:
    builder = CandidateModelBuilder(
        spec=PRICING_BUILDER,
        provider=CapturingBuilderProvider(HypothesisFamily.ACQUISITION_EFFICIENCY),
        provider_name="openrouter",
    )

    with pytest.raises(ValueError, match="expected pricing_response"):
        await builder.propose(challenge_package())


@pytest.mark.asyncio
async def test_builder_rejects_a_parameter_outside_its_assigned_family() -> None:
    builder = CandidateModelBuilder(
        spec=RETENTION_BUILDER,
        provider=CapturingBuilderProvider(HypothesisFamily.RETENTION_QUALITY),
        provider_name="openrouter",
    )

    with pytest.raises(ValueError, match="outside its family"):
        await builder.propose(challenge_package())


def test_builder_output_rejects_actions_unknown_fields_and_model_supplied_evidence() -> None:
    with_action = valid_builder_output()
    with_action["execute_action"] = {"tool": "set_prices"}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ModelBuilderOutput.model_validate(with_action)

    invented_evidence = valid_builder_output()
    invented_evidence["evidence"] = [
        {
            "kind": "observation",
            "reference": "observation:invented",
            "observed_day": 21,
        }
    ]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ModelBuilderOutput.model_validate(invented_evidence)

    duplicate = valid_builder_output()
    duplicate["parameter_adjustments"] *= 2
    with pytest.raises(ValidationError, match="parameter twice"):
        ModelBuilderOutput.model_validate(duplicate)


@pytest.mark.parametrize(
    ("spec", "family"),
    [
        (PRICING_BUILDER, HypothesisFamily.PRICING_RESPONSE),
        (ACQUISITION_BUILDER, HypothesisFamily.ACQUISITION_EFFICIENCY),
        (RETENTION_BUILDER, HypothesisFamily.RETENTION_QUALITY),
    ],
)
def test_each_specialist_has_a_bounded_versioned_prompt(spec, family) -> None:
    provider = CapturingBuilderProvider(family)
    builder = CandidateModelBuilder(
        spec=spec,
        provider=provider,
        provider_name="openrouter",
    )

    assert spec.family is family
    assert spec.version == "2.0"
    assert spec.prompt_version.endswith("-v2-grounded-evidence")
    assert "never invent" in builder.system_prompt


def test_gemini_can_build_a_named_toolless_specialist_agent() -> None:
    provider = GeminiAdkProvider(
        api_key="test-key",
        model="gemini-3.7-flash",
        agent_name=RETENTION_BUILDER.name,
        agent_description="Challenges retention and capacity assumptions.",
    )

    agent = provider.build_agent(
        system_prompt="Return one bounded hypothesis.",
        output_schema=ModelBuilderOutput,
    )

    assert agent.name == RETENTION_BUILDER.name
    assert agent.tools == []
    assert provider.model_id == "gemini-3.7-flash"
