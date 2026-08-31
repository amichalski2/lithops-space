from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from lithops.agents.common import (
    ExecutiveActionProposalOutput,
    ExecutiveDecisionOutput,
    ExecutiveProposalOutput,
)
from lithops.agents.executive import ExecutiveDecisionEngine
from lithops.config import Settings
from lithops.domain.models import ObservationSnapshot, RunRecord
from lithops.infrastructure.llm.gemini_adk_provider import (
    GeminiAdkProvider,
    gemini_developer_schema,
)
from lithops.infrastructure.llm.openrouter_provider import OpenRouterProvider
from pydantic import ValidationError


def valid_output() -> dict[str, Any]:
    return {
        "name": "Measured weekly investment",
        "strategy_family": "controlled_exploration",
        "rationale": "Keep the experiment bounded while preserving runway.",
        "daily_spend": {"operations": 500, "development": 250},
        "cash_forecasts": [
            {"horizon_days": "7", "point": 995_000, "lower": 980_000, "upper": 1_010_000},
            {"horizon_days": "28", "point": 980_000, "lower": 920_000, "upper": 1_040_000},
            {"horizon_days": "84", "point": 940_000, "lower": 760_000, "upper": 1_100_000},
            {
                "horizon_days": "182",
                "point": 900_000,
                "lower": 500_000,
                "upper": 1_250_000,
            },
        ],
    }


def valid_proposal_output() -> dict[str, Any]:
    return {
        "decision_summary": "Compare continuation with a bounded quality hypothesis.",
        "candidates": [
            {
                "name": "Continue and observe",
                "hypothesis_id": "continuation_is_safest",
                "proposal_kind": "operating",
                "experiment_control": "none",
                "strategy_family": "continuation",
                "hypothesis": "Current controls remain the least destructive option.",
                "expected_observation": "Cash burn and funnel rates remain stable.",
                "rationale": "Negative funnel evidence does not support more acquisition spend.",
                "catalog_price_multiplier": 1.0,
                "weekly_marketing_spend": 0,
                "daily_spend": {"operations": 500, "development": 250},
                "model_tier_a": 1,
                "model_tier_b": 1,
                "model_tier_c": 1,
                "lead_promotion_fraction": 0,
                "target_channel": "search_ads",
                "target_segment": "S1",
            },
            {
                "name": "Test delivered quality",
                "hypothesis_id": "quality_blocks_conversion",
                "proposal_kind": "operating",
                "experiment_control": "none",
                "strategy_family": "product_quality",
                "hypothesis": "Low delivered quality blocks otherwise valid leads.",
                "expected_observation": "Quality rises before conversion evidence changes.",
                "rationale": "Change quality controls while holding acquisition spend bounded.",
                "catalog_price_multiplier": 1.0,
                "weekly_marketing_spend": 700,
                "daily_spend": {"operations": 500, "development": 500},
                "model_tier_a": 2,
                "model_tier_b": 2,
                "model_tier_c": 2,
                "lead_promotion_fraction": 0.1,
                "target_channel": "linkedin",
                "target_segment": "E1",
            },
            {
                "name": "One-week development probe",
                "hypothesis_id": "development_lag_is_blocking",
                "proposal_kind": "experiment",
                "experiment_control": "development",
                "experiment_duration_weeks": 4,
                "strategy_family": "product_quality",
                "hypothesis": "A bounded development increment changes observed quality.",
                "expected_observation": "Quality moves after the documented lag.",
                "rationale": "Test one causal control without increasing acquisition spend.",
                "catalog_price_multiplier": 1.2,
                "weekly_marketing_spend": 7_000,
                "daily_spend": {"operations": 900, "development": 600},
                "model_tier_a": 5,
                "model_tier_b": 5,
                "model_tier_c": 5,
                "lead_promotion_fraction": 0.25,
                "target_channel": "search_ads",
                "target_segment": "S1",
            },
            {
                "name": "Build S2 quality, then measure demand",
                "hypothesis_id": "s2_quality_frontier",
                "proposal_kind": "experiment",
                "experiment_control": "targeted_development",
                "experiment_duration_weeks": 3,
                "strategy_family": "product_quality",
                "hypothesis": "S2 leads were observed only below the relevant quality support.",
                "expected_observation": "Quality changes before a one-week S2 acquisition probe.",
                "rationale": "Create overlap between changed product support and lead exposure.",
                "catalog_price_multiplier": 1.0,
                "weekly_marketing_spend": 7_000,
                "daily_spend": {"operations": 500, "development": 250},
                "targeted_development_daily": 2_000,
                "model_tier_a": 1,
                "model_tier_b": 1,
                "model_tier_c": 1,
                "lead_promotion_fraction": 0,
                "target_channel": "social_media",
                "target_segment": "S2",
            },
        ],
    }


def test_executive_output_is_bounded_and_converts_to_domain_models() -> None:
    output = ExecutiveDecisionOutput.model_validate(valid_output())
    run = RunRecord()

    plan, forecasts = output.to_domain(run_id=run.id, week=3)

    assert plan.commands[0].tool == "set_daily_spend"
    assert plan.commands[0].idempotency_key == f"{run.id}:week-3:executive-spend-0"
    assert [item.horizon_days for item in forecasts.ordered()] == [7, 28, 84, 182]

    invalid = valid_output()
    invalid["daily_spend"]["operations"] = 10_001
    with pytest.raises(ValidationError):
        ExecutiveDecisionOutput.model_validate(invalid)


class CapturingProvider:
    model_id = "fake/structured"

    def __init__(self) -> None:
        self.output_schema = None

    async def generate_structured(self, *, system_prompt, user_prompt, output_schema):
        self.output_schema = output_schema
        assert "Lithops Executive v1" in system_prompt
        assert json.loads(user_prompt)["observation"]["cash"] == 1_000_000
        return output_schema.model_validate(valid_output())


@pytest.mark.asyncio
async def test_executive_engine_requests_structured_output() -> None:
    provider = CapturingProvider()
    engine = ExecutiveDecisionEngine(provider)

    plan, forecasts = await engine.decide(
        run=RunRecord(),
        observation=ObservationSnapshot(day=0, cash=1_000_000),
    )

    assert provider.output_schema is ExecutiveDecisionOutput
    assert plan.strategy_family == "controlled_exploration"
    assert len(forecasts.items) == 4


class ProposalCapturingProvider:
    model_id = "fake/proposal"

    def __init__(self) -> None:
        self.payload = None

    async def generate_structured(self, *, system_prompt, user_prompt, output_schema):
        assert "Executive v3" in system_prompt
        self.payload = json.loads(user_prompt)
        assert output_schema is ExecutiveProposalOutput
        return output_schema.model_validate(valid_proposal_output())


@pytest.mark.asyncio
async def test_executive_v2_proposes_multiple_semantic_action_plans() -> None:
    provider = ProposalCapturingProvider()
    engine = ExecutiveDecisionEngine(provider)
    run = RunRecord(horizon_days=500)
    observation = ObservationSnapshot(
        day=14,
        cash=900_000,
        metrics={
            "price_a": 25,
            "price_b": 69,
            "price_c": 179,
            "known_segments": "S1,E1",
            "lead_promotion_monthly": 0,
        },
    )

    batch = await engine.propose_actions(run=run, observation=observation)

    plans = batch.plans
    assert batch.rejections == ()
    assert len(plans) == 4
    assert provider.payload["remaining_days"] == 486
    # The declared surface is the model's whole sense of what it may consider,
    # so it must track what the run can execute. This assertion once pinned
    # `enterprise_deals` as unavailable long after the negotiation loop shipped,
    # which hid the benchmark's largest revenue lever from every proposal.
    capabilities = provider.payload["semantic_capabilities"]
    assert capabilities["unavailable_until_modeled"] == []
    assert {
        "enterprise_deals",
        "research_programmes",
        "recurring_promotion",
        "in_product_ads_strength",
        "owned_channel_social_post",
        "targeted_operations_spend",
    } <= set(capabilities["supported"])
    quality = plans[1]
    by_tool = {command.tool: command for command in quality.commands}
    assert set(by_tool) == {
        "set_prices",
        "set_model_tiers",
        "set_usage_quotas",
        "set_capacity_tier",
        "set_daily_spend",
        "set_targeted_ad_spend",
        "set_targeted_dev_spend",
        "set_lead_promotion",
    }
    assert by_tool["set_model_tiers"].arguments == {"A": 2, "B": 2, "C": 2}
    assert by_tool["set_targeted_ad_spend"].arguments == {
        "targeted_spend": {"linkedin": {"E1": 100.0}}
    }
    assert by_tool["set_lead_promotion"].arguments == {"global_promotion": 2.5}
    assert "Low delivered quality" in quality.rationale
    experiment = plans[2]
    experiment_by_tool = {command.tool: command for command in experiment.commands}
    assert experiment.strategy_family == (
        "executive_experiment_development_development_lag_is_blocking"
    )
    assert experiment.proposal_kind == "experiment"
    assert experiment.hypothesis_id == "development_lag_is_blocking"
    assert experiment.experiment_control == "development"
    assert experiment.evidence_regime == "leads_none:quality_0:customers_zero"
    assert experiment.experiment_expires_week == 7
    assert experiment.experiment_program is not None
    assert experiment.experiment_program.minimum_maturity_week == 6
    assert experiment.experiment_program.maximum_end_week == 7
    assert experiment.experiment_program.acquisition_probe_weekly_spend == 7_000
    assert experiment.experiment_program.control == "development"
    assert experiment_by_tool["set_prices"].arguments == {
        "A": 25.0,
        "B": 69.0,
        "C": 179.0,
    }
    assert experiment_by_tool["set_model_tiers"].arguments == {
        "A": 1,
        "B": 1,
        "C": 1,
    }
    assert experiment_by_tool["set_daily_spend"].arguments == {
        "operations": 0.0,
        "development": 600.0,
    }
    assert experiment_by_tool["set_targeted_ad_spend"].arguments == {
        "targeted_spend": {}
    }
    assert "set_lead_promotion" not in experiment_by_tool
    targeted = plans[3]
    targeted_by_tool = {command.tool: command for command in targeted.commands}
    assert targeted.experiment_control == "targeted_development"
    assert targeted.experiment_program is not None
    assert targeted.experiment_program.minimum_maturity_week == 5
    assert targeted.experiment_program.maximum_end_week == 6
    assert targeted.experiment_program.acquisition_probe_weekly_spend == 7_000
    # The requested segment is carried through as asked. If the run has never
    # observed it the candidate is refused by name downstream, rather than being
    # quietly retargeted at a segment nobody chose.
    assert targeted_by_tool["set_targeted_dev_spend"].arguments == {
        "targeted_spend": {"S2": 2_000.0}
    }
    assert targeted.evidence_regime is not None
    assert targeted.evidence_regime.endswith(":unobserved_segment")
    assert targeted_by_tool["set_targeted_ad_spend"].arguments == {
        "targeted_spend": {}
    }


@pytest.mark.parametrize(
    ("control", "expected_prices", "expected_tiers"),
    [
        ("price", {"A": 20.0, "B": 55.2, "C": 143.2}, {"A": 1, "B": 2, "C": 3}),
        ("tier", {"A": 25.0, "B": 69.0, "C": 179.0}, {"A": 5, "B": 5, "C": 5}),
    ],
)
def test_experiment_control_materially_changes_price_or_tier(
    control: str,
    expected_prices: dict[str, float],
    expected_tiers: dict[str, int],
) -> None:
    payload = valid_proposal_output()["candidates"][2]
    payload.update(
        {
            "name": f"Test {control}",
            "hypothesis_id": f"{control}_changes_conversion",
            "experiment_control": control,
            "experiment_duration_weeks": 1,
            "catalog_price_multiplier": 0.8,
            "model_tier_a": 5,
            "model_tier_b": 5,
            "model_tier_c": 5,
        }
    )
    proposal = ExecutiveActionProposalOutput.model_validate(payload)
    observation = ObservationSnapshot(
        day=42,
        cash=900_000,
        metrics={
            "price_a": 25,
            "price_b": 69,
            "price_c": 179,
            "model_tier_a": 1,
            "model_tier_b": 2,
            "model_tier_c": 3,
            "known_segments": "S1",
        },
    )

    plan = ExecutiveDecisionEngine._proposal_plan(
        proposal,
        run=RunRecord(horizon_days=500),
        observation=observation,
        candidate_index=0,
    )
    by_tool = {command.tool: command for command in plan.commands}

    assert by_tool["set_prices"].arguments == pytest.approx(expected_prices)
    assert by_tool["set_model_tiers"].arguments == expected_tiers
    assert plan.experiment_program is not None
    assert plan.experiment_program.protocol_version == "experiment-program-v2"
    assert plan.experiment_program.baseline_configuration != (
        plan.experiment_program.treatment_configuration
    )
    assert {item.source for item in plan.experiment_program.measurement_plan} == {
        "configuration",
        "cohort",
        "quality",
        "ledger",
    }


@pytest.mark.asyncio
async def test_openrouter_sends_strict_json_schema_and_validates_response() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["strict"] is True
        assert payload["provider"]["require_parameters"] is True
        assert payload["provider"]["sort"] == "throughput"
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(valid_output())}}
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenRouterProvider(
            api_key="test-key",
            client=client,
            provider_sort="throughput",
        )
        output = await provider.generate_structured(
            system_prompt="system",
            user_prompt="input",
            output_schema=ExecutiveDecisionOutput,
        )

    assert output.daily_spend.operations == 500
    assert provider.model_id == "qwen/qwen3-32b"


@pytest.mark.asyncio
async def test_openrouter_retry_includes_invalid_response_and_validation_feedback() -> None:
    calls: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        output = valid_output()
        if len(calls) == 1:
            output["unexpected"] = "invalid"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(output)}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        output = await OpenRouterProvider(
            api_key="test-key",
            client=client,
        ).generate_structured(
            system_prompt="system",
            user_prompt="input",
            output_schema=ExecutiveDecisionOutput,
        )

    assert output.name == "Measured weekly investment"
    assert len(calls) == 2
    assert calls[1]["messages"][-2]["role"] == "assistant"
    assert "unexpected" in calls[1]["messages"][-2]["content"]
    assert "Validation errors" in calls[1]["messages"][-1]["content"]


def test_gemini_provider_builds_a_toolless_adk_agent_with_compatible_schema() -> None:
    provider = GeminiAdkProvider(api_key="test-key", model="gemini-3.7-flash")

    agent = provider.build_agent(
        system_prompt="Return a bounded decision.",
        output_schema=ExecutiveDecisionOutput,
    )

    assert isinstance(agent.output_schema, dict)
    assert "additionalProperties" not in json.dumps(agent.output_schema)
    assert agent.output_schema["$defs"]["CashForecastOutput"]["properties"][
        "horizon_days"
    ]["enum"] == ["7", "28", "84", "182"]
    assert agent.tools == []
    assert agent.model.model == "gemini-3.7-flash"
    assert provider.model_id == "gemini-3.7-flash"


def test_gemini_schema_sanitization_does_not_weaken_pydantic_validation() -> None:
    original = ExecutiveDecisionOutput.model_json_schema()
    compatible = gemini_developer_schema(ExecutiveDecisionOutput)

    assert original["additionalProperties"] is False
    assert original["$defs"]["SpendAllocation"]["additionalProperties"] is False
    assert "additionalProperties" not in json.dumps(compatible)
    assert "\"default\"" not in json.dumps(compatible)
    assert "\"pattern\"" not in json.dumps(compatible)
    assert "minLength" not in json.dumps(compatible)
    assert "maxLength" not in json.dumps(compatible)
    assert "minItems" not in json.dumps(compatible)
    assert "maxItems" not in json.dumps(compatible)
    assert all(definition for definition in compatible.get("$defs", {}).values())

    invalid = valid_output()
    invalid["unexpected"] = "still rejected after Gemini returns"
    with pytest.raises(ValidationError):
        ExecutiveDecisionOutput.model_validate(invalid)


@pytest.mark.asyncio
async def test_gemini_retry_includes_invalid_response_and_validation_feedback(
    monkeypatch,
) -> None:
    provider = GeminiAdkProvider(api_key="test-key", model="gemini-3.7-flash")
    invalid = valid_output()
    invalid["unexpected"] = "invalid"
    responses = iter((json.dumps(invalid), json.dumps(valid_output())))
    prompts: list[str] = []

    async def fake_generate_text(*, system_prompt, user_prompt, output_schema):
        prompts.append(user_prompt)
        return next(responses)

    monkeypatch.setattr(provider, "_generate_text", fake_generate_text)
    output = await provider.generate_structured(
        system_prompt="system",
        user_prompt="input",
        output_schema=ExecutiveDecisionOutput,
    )

    assert output.name == "Measured weekly investment"
    assert len(prompts) == 2
    assert "unexpected" in prompts[1]
    assert "Validation errors" in prompts[1]


def test_model_provider_settings_require_the_selected_credential() -> None:
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        Settings(model_provider="openrouter").validate()
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        Settings(model_provider="gemini").validate()

    Settings(model_provider="openrouter", openrouter_api_key="test-key").validate()
    Settings(model_provider="gemini", gemini_api_key="test-key").validate()


def _targeted_development_proposal(**overrides: Any) -> dict[str, Any]:
    proposal = {
        "name": "Probe the improved quality regime",
        "hypothesis_id": "quality_regime_untested",
        "proposal_kind": "experiment",
        "experiment_control": "targeted_development",
        "experiment_duration_weeks": 3,
        "strategy_family": "product_quality",
        "hypothesis": "Zero conversions were measured under a quality regime we changed.",
        "expected_observation": "Conversions appear once the improved regime is probed.",
        "rationale": "Existing evidence cannot speak about an untested quality regime.",
        "catalog_price_multiplier": 1.0,
        "weekly_marketing_spend": 4_000,
        "targeted_development_daily": 400,
        "daily_spend": {"operations": 500, "development": 500},
        "model_tier_a": 2,
        "model_tier_b": 2,
        "model_tier_c": 2,
        "lead_promotion_fraction": 0.0,
        "target_channel": "linkedin",
        "target_segment": "E1",
    }
    proposal.update(overrides)
    return proposal


def test_targeted_development_without_probe_spend_is_refused_at_the_schema_boundary() -> None:
    """The invariant is mirrored where the author can still fix it: in its own answer."""

    with pytest.raises(ValidationError, match="acquisition_probe_missing"):
        ExecutiveActionProposalOutput.model_validate(
            _targeted_development_proposal(weekly_marketing_spend=0)
        )

    ExecutiveActionProposalOutput.model_validate(_targeted_development_proposal())


def test_development_without_probe_spend_stays_a_legal_proposal() -> None:
    """Only targeted_development crashes construction, so only it is mirrored here.

    A plain development experiment with no probe is still vetoed, but after
    simulation and by an evaluation card, which is where that judgement belongs.
    """

    ExecutiveActionProposalOutput.model_validate(
        _targeted_development_proposal(
            experiment_control="development",
            weekly_marketing_spend=0,
        )
    )


def test_experiment_identity_wider_than_the_plan_allows_is_refused() -> None:
    with pytest.raises(ValidationError, match="strategy_identity_too_long"):
        ExecutiveActionProposalOutput.model_validate(
            _targeted_development_proposal(hypothesis_id="q" * 40)
        )


@pytest.mark.asyncio
async def test_one_unbuildable_proposal_is_refused_without_losing_the_batch() -> None:
    """A proposal that cannot become a plan costs its own candidacy, nothing more."""

    payload = valid_proposal_output()
    poisoned = _targeted_development_proposal(weekly_marketing_spend=0)
    candidates = [
        ExecutiveActionProposalOutput.model_validate(item)
        for item in payload["candidates"]
    ]
    # Bypassing validation here is the point: it exercises the construction-time
    # safety net rather than the schema boundary tested above.
    candidates.insert(1, ExecutiveActionProposalOutput.model_construct(**poisoned))

    class PoisonedProvider:
        model_id = "fake/poisoned"

        async def generate_structured(self, *, system_prompt, user_prompt, output_schema):
            return ExecutiveProposalOutput.model_construct(
                decision_summary=payload["decision_summary"],
                candidates=candidates,
            )

    engine = ExecutiveDecisionEngine(PoisonedProvider())
    batch = await engine.propose_actions(
        run=RunRecord(horizon_days=500),
        observation=ObservationSnapshot(
            day=14,
            cash=900_000,
            metrics={
                "price_a": 25,
                "price_b": 69,
                "price_c": 179,
                "known_segments": "S1,E1",
                "lead_promotion_monthly": 0,
            },
        ),
    )

    assert len(batch.plans) == len(payload["candidates"])
    assert len(batch.rejections) == 1
    rejection = batch.rejections[0]
    assert rejection.veto_codes == ("acquisition_probe_missing",)
    assert rejection.stage == "construction"
    assert rejection.candidate_index == 1
    assert rejection.week == 2
    assert "acquisition probe spend" in rejection.detail


@pytest.mark.asyncio
async def test_openrouter_accounts_for_tokens_cost_and_schema_retries() -> None:
    responses = [
        httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps({"unexpected": True})}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.0004},
            },
        ),
        httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(valid_output())}}
                ],
                "usage": {"prompt_tokens": 150, "completion_tokens": 40, "cost": 0.0006},
            },
        ),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["usage"] == {"include": True}
        return responses.pop(0)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenRouterProvider(api_key="test-key", client=client)
        await provider.generate_structured(
            system_prompt="system",
            user_prompt="input",
            output_schema=ExecutiveDecisionOutput,
        )

    usage = provider.usage_snapshot()
    assert usage["logical_calls"] == 1
    assert usage["http_calls"] == 2
    assert usage["validation_retries"] == 1
    assert usage["prompt_tokens"] == 250
    assert usage["completion_tokens"] == 60
    assert usage["cost_usd"] == pytest.approx(0.001)


@pytest.mark.asyncio
async def test_usage_accounting_tolerates_providers_that_report_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(valid_output())}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenRouterProvider(api_key="test-key", client=client)
        await provider.generate_structured(
            system_prompt="system",
            user_prompt="input",
            output_schema=ExecutiveDecisionOutput,
        )

    usage = provider.usage_snapshot()
    assert usage["http_calls"] == 1
    assert usage["prompt_tokens"] == 0
    assert usage["cost_usd"] == 0.0


def test_gemini_sums_token_usage_across_events() -> None:
    class Metadata:
        prompt_token_count = 120
        candidates_token_count = 30

    provider = GeminiAdkProvider(api_key="test-key")
    provider._accumulate_usage(Metadata())
    provider._accumulate_usage(Metadata())
    provider._accumulate_usage(None)

    usage = provider.usage_snapshot()
    assert usage["prompt_tokens"] == 240
    assert usage["completion_tokens"] == 60
