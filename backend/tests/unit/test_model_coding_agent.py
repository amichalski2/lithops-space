from __future__ import annotations

import json

import pytest
from lithops.agents.model_coding_agent import (
    ACQUISITION_MODEL_CODER,
    CAPACITY_MODEL_CODER,
    PRICING_MODEL_CODER,
    RETENTION_MODEL_CODER,
    ModelArtifactDraft,
    ModelCodingAgent,
)
from lithops.domain.model_challenge import HypothesisFamily
from lithops.model_runtime import (
    FixedBaselineModel,
    SandboxedCompanyModel,
    SandboxedPythonRunner,
    TemporalEvaluationPolicy,
    TemporalModelEvaluator,
    TemporalObservation,
)
from pydantic import ValidationError

from backend.tests.unit.test_hypothesis_backtest import challenge_package

SOURCE = """
def fit(history, prior, seed):
    acquisitions = [
        row.get("weekly_acquisition", 0.0)
        for row in history
        if row.get("weekly_acquisition", 0.0) >= 0.0
    ]
    return {
        "weekly_cash_delta": prior["weekly_cash_delta"],
        "marketing_cash_return": prior["marketing_cash_return"],
        "weekly_acquisition": sum(acquisitions) / len(acquisitions) if acquisitions else 0.0,
        "history_count": len(history),
    }

def predict(fitted, state, action, horizons_days, n_samples, seed):
    samples = []
    for rollout_index in range(n_samples):
        for horizon_days in horizons_days:
            weeks = horizon_days / 7
            policy_action_path = action.get("policy_action_path", [])
            duration = action.get("experiment_duration_weeks", weeks)
            marketing_delay = action.get("marketing_spend_start_after_weeks", 0.0)
            pre_marketing_weeks = min(weeks, marketing_delay)
            active_weeks = min(
                max(0.0, weeks - marketing_delay),
                max(0.0, duration - marketing_delay),
            )
            inactive_weeks = max(0.0, weeks - pre_marketing_weeks - active_weeks)
            promotion_duration = action.get("lead_promotion_duration_weeks", weeks)
            promotion_active_weeks = min(weeks, promotion_duration)
            promotion_inactive_weeks = max(0.0, weeks - promotion_active_weeks)
            current_price = state.get(
                "catalog_price_per_customer_weekly",
                state.get("price_per_customer_weekly", 1.0),
            )
            action_price = action.get("price_per_customer_weekly", current_price)
            price_ratio = action_price / max(current_price, 0.01)
            marketing_spend = (
                state.get("marketing_spend", 0.0) * pre_marketing_weeks
                + action.get("marketing_spend", 0.0) * active_weeks
                + action.get("marketing_spend_after_experiment", 0.0) * inactive_weeks
            )
            if policy_action_path:
                marketing_spend = sum(
                    step.get("marketing_spend", 0.0)
                    for step in policy_action_path[:int(weeks)]
                )
            development_spend = (
                action.get("development_spend", 0.0)
                * min(
                    weeks,
                    action.get("development_spend_duration_weeks", duration),
                )
                + action.get("development_spend_after_experiment", 0.0)
                * max(
                    0.0,
                    weeks
                    - min(
                        weeks,
                        action.get("development_spend_duration_weeks", duration),
                    ),
                )
            )
            targeted_duration = action.get("targeted_development_duration_weeks", weeks)
            targeted_active_weeks = min(weeks, targeted_duration)
            targeted_inactive_weeks = max(0.0, weeks - targeted_active_weeks)
            targeted_development_spend = (
                action.get("targeted_development_spend_weekly", 0.0) * targeted_active_weeks
                + action.get("targeted_development_spend_after_experiment", 0.0)
                * targeted_inactive_weeks
            )
            development_spend += targeted_development_spend
            if policy_action_path:
                development_spend = sum(
                    step.get("development_spend", 0.0)
                    + step.get("targeted_development_spend_weekly", 0.0)
                    for step in policy_action_path[:int(weeks)]
                )
            operations_spend = action.get("operations_spend", 0.0) * weeks
            weighted_promotion = (
                action.get("lead_promotion_monthly", 0.0) * promotion_active_weeks
                + action.get("lead_promotion_after_experiment", 0.0)
                * promotion_inactive_weeks
            ) / max(weeks, 1.0)
            entry_price = max(state.get("entry_price_monthly", 25.0), 0.01)
            promotion_ratio = min(1.0, weighted_promotion / entry_price)
            state_tier_total = max(
                1,
                state.get("model_tier_a", 1)
                + state.get("model_tier_b", 1)
                + state.get("model_tier_c", 1),
            )
            action_tier_total = (
                action.get("model_tier_a", state.get("model_tier_a", 1))
                + action.get("model_tier_b", state.get("model_tier_b", 1))
                + action.get("model_tier_c", state.get("model_tier_c", 1))
            )
            tier_cost_ratio = action_tier_total / state_tier_total
            incremental_cash = marketing_spend * fitted["marketing_cash_return"]
            recognized_revenue = state["revenue_weekly"] * price_ratio * weeks
            pricing_cash_delta = state["revenue_weekly"] * (price_ratio - 1.0) * weeks
            promotion_acquisition = (
                fitted["weekly_acquisition"] * weeks * promotion_ratio
            )
            promotion_cost = promotion_acquisition * weighted_promotion * 7.0 / 30.0
            shock = normal01(seed, rollout_index) * 5.0 * weeks
            operating_cost = (
                state.get("operating_cost_per_customer_weekly", 0.0)
                * state["customers"]
                * tier_cost_ratio
                * weeks
            )
            ending_cash = (
                state["cash"]
                + fitted["weekly_cash_delta"] * weeks
                + incremental_cash
                - marketing_spend
                - development_spend
                - operations_spend
                + pricing_cash_delta
                + shock
                - operating_cost
                - promotion_cost
            )
            price_churn = max(0.0, price_ratio - 1.0) * 0.1
            churn_rate = min(1.0, state["churn_rate"] + price_churn)
            customers = max(
                0.0,
                state["customers"] * (1.0 - churn_rate * weeks)
                + fitted["weekly_acquisition"] * weeks
                + promotion_acquisition,
            )
            samples.append({
                "rollout_index": rollout_index,
                "horizon_days": horizon_days,
                "cash": ending_cash,
                "revenue_weekly": state["revenue_weekly"] * price_ratio,
                "customers": customers,
                "churn_rate": churn_rate,
                "accounting": {
                    "period_days": horizon_days,
                    "starting_cash": state["cash"],
                    "recognized_revenue": recognized_revenue,
                    "other_inflows": incremental_cash,
                    "operating_cost": operating_cost,
                    "operations_spend": operations_spend,
                    "marketing_spend": marketing_spend,
                    "development_spend": development_spend,
                    "other_outflows": (
                        recognized_revenue
                        - fitted["weekly_cash_delta"] * weeks
                        - pricing_cash_delta
                        - shock
                        + promotion_cost
                    ),
                    "ending_cash": ending_cash,
                    "currency": "USD",
                },
            })
    return {"samples": samples}

def diagnostics(fitted):
    return {
        "history_count": fitted["history_count"],
        "marketing_cash_return": fitted["marketing_cash_return"],
    }
""".strip()


def valid_output() -> dict[str, object]:
    return {
        "name": "acquisition-cash-response-v1",
        "family": "acquisition_efficiency",
        "scope": "acquisition_cash_response",
        "hypothesis": "Marketing has a bounded incremental cash return after baseline burn.",
        "source_lines": SOURCE.splitlines(),
        "required_features_json": json.dumps(
            [
                {"name": "history.day", "unit": "day", "required": True},
                {
                    "name": "history.weekly_acquisition",
                    "unit": "customer/week",
                    "required": True,
                },
                {"name": "state.cash", "unit": "USD", "required": True},
                {
                    "name": "state.revenue_weekly",
                    "unit": "USD/week",
                    "required": True,
                },
                {"name": "state.customers", "unit": "customer", "required": True},
                {"name": "state.churn_rate", "unit": "ratio_0_1", "required": True},
                {
                    "name": "state.entry_price_monthly",
                    "unit": "USD/customer/month_30_day",
                    "required": True,
                },
                {
                    "name": "state.lead_promotion_monthly",
                    "unit": "USD/customer/month_30_day",
                    "required": True,
                },
                {
                    "name": "state.operating_cost_per_customer_weekly",
                    "unit": "USD/customer/week",
                    "required": True,
                },
                {"name": "state.model_tier_a", "unit": "tier_1_5", "required": True},
                {"name": "state.model_tier_b", "unit": "tier_1_5", "required": True},
                {"name": "state.model_tier_c", "unit": "tier_1_5", "required": True},
                {
                    "name": "action.price_per_customer_weekly",
                    "unit": "USD/customer/week",
                    "required": True,
                },
                {
                    "name": "action.marketing_spend",
                    "unit": "USD/week",
                    "required": True,
                },
                {
                    "name": "action.development_spend",
                    "unit": "USD/week",
                    "required": True,
                },
                {
                    "name": "action.targeted_development_spend_weekly",
                    "unit": "USD/week",
                    "required": True,
                },
                {
                    "name": "action.targeted_development_duration_weeks",
                    "unit": "week",
                    "required": True,
                },
                {
                    "name": "action.targeted_development_spend_after_experiment",
                    "unit": "USD/week",
                    "required": True,
                },
                {
                    "name": "action.marketing_spend_start_after_weeks",
                    "unit": "week",
                    "required": True,
                },
                {
                    "name": "action.operations_spend",
                    "unit": "USD/week",
                    "required": True,
                },
                {"name": "action.model_tier_a", "unit": "tier_1_5", "required": True},
                {"name": "action.model_tier_b", "unit": "tier_1_5", "required": True},
                {"name": "action.model_tier_c", "unit": "tier_1_5", "required": True},
                {
                    "name": "action.experiment_duration_weeks",
                    "unit": "week",
                    "required": True,
                },
                {
                    "name": "action.development_spend_duration_weeks",
                    "unit": "week",
                    "required": True,
                },
                {
                    "name": "action.marketing_spend_after_experiment",
                    "unit": "USD/week",
                    "required": True,
                },
                {
                    "name": "action.development_spend_after_experiment",
                    "unit": "USD/week",
                    "required": True,
                },
                {
                    "name": "action.lead_promotion_monthly",
                    "unit": "USD/customer/month_30_day",
                    "required": True,
                },
                {
                    "name": "action.lead_promotion_duration_weeks",
                    "unit": "week",
                    "required": True,
                },
                {
                    "name": "action.lead_promotion_after_experiment",
                    "unit": "USD/customer/month_30_day",
                    "required": True,
                },
                {
                    "name": "action.policy_action_path",
                    "unit": "weekly_action_sequence",
                    "required": True,
                },
            ]
        ),
        "required_priors": ["weekly_cash_delta", "marketing_cash_return"],
        "tests_json": json.dumps(
            [
                {
                    "name": "fit_preserves_priors",
                    "entrypoint": "fit",
                    "arguments": {
                        "history": [
                            {"day": 0, "cash": 1000, "weekly_acquisition": 2.0}
                        ],
                        "prior": {
                            "weekly_cash_delta": -100,
                            "marketing_cash_return": 0.5,
                        },
                        "seed": 1,
                    },
                    "assertions": [
                        {
                            "path": "weekly_cash_delta",
                            "operator": "equals",
                            "expected": -100,
                        }
                    ],
                },
                {
                    "name": "predict_reconciles_week",
                    "entrypoint": "predict",
                    "arguments": {
                        "fitted": {
                            "weekly_cash_delta": -100,
                            "marketing_cash_return": 0.5,
                            "weekly_acquisition": 2.0,
                            "history_count": 1,
                        },
                        "state": {
                            "cash": 1000,
                            "revenue_weekly": 100,
                            "customers": 10,
                            "churn_rate": 0.05,
                            "price_per_customer_weekly": 10.0,
                            "catalog_price_per_customer_weekly": 10.0,
                            "entry_price_monthly": 25.0,
                            "lead_promotion_monthly": 0.0,
                            "operating_cost_per_customer_weekly": 1.0,
                            "model_tier_a": 1,
                            "model_tier_b": 1,
                            "model_tier_c": 1,
                        },
                        "action": {
                            "price_per_customer_weekly": 10.0,
                            "marketing_spend": 0.0,
                            "development_spend": 0.0,
                            "operations_spend": 0.0,
                            "model_tier_a": 1,
                            "model_tier_b": 1,
                            "model_tier_c": 1,
                            "experiment_duration_weeks": 4.0,
                            "marketing_spend_after_experiment": 0.0,
                            "development_spend_after_experiment": 0.0,
                            "lead_promotion_monthly": 0.0,
                            "lead_promotion_duration_weeks": 4.0,
                            "lead_promotion_after_experiment": 0.0,
                        },
                        "horizons_days": [7],
                        "n_samples": 1,
                        "seed": 1,
                    },
                    "assertions": [
                        {
                            "path": "samples.0.cash",
                            "operator": "approx",
                            "expected": 900,
                            "tolerance": 30,
                        },
                        {
                            "path": "samples.0.accounting.ending_cash",
                            "operator": "approx",
                            "expected": 900,
                            "tolerance": 30,
                        },
                    ],
                },
            ]
        ),
        "limitations": [
            "Assumes the fitted baseline weekly cash delta remains locally stable.",
            "Does not infer causal lift beyond the supplied marketing prior.",
        ],
    }


class CapturingCodingProvider:
    model_id = "qwen/test-coder"

    def __init__(self, output: dict[str, object] | None = None) -> None:
        self.output = output or valid_output()
        self.payload: dict[str, object] | None = None

    async def generate_structured(self, *, system_prompt, user_prompt, output_schema):
        assert "no imports" in system_prompt
        assert output_schema is ModelArtifactDraft
        self.payload = json.loads(user_prompt)
        return output_schema.model_validate(self.output)


def mutate_json_list(output, key, mutation) -> None:
    values = json.loads(output[key])
    mutation(values)
    output[key] = json.dumps(values)


def test_capacity_coder_is_scoped_to_historical_cost_and_capacity_evidence() -> None:
    allowed = set(CAPACITY_MODEL_CODER.allowed_features)

    assert CAPACITY_MODEL_CODER.family is HypothesisFamily.CAPACITY_PRESSURE
    assert CAPACITY_MODEL_CODER.prompt_version == "executable-model-coder-v12"
    assert {
        "history.operations_spend",
        "history.capacity_spend_weekly",
        "history.capacity",
        "history.operating_cost_per_customer_weekly",
        "state.cash",
        "action.operations_spend",
    } <= allowed
    assert {name for name, _, _ in CAPACITY_MODEL_CODER.available_priors} >= {
        "weekly_cash_delta",
        "price_elasticity",
        "marketing_cash_return",
        "churn_sensitivity",
    }


def test_zero_conversion_routes_competing_cross_subsystem_hypotheses() -> None:
    package = challenge_package()
    package = package.model_copy(
        update={
            "health_signal": package.health_signal.model_copy(
                update={"trigger_codes": ("persistent_zero_conversion_funnel",)}
            )
        }
    )

    def supports(spec) -> bool:
        return ModelCodingAgent(
            spec=spec,
            provider=CapturingCodingProvider(),
            provider_name="test",
        ).supports(package)

    assert supports(ACQUISITION_MODEL_CODER)
    assert supports(PRICING_MODEL_CODER)
    assert supports(RETENTION_MODEL_CODER)
    assert not supports(CAPACITY_MODEL_CODER)


class SequencedCodingProvider(CapturingCodingProvider):
    def __init__(self, outputs: list[dict[str, object]]) -> None:
        super().__init__(outputs[-1])
        self.outputs = outputs
        self.payloads: list[dict[str, object]] = []

    async def generate_structured(self, *, system_prompt, user_prompt, output_schema):
        assert "no imports" in system_prompt
        self.payloads.append(json.loads(user_prompt))
        output = self.outputs[min(len(self.payloads) - 1, len(self.outputs) - 1)]
        return output_schema.model_validate(output)


@pytest.mark.asyncio
async def test_scoped_coding_agent_authors_a_sandboxed_evaluable_artifact() -> None:
    package = challenge_package()
    parent = FixedBaselineModel().artifact
    provider = CapturingCodingProvider()
    agent = ModelCodingAgent(
        spec=ACQUISITION_MODEL_CODER,
        provider=provider,
        provider_name="openrouter",
    )

    artifact = await agent.author(package=package, parent_artifact=parent)
    runner = SandboxedPythonRunner()
    model = SandboxedCompanyModel(artifact, runner)
    test_results = runner.run_artifact_tests(artifact)
    observations = tuple(
        TemporalObservation(
            observation_id=f"obs-{day}",
            day=day,
            state={
                "cash": cash,
                "revenue_weekly": 100.0,
                "customers": 10.0,
                "churn_rate": 0.05,
            },
            action_from_previous={"marketing_spend": 0.0},
        )
        for day, cash in ((0, 1000.0), (7, 900.0), (14, 800.0))
    )
    result = TemporalModelEvaluator(
        TemporalEvaluationPolicy(n_rollouts=2, runtime_budget_ms=10_000)
    ).evaluate(
        run_id=package.run_id,
        challenge_id=package.challenge_id,
        runtime=model,
        observations=observations,
        prior={"weekly_cash_delta": -100.0, "marketing_cash_return": 0.5},
        seed=4,
    )

    assert provider.payload is not None
    assert provider.payload["assigned_family"] == "acquisition_efficiency"
    assert "allowed_features_and_units" in provider.payload
    assert artifact.parent_artifact_id == parent.id
    assert artifact.provider == "openrouter"
    assert artifact.model_name == provider.model_id
    assert artifact.dependencies == ()
    assert artifact.limitations
    assert all(item.passed for item in test_results)
    assert result.passed
    assert len(result.folds) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda output: mutate_json_list(
                output,
                    "required_features_json",
                    lambda values: values.append(
                        {"name": "action.segment_focus", "unit": "ratio", "required": True}
                    ),
            ),
            "outside its scope",
        ),
        (
            lambda output: mutate_json_list(
                output,
                "required_features_json",
                lambda values: next(
                    value
                    for value in values
                    if value["name"] == "action.marketing_spend"
                ).update({"unit": "USD/month"}),
            ),
            "incorrect units",
        ),
        (
            lambda output: mutate_json_list(
                output,
                "required_features_json",
                lambda values: values.__setitem__(
                    slice(None),
                    [
                        value
                        for value in values
                        if value["name"] != "history.weekly_acquisition"
                    ],
                ),
            ),
            "family-specific feature",
        ),
        (
            lambda output: mutate_json_list(
                output,
                "required_features_json",
                lambda values: values.__setitem__(
                    slice(None),
                    [
                        value
                        for value in values
                        if value["name"] != "action.price_per_customer_weekly"
                    ],
                ),
            ),
            "omitted planning action features",
        ),
        (
            lambda output: output.update({"family": "pricing_response"}),
            "expected acquisition_efficiency",
        ),
        (
            lambda output: output["required_priors"].append("secret_prior"),
            "priors outside its scope",
        ),
    ],
)
async def test_coding_agent_rejects_out_of_scope_features_and_units(mutation, error) -> None:
    output = valid_output()
    mutation(output)
    agent = ModelCodingAgent(
        spec=ACQUISITION_MODEL_CODER,
        provider=CapturingCodingProvider(output),
        provider_name="openrouter",
    )

    with pytest.raises(ValueError, match=error):
        await agent.author(
            package=challenge_package(),
            parent_artifact=FixedBaselineModel().artifact,
        )


def test_model_draft_rejects_company_actions_unknown_fields_and_missing_tests() -> None:
    action = valid_output()
    action["execute_action"] = {"tool": "set_prices"}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ModelArtifactDraft.model_validate(action)

    missing_predict_test = valid_output()
    mutate_json_list(
        missing_predict_test,
        "tests_json",
        lambda values: values.__delitem__(slice(1, None)),
    )
    with pytest.raises(ValidationError, match="both fit and predict tests"):
        ModelArtifactDraft.model_validate(missing_predict_test)

    object_expected = valid_output()
    mutate_json_list(
        object_expected,
        "tests_json",
        lambda values: values[0]["assertions"][0].update({"expected": {"value": -100}}),
    )
    with pytest.raises(ValidationError, match="tests_json"):
        ModelArtifactDraft.model_validate(object_expected)

    prefixed_path = valid_output()
    mutate_json_list(
        prefixed_path,
        "tests_json",
        lambda values: values[0]["assertions"][0].update({"path": "result.weekly_cash_delta"}),
    )
    with pytest.raises(ValidationError, match="tests_json"):
        ModelArtifactDraft.model_validate(prefixed_path)


@pytest.mark.asyncio
async def test_coding_agent_returns_executable_test_feedback_for_revision() -> None:
    failing = valid_output()
    mutate_json_list(
        failing,
        "tests_json",
        lambda values: values[0]["assertions"][0].update({"expected": -999}),
    )
    provider = SequencedCodingProvider([failing, valid_output()])
    agent = ModelCodingAgent(
        spec=ACQUISITION_MODEL_CODER,
        provider=provider,
        provider_name="openrouter",
        max_attempts=2,
    )

    artifact = await agent.author(
        package=challenge_package(),
        parent_artifact=FixedBaselineModel().artifact,
    )

    assert artifact.tests[0].assertions[0].expected == -100
    assert len(provider.payloads) == 2
    assert (
        "artifact executable tests failed" in provider.payloads[1]["semantic_validation_feedback"]
    )


@pytest.mark.asyncio
async def test_coding_agent_rejects_declared_but_causally_ignored_price_action() -> None:
    output = valid_output()
    output["source_lines"] = [
        (
            "            price_ratio = action_price * 0.0 + 1.0"
            if line.strip() == "price_ratio = action_price / max(current_price, 0.01)"
            else line
        )
        for line in output["source_lines"]
    ]
    agent = ModelCodingAgent(
        spec=ACQUISITION_MODEL_CODER,
        provider=CapturingCodingProvider(output),
        provider_name="openrouter",
        max_attempts=1,
    )

    with pytest.raises(
        ValueError,
        match=r"action\.price_per_customer_weekly:no_effect",
    ):
        await agent.author(
            package=challenge_package(),
            parent_artifact=FixedBaselineModel().artifact,
        )
