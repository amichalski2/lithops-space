from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest
from lithops.application.weekly_planning import sandbox_action_payload
from lithops.domain.executable_model import (
    CompanyModelFitRequest,
    CompanyModelPredictRequest,
    ModelArtifact,
    ModelArtifactAssertion,
    ModelArtifactTestCase,
    ModelEntrypoint,
    ModelRuntimeKind,
)
from lithops.domain.models import ObservationSnapshot
from lithops.model_runtime import (
    FixedBaselineModel,
    SandboxedCompanyModel,
    SandboxedPythonRunner,
    SandboxExecutionError,
    SandboxPolicy,
    SandboxPolicyError,
    SandboxTimeoutError,
)
from lithops.simulator import (
    SimulationAction,
    SimulationState,
    TargetedAdAllocation,
    TargetedDevelopmentAllocation,
)
from lithops.world_model import bootstrap_world_model

RUN_ID = UUID("77777777-7777-7777-7777-777777777777")

VALID_SOURCE = """
def fit(history, prior, seed):
    return {
        "weekly_cash_delta": prior["weekly_cash_delta"],
        "history_count": len(history),
    }

def predict(fitted, state, action, horizons_days, n_samples, seed):
    samples = []
    for rollout_index in range(n_samples):
        for horizon_days in horizons_days:
            weeks = horizon_days / 7
            samples.append({
                "rollout_index": rollout_index,
                "horizon_days": horizon_days,
                "cash": state["cash"] + fitted["weekly_cash_delta"] * weeks,
                "revenue_weekly": state["revenue_weekly"],
                "customers": state["customers"],
                "churn_rate": state["churn_rate"],
                "accounting": {
                    "period_days": horizon_days,
                    "starting_cash": state["cash"],
                    "recognized_revenue": state["revenue_weekly"] * weeks,
                    "operating_cost": 0,
                    "marketing_spend": 0,
                    "development_spend": 0,
                    "other_outflows": (
                        state["revenue_weekly"] - fitted["weekly_cash_delta"]
                    ) * weeks,
                    "ending_cash": (
                        state["cash"] + fitted["weekly_cash_delta"] * weeks
                    ),
                },
            })
    return {"samples": samples}

def diagnostics(fitted):
    return {
        "history_count": fitted["history_count"],
        "weekly_cash_delta": fitted["weekly_cash_delta"],
    }
""".strip()


def generated_artifact(source_code: str = VALID_SOURCE) -> ModelArtifact:
    return ModelArtifact.create(
        name="generated-cash-trend-v1",
        runtime_kind=ModelRuntimeKind.SANDBOXED_PYTHON,
        scope="cash",
        hypothesis="A fitted weekly cash delta predicts near-term cash.",
        authoring_agent="cash_model_coding_agent",
        provider="openrouter",
        model_name="qwen/qwen3-32b",
        prompt_version="cash-coding-v1",
        source_code=source_code,
        tests=(
            ModelArtifactTestCase(
                name="fit_uses_declared_prior",
                entrypoint=ModelEntrypoint.FIT,
                arguments={
                    "history": [{"day": 0, "cash": 1000}],
                    "prior": {"weekly_cash_delta": -25},
                    "seed": 1,
                },
                assertions=(
                    ModelArtifactAssertion(
                        path="weekly_cash_delta",
                        operator="equals",
                        expected=-25,
                    ),
                    ModelArtifactAssertion(
                        path="history_count",
                        operator="equals",
                        expected=1,
                    ),
                ),
            ),
        ),
    )


def fit_request(*, prior=None) -> CompanyModelFitRequest:
    return CompanyModelFitRequest(
        observation_ids=("observation:run:0", "observation:run:7"),
        training_start_day=0,
        training_end_day=7,
        history=({"day": 0, "cash": 1000}, {"day": 7, "cash": 900}),
        prior=prior or {"weekly_cash_delta": -100},
        seed=5,
    )


def prediction_state() -> dict:
    return {
        "week": 0,
        "cash": 1_000.0,
        "revenue_weekly": 100.0,
        "customers": 10.0,
        "churn_rate": 0.05,
        "price_per_customer_weekly": 10.0,
        "weekly_acquisition": 1.0,
        "marketing_spend": 0.0,
        "development_spend": 0.0,
        "product_quality": 0.5,
        "capacity": 100.0,
        "reputation": 0.5,
    }


def prediction_action() -> dict:
    return {
        "name": "hold",
        "price_per_customer_weekly": 10.0,
        "marketing_spend": 0.0,
        "development_spend": 0.0,
        "segment_focus": 1.0,
    }


def test_fixed_baseline_adapts_derived_channel_exposures_from_temporal_actions() -> None:
    model = FixedBaselineModel()
    baseline_world_model = bootstrap_world_model(
        RUN_ID,
        ObservationSnapshot(
            day=0,
            cash=500_000,
            observed_at=datetime(2026, 8, 25, tzinfo=UTC),
        ),
    )
    fitted = model.fit(
        fit_request(
            prior={
                "legacy_world_model": baseline_world_model.model_dump(mode="json"),
            }
        )
    )
    state = SimulationState.model_validate(prediction_state())
    simulation_action = SimulationAction.model_validate(
        {
            **prediction_action(),
            "marketing_spend": 100.0,
            "targeted_ad_allocations": (
                TargetedAdAllocation(
                    channel="social_media",
                    segment="S1",
                    daily_spend=40.0 / 7.0,
                ),
                TargetedAdAllocation(
                    channel="search_ads",
                    segment="S2",
                    daily_spend=60.0 / 7.0,
                ),
            ),
        }
    )
    action = sandbox_action_payload(simulation_action, state, horizon_weeks=1)

    result = model.predict(
        CompanyModelPredictRequest(
            fitted_model=fitted,
            state=state.model_dump(mode="json"),
            action=action,
            horizons_days=(7,),
            n_rollouts=3,
            seed=8,
        )
    )

    assert len(result.samples) == 3


def test_sandboxed_model_fits_predicts_and_runs_artifact_tests() -> None:
    artifact = generated_artifact()
    runner = SandboxedPythonRunner(SandboxPolicy(timeout_seconds=3))
    model = SandboxedCompanyModel(artifact, runner)

    test_results = runner.run_artifact_tests(artifact)
    fitted = model.fit(fit_request())
    prediction = model.predict(
        CompanyModelPredictRequest(
            fitted_model=fitted,
            state=prediction_state(),
            action=prediction_action(),
            horizons_days=(7, 28),
            n_rollouts=3,
            seed=8,
        )
    )

    assert all(result.passed for result in test_results)
    assert len(prediction.samples) == 6
    assert prediction.samples[0].cash == pytest.approx(900)
    assert prediction.samples[1].cash == pytest.approx(600)
    assert model.diagnostics(fitted) == {
        "history_count": 2,
        "weekly_cash_delta": -100,
    }


NOISY_SOURCE = """
def fit(history, prior, seed):
    return {"weekly_cash_delta": prior["weekly_cash_delta"]}

def predict(fitted, state, action, horizons_days, n_samples, seed):
    samples = []
    for rollout_index in range(n_samples):
        for horizon_days in horizons_days:
            weeks = horizon_days / 7
            shock = normal01(seed, rollout_index) * 10.0
            drift = fitted["weekly_cash_delta"] * weeks
            ending_cash = state["cash"] + drift + shock
            samples.append({
                "rollout_index": rollout_index,
                "horizon_days": horizon_days,
                "cash": ending_cash,
                "revenue_weekly": state["revenue_weekly"],
                "customers": state["customers"],
                "churn_rate": state["churn_rate"],
                "accounting": {
                    "period_days": horizon_days,
                    "starting_cash": state["cash"],
                    "recognized_revenue": state["revenue_weekly"] * weeks,
                    "operating_cost": 0,
                    "marketing_spend": 0,
                    "development_spend": 0,
                    "other_outflows": state["revenue_weekly"] * weeks - drift - shock,
                    "ending_cash": ending_cash,
                },
            })
    return {"samples": samples}

def diagnostics(fitted):
    return {"uniform": uniform01(1, 1)}
""".strip()


def test_sandbox_rng_helpers_are_deterministic_and_produce_spread() -> None:
    artifact = generated_artifact(NOISY_SOURCE)
    runner = SandboxedPythonRunner(SandboxPolicy(timeout_seconds=5))
    model = SandboxedCompanyModel(artifact, runner)
    fitted = model.fit(fit_request())
    request = CompanyModelPredictRequest(
        fitted_model=fitted,
        state=prediction_state(),
        action=prediction_action(),
        horizons_days=(28,),
        n_rollouts=5,
        seed=11,
    )

    first = model.predict(request)
    second = model.predict(request)
    cash = [sample.cash for sample in first.samples]

    assert [sample.cash for sample in second.samples] == cash, (
        "the same seed must replay the same draws"
    )
    assert len(set(cash)) == 5, "each rollout must receive its own draw"
    assert model.diagnostics(fitted)["uniform"] == pytest.approx(
        uniform01_reference(1, 1)
    )


def uniform01_reference(seed: int, index: int) -> float:
    mask = 0xFFFFFFFFFFFFFFFF
    value = (seed * 0x9E3779B97F4A7C15 + index * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    value = value ^ (value >> 31)
    return (value >> 11) / 9007199254740992.0


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("import os\n    return {}", "Import"),
        ("return open('secret.txt').read()", "open"),
        ("return {'type': ().__class__.__name__}", "dunder"),
    ],
)
def test_sandbox_rejects_import_file_access_and_object_graph_escape(
    body: str,
    expected: str,
) -> None:
    source = (
        f"def fit(history, prior, seed):\n    {body}\n\n"
        "def predict(fitted, state, action, horizons_days, n_samples, seed):\n"
        "    return {'samples': []}\n\n"
        "def diagnostics(fitted):\n    return {}\n"
    )

    with pytest.raises(SandboxPolicyError, match=expected):
        SandboxedPythonRunner().validate(generated_artifact(source))


def test_sandbox_terminates_non_terminating_candidate() -> None:
    source = (
        "def fit(history, prior, seed):\n"
        "    while True:\n"
        "        seed = seed + 1\n\n"
        "def predict(fitted, state, action, horizons_days, n_samples, seed):\n"
        "    return {'samples': []}\n\n"
        "def diagnostics(fitted):\n    return {}\n"
    )
    model = SandboxedCompanyModel(
        generated_artifact(source),
        SandboxedPythonRunner(SandboxPolicy(timeout_seconds=0.2)),
    )

    with pytest.raises(SandboxTimeoutError, match="timeout"):
        model.fit(fit_request())


def test_sandbox_rejects_candidate_that_omits_costs_from_cash_bridge() -> None:
    source = VALID_SOURCE.replace(
        '"other_outflows": (\n'
        '                        state["revenue_weekly"] - fitted["weekly_cash_delta"]\n'
        "                    ) * weeks,",
        '"other_outflows": 0,',
    )
    assert source != VALID_SOURCE
    model = SandboxedCompanyModel(generated_artifact(source))
    fitted = model.fit(fit_request())

    with pytest.raises(SandboxExecutionError, match="accounting_mismatch"):
        model.predict(
            CompanyModelPredictRequest(
                fitted_model=fitted,
                state=prediction_state(),
                action=prediction_action(),
                horizons_days=(7,),
                n_rollouts=1,
                seed=8,
            )
        )


def test_existing_simulator_is_a_reproducible_registered_baseline() -> None:
    baseline = FixedBaselineModel()
    world_model = bootstrap_world_model(
        RUN_ID,
        ObservationSnapshot(
            day=0,
            cash=500_000,
            observed_at=datetime(2026, 8, 25, tzinfo=UTC),
        ),
    )
    request = fit_request(prior={"legacy_world_model": world_model.model_dump(mode="json")})
    fitted = baseline.fit(request)
    state = SimulationState.model_validate(prediction_state())
    action = SimulationAction.model_validate(prediction_action())
    prediction_request = CompanyModelPredictRequest(
        fitted_model=fitted,
        state=state.model_dump(mode="json"),
        action=action.model_dump(mode="json"),
        horizons_days=(7, 28),
        n_rollouts=5,
        seed=21,
    )

    first = baseline.predict(prediction_request)
    second = baseline.predict(prediction_request)

    assert baseline.artifact.name == "fixed-baseline-v12"
    assert baseline.artifact.runtime_kind == ModelRuntimeKind.TRUSTED_BASELINE
    assert first == second
    assert len(first.samples) == 10
    assert baseline.diagnostics(fitted)["baseline"] is True
    terminal_cash = {
        sample.cash for sample in first.samples if sample.horizon_days == 28
    }
    assert len(terminal_cash) > 1, (
        "the baseline must expose process noise so forecast intervals never collapse"
    )
    assert baseline.diagnostics(fitted)["cash_flow_residual_sigma_weekly"] >= 0


def test_baseline_accepts_shared_experiment_duration_fields() -> None:
    baseline = FixedBaselineModel()
    world_model = bootstrap_world_model(
        RUN_ID,
        ObservationSnapshot(
            day=0,
            cash=500_000,
            observed_at=datetime(2026, 8, 25, tzinfo=UTC),
        ),
    )
    fitted = baseline.fit(
        fit_request(prior={"legacy_world_model": world_model.model_dump(mode="json")})
    )
    state = SimulationState.model_validate(prediction_state())
    action = {
        **SimulationAction.model_validate(prediction_action()).model_dump(mode="json"),
        "lead_promotion_monthly": 5.0,
        "lead_promotion_after_experiment": 0.0,
        "experiment_duration_weeks": 1,
        "lead_promotion_duration_weeks": 1,
    }

    result = baseline.predict(
        CompanyModelPredictRequest(
            fitted_model=fitted,
            state=state.model_dump(mode="json"),
            action=action,
            horizons_days=(7,),
            n_rollouts=2,
            seed=55,
        )
    )

    assert len(result.samples) == 2


def test_baseline_protocol_charges_targeted_development_cash_leg() -> None:
    baseline = FixedBaselineModel()
    world_model = bootstrap_world_model(
        RUN_ID,
        ObservationSnapshot(
            day=0,
            cash=500_000,
            observed_at=datetime(2026, 8, 25, tzinfo=UTC),
        ),
    )
    fitted = baseline.fit(
        fit_request(prior={"legacy_world_model": world_model.model_dump(mode="json")})
    )
    state = SimulationState.model_validate(prediction_state()).model_copy(update={"week": 4})
    base = SimulationAction.model_validate(prediction_action())
    targeted = base.model_copy(
        update={
            "targeted_development_allocations": (
                TargetedDevelopmentAllocation(segment="S1", daily_spend=500),
            ),
            "targeted_development_spend_until_week": 5,
            "targeted_development_spend_after_experiment": 0,
        }
    )

    def cash(action: SimulationAction) -> float:
        result = baseline.predict(
            CompanyModelPredictRequest(
                fitted_model=fitted,
                state=state.model_dump(mode="json"),
                action=sandbox_action_payload(action, state, horizon_weeks=1),
                horizons_days=(7,),
                n_rollouts=1,
                seed=91,
            )
        )
        return result.samples[0].cash

    assert cash(base) - cash(targeted) == pytest.approx(3_500)
