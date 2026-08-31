from __future__ import annotations

import pytest
from lithops.application.learning_demo import run_learning_demo
from lithops.benchmark.learning_scenario import LearningScenarioBenchmarkAdapter
from lithops.infrastructure.persistence.repositories import InMemoryRunRepository


async def _run_scenario() -> dict[str, object]:
    return await run_learning_demo(
        repository=InMemoryRunRepository(),
        benchmark=LearningScenarioBenchmarkAdapter(),
    )


@pytest.mark.asyncio
async def test_learning_scenario_is_reproducible_and_causally_explained() -> None:
    first = await _run_scenario()
    second = await _run_scenario()

    assert first == second
    assert first["run"]["prediction_before_action_verified"] is True
    assert first["run"]["replay_without_duplicate_action_or_advance"] is True
    prediction_miss = first["prediction_miss"]
    assert prediction_miss["interval_hit"] is False
    assert prediction_miss["signed_residual_actual_minus_prediction"] < 0
    assert first["model_health"] == {
        "evaluated_day": 28,
        "status": "degraded",
        "rebuild_recommended": True,
        "trigger_codes": [
            "two_of_last_three_interval_misses",
            "persistent_directional_bias",
        ],
    }
    parameter_update = first["parameter_update"]
    assert {
        key: value
        for key, value in parameter_update.items()
        if key not in {"old_confidence", "new_confidence"}
    } == {
        "name": "marketing_saturation",
        "old_model_version": 4,
        "new_model_version": 5,
        "old_estimate": 0.44,
        "new_estimate": 0.37,
        "method": "model_challenge_backtest_v1",
        "evidence_days": [7, 14, 21, 28],
    }
    assert 0 < parameter_update["old_confidence"] <= 1
    assert parameter_update["new_confidence"] == parameter_update["old_confidence"]
    strategy = first["causal_strategy_replay"]
    assert strategy["maximum_robust_utility_delta"] > 0
    assert strategy["only_marketing_saturation_updated_strategy"] == strategy[
        "full_new_model_strategy"
    ]
    assert strategy["full_new_model_strategy"] == strategy["persisted_strategy"]

    fleet = first["dynamic_fleet"]
    assert fleet["trigger_status"] == "completed"
    assert fleet["requested_builders"] == ["acquisition_builder", "pricing_builder"]
    assert fleet["resolution"] == "accepted"
    assert fleet["selected_builders"] == ["acquisition_builder"]
    assert fleet["rejected_builders"] == ["pricing_builder"]
    assert {item["builder_name"]: item["supported"] for item in fleet["hypotheses"]} == {
        "acquisition_builder": True,
        "pricing_builder": False,
    }
    assert all(item["status"] == "completed" for item in fleet["builder_calls"])
    assert fleet["permission_boundary"] == {
        "denied_tool": "set_prices",
        "reason_code": "tool_not_allowed_for_role",
        "input_hash_present": True,
        "raw_input_absent": True,
        "benchmark_calls_unchanged": True,
    }
