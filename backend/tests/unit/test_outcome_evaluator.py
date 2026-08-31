from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from lithops.domain.evaluation import (
    ModelHealthStatus,
    ParameterResidualAttribution,
)
from lithops.domain.models import CashForecast, CashForecasts, ObservationSnapshot
from lithops.domain.world_model import EvidenceKind, WorldModelParameterName
from lithops.evaluation.model_health import evaluate_model_health
from lithops.evaluation.prediction_ledger import (
    create_cash_prediction,
    mature_cash_predictions,
)
from lithops.world_model import bootstrap_world_model, recalibrate_world_model

RUN_ID = UUID("77777777-7777-7777-7777-777777777777")
STARTED_AT = datetime(2026, 8, 25, tzinfo=UTC)


def model():
    return bootstrap_world_model(
        RUN_ID,
        ObservationSnapshot(day=0, cash=100, observed_at=STARTED_AT),
    )


def entry(
    world_model_id: UUID,
    *,
    issued_day: int = 0,
    point: float = 100,
):
    decision_id = uuid4()
    return create_cash_prediction(
        run_id=RUN_ID,
        decision_id=decision_id,
        decision_week=issued_day // 7,
        issued_day=issued_day,
        model_version_id=world_model_id,
        prompt_version="executive-v1",
        observation_reference=f"observation:{issued_day}",
        assumptions=("Local simulator sensitivity remains valid.",),
        evidence_references=(f"world-model:{world_model_id}",),
        uncertainty_source="world-model-parameter-sampling",
        confidence=0.5,
        forecasts=CashForecasts(
            items=[
                CashForecast(
                    horizon_days=horizon,
                    point=point,
                    lower=90,
                    upper=110,
                )
                for horizon in (7, 28, 84, 182)
            ]
        ),
        committed_at=STARTED_AT + timedelta(days=issued_day),
    )


def outcome(prediction, *, observed_day: int, cash: float):
    return mature_cash_predictions(
        (prediction,),
        ObservationSnapshot(
            day=observed_day,
            cash=cash,
            observed_at=STARTED_AT + timedelta(days=observed_day),
        ),
        observation_reference=f"observation:{observed_day}",
    )[0]


def marketing_parameter(world_model):
    return next(
        parameter
        for parameter in world_model.parameters
        if parameter.name is WorldModelParameterName.MARKETING_SATURATION
    )


def test_single_miss_lowers_confidence_without_triggering_rebuild() -> None:
    current = model()
    prediction = entry(current.id)
    missed = outcome(prediction, observed_day=7, cash=130)

    health = evaluate_model_health(
        model_version_id=current.id,
        entries=(prediction,),
        outcomes=(missed,),
    )
    updated = recalibrate_world_model(
        world_model=current,
        entries=(prediction,),
        outcomes=(missed,),
        attributions=(
            ParameterResidualAttribution(
                outcome_id=missed.id,
                parameter_name=WorldModelParameterName.MARKETING_SATURATION,
                cash_sensitivity_per_unit=100,
                evidence_reference="simulator-local-sensitivity:marketing-saturation",
            ),
        ),
    )

    before = marketing_parameter(current)
    after = marketing_parameter(updated)
    assert health.status is ModelHealthStatus.WATCHING
    assert health.rebuild_recommended is False
    assert after.estimate == pytest.approx(before.estimate + 0.06)
    assert after.confidence < before.confidence
    assert current.version == 1
    assert updated.version == 2
    assert updated.based_on_version_id == current.id
    assert updated.changes[0].parameter_name is WorldModelParameterName.MARKETING_SATURATION
    assert updated.changes[0].evidence[0].kind is EvidenceKind.PREDICTION_RESIDUAL

    corrected_point = 100 + 100 * (after.estimate - before.estimate)
    assert abs(130 - corrected_point) < abs(130 - 100)


def test_recalibration_is_bounded_and_replay_deterministic() -> None:
    current = model()
    prediction = entry(current.id)
    missed = outcome(prediction, observed_day=7, cash=1_000)
    attribution = ParameterResidualAttribution(
        outcome_id=missed.id,
        parameter_name=WorldModelParameterName.MARKETING_SATURATION,
        cash_sensitivity_per_unit=1,
        evidence_reference="simulator-local-sensitivity:marketing-saturation",
    )

    first = recalibrate_world_model(
        world_model=current,
        entries=(prediction,),
        outcomes=(missed,),
        attributions=(attribution,),
    )
    replay = recalibrate_world_model(
        world_model=current,
        entries=(prediction,),
        outcomes=(missed,),
        attributions=(attribution,),
    )

    before = marketing_parameter(current)
    after = marketing_parameter(first)
    assert first == replay
    assert after.estimate - before.estimate == pytest.approx(
        0.10 * (before.upper_bound - before.lower_bound)
    )
    assert after.lower_bound <= after.estimate <= after.upper_bound


def test_interval_hit_increases_derived_confidence() -> None:
    current = model()
    prediction = entry(current.id)
    inside = outcome(prediction, observed_day=7, cash=105)
    updated = recalibrate_world_model(
        world_model=current,
        entries=(prediction,),
        outcomes=(inside,),
        attributions=(
            ParameterResidualAttribution(
                outcome_id=inside.id,
                parameter_name=WorldModelParameterName.MARKETING_SATURATION,
                cash_sensitivity_per_unit=100,
                evidence_reference="simulator-local-sensitivity:marketing-saturation",
            ),
        ),
    )

    assert marketing_parameter(updated).confidence > marketing_parameter(current).confidence


def test_persistent_error_emits_rebuild_recommendation() -> None:
    current = model()
    entries = tuple(entry(current.id, issued_day=day) for day in (0, 7, 14))
    outcomes = tuple(
        outcome(prediction, observed_day=prediction.issued_day + 7, cash=cash)
        for prediction, cash in zip(entries, (130, 95, 140), strict=True)
    )

    health = evaluate_model_health(
        model_version_id=current.id,
        entries=entries,
        outcomes=outcomes,
    )

    assert health.status is ModelHealthStatus.DEGRADED
    assert health.rebuild_recommended is True
    assert "two_of_last_three_interval_misses" in health.trigger_codes
    assert "persistent_directional_bias" in health.trigger_codes
    assert health.interval_miss_count == 2


def test_persistent_zero_conversion_funnel_triggers_structural_diagnosis() -> None:
    current = model()
    prediction = entry(current.id)
    inside = outcome(prediction, observed_day=7, cash=100)

    health = evaluate_model_health(
        model_version_id=current.id,
        entries=(prediction,),
        outcomes=(inside,),
        observations=(
            ObservationSnapshot(
                day=7,
                cash=100,
                metrics={"weekly_leads": 45, "weekly_conversions": 0},
                observed_at=STARTED_AT + timedelta(days=7),
            ),
        ),
    )

    assert health.status is ModelHealthStatus.DEGRADED
    assert health.rebuild_recommended is True
    assert health.trigger_codes == ("persistent_zero_conversion_funnel",)


def test_zero_conversion_diagnosis_survives_one_early_conversion() -> None:
    """An early converted week must not silence the funnel diagnosis for good.

    The lifetime-total form of this trigger went dead after the first
    conversion; hundreds of later leads converting at zero then looked healthy.
    The diagnosis is windowed over recent observations instead.
    """

    current = model()
    prediction = entry(current.id)
    inside = outcome(prediction, observed_day=7, cash=100)

    observations = [
        ObservationSnapshot(
            day=7,
            cash=100,
            metrics={"weekly_leads": 10, "weekly_conversions": 1},
            observed_at=STARTED_AT + timedelta(days=7),
        )
    ]
    observations += [
        ObservationSnapshot(
            day=day,
            cash=100,
            metrics={"weekly_leads": 100, "weekly_conversions": 0},
            observed_at=STARTED_AT + timedelta(days=day),
        )
        for day in (14, 21, 28, 35)
    ]

    health = evaluate_model_health(
        model_version_id=current.id,
        entries=(prediction,),
        outcomes=(inside,),
        observations=tuple(observations),
    )

    assert "persistent_zero_conversion_funnel" in health.trigger_codes
    assert health.rebuild_recommended is True
