from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid5

import pytest
from lithops.domain.models import CashForecast, CashForecasts, ObservationSnapshot
from lithops.evaluation.model_health import evaluate_model_health
from lithops.evaluation.prediction_ledger import (
    create_cash_prediction,
    mature_cash_predictions,
)
from lithops.world_model import assemble_model_challenge_package, bootstrap_world_model

STARTED_AT = datetime(2026, 8, 25, 9, 0, tzinfo=UTC)
RUN_ID = UUID("11111111-1111-4111-8111-111111111111")


def history():
    observations = tuple(
        ObservationSnapshot(
            day=day,
            cash=cash,
            metrics={
                "weekly_acquisition": acquisition,
                "marketing_spend": 10_000,
            },
            observed_at=STARTED_AT + timedelta(days=day),
        )
        for day, cash, acquisition in (
            (0, 1_000_000, 80),
            (7, 800_000, 45),
            (14, 650_000, 30),
            (21, 500_000, 20),
        )
    )
    model = bootstrap_world_model(RUN_ID, observations[0])
    entries = []
    outcomes = []
    for index, issued_day in enumerate((0, 7, 14), start=1):
        entry = create_cash_prediction(
            run_id=RUN_ID,
            decision_id=UUID(int=index),
            decision_week=issued_day // 7,
            issued_day=issued_day,
            model_version_id=model.id,
            prompt_version="static-executive-v1",
            observation_reference=f"observation:{RUN_ID}:{issued_day}",
            assumptions=("Current acquisition efficiency persists.",),
            evidence_references=(f"world-model:{model.id}",),
            uncertainty_source="test-world-model-rollouts",
            confidence=0.4,
            forecasts=CashForecasts(
                items=[
                    CashForecast(
                        horizon_days=horizon,
                        point=1_000_000,
                        lower=900_000,
                        upper=1_100_000,
                    )
                    for horizon in (7, 28, 84, 182)
                ]
            ),
            committed_at=STARTED_AT + timedelta(days=issued_day),
        )
        entries.append(entry)
        outcomes.extend(
            mature_cash_predictions(
                (entry,),
                observations[index],
                observation_reference=f"observation:{RUN_ID}:{observations[index].day}",
            )
        )
    health = evaluate_model_health(
        model_version_id=model.id,
        entries=tuple(entries),
        outcomes=tuple(outcomes),
    )
    return observations, model, tuple(entries), tuple(outcomes), health


def test_package_is_canonical_across_input_order_and_retry() -> None:
    observations, model, entries, outcomes, health = history()

    first = assemble_model_challenge_package(
        health_signal=health,
        active_model=model,
        observations=observations,
        predictions=entries,
        outcomes=outcomes,
    )
    replay = assemble_model_challenge_package(
        health_signal=health,
        active_model=model,
        observations=reversed(observations),
        predictions=reversed(entries),
        outcomes=reversed(outcomes),
    )

    assert first == replay
    assert first.challenge_id == uuid5(health.id, "model-challenge:1.0")
    assert [item.day for item in first.observations] == [0, 7, 14, 21]
    assert [item.observed_day for item in first.residuals] == [7, 14, 21]
    assert first.created_at == health.evaluated_at


def test_package_normalizes_external_metric_case_for_historical_observations() -> None:
    observations, model, entries, outcomes, health = history()
    historical = observations[0].model_copy(
        update={"metrics": {"price_A": 25, "Weekly-Revenue": 1000}}
    )

    package = assemble_model_challenge_package(
        health_signal=health,
        active_model=model,
        observations=(historical, *observations[1:]),
        predictions=entries,
        outcomes=outcomes,
    )

    names = {metric.name for metric in package.observations[0].metrics}
    assert names == {"price_a", "weekly_revenue"}


def test_package_accepts_sparse_history_when_trigger_day_is_present() -> None:
    observations, model, entries, outcomes, health = history()

    package = assemble_model_challenge_package(
        health_signal=health,
        active_model=model,
        observations=(observations[-1],),
        predictions=entries,
        outcomes=outcomes,
    )

    assert [item.day for item in package.observations] == [21]
    assert len(package.residuals) == 3


def test_package_rejects_a_single_non_triggering_miss() -> None:
    observations, model, entries, outcomes, _ = history()
    watching = evaluate_model_health(
        model_version_id=model.id,
        entries=entries,
        outcomes=(outcomes[0],),
    )

    with pytest.raises(ValueError, match="persistent degraded-model trigger"):
        assemble_model_challenge_package(
            health_signal=watching,
            active_model=model,
            observations=observations[:2],
            predictions=entries,
            outcomes=(outcomes[0],),
        )


def test_package_rejects_missing_or_unresolvable_trigger_evidence() -> None:
    observations, model, entries, outcomes, health = history()

    with pytest.raises(ValueError, match="missing a health-signal outcome"):
        assemble_model_challenge_package(
            health_signal=health,
            active_model=model,
            observations=observations,
            predictions=entries,
            outcomes=outcomes[:-1],
        )

    with pytest.raises(ValueError, match="cannot resolve its prediction"):
        assemble_model_challenge_package(
            health_signal=health,
            active_model=model,
            observations=observations,
            predictions=entries[:-1],
            outcomes=outcomes,
        )


def test_package_rejects_conflicting_observations_for_one_day() -> None:
    observations, model, entries, outcomes, health = history()
    conflicting = observations[-1].model_copy(update={"cash": 123})

    with pytest.raises(ValueError, match="conflict for the same day"):
        assemble_model_challenge_package(
            health_signal=health,
            active_model=model,
            observations=(*observations, conflicting),
            predictions=entries,
            outcomes=outcomes,
        )
