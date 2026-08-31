from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from lithops.domain.errors import ConflictError
from lithops.domain.models import (
    ActionCommand,
    ActionPlan,
    CashForecast,
    CashForecasts,
    DecisionRecord,
    ObservationSnapshot,
)
from lithops.domain.predictions import PredictionOutcomeAttribution, PredictionStatus
from lithops.evaluation.forecast_scoring import score_cash_prediction
from lithops.evaluation.maturity import prediction_status
from lithops.evaluation.prediction_ledger import (
    attribute_prediction_policy_path,
    create_cash_prediction,
    mature_cash_predictions,
)
from pydantic import ValidationError

RUN_ID = UUID("44444444-4444-4444-4444-444444444444")
DECISION_ID = UUID("55555555-5555-5555-5555-555555555555")
MODEL_ID = UUID("66666666-6666-6666-6666-666666666666")
COMMITTED_AT = datetime(2026, 8, 25, tzinfo=UTC)


def forecasts(point: float = 100.0) -> CashForecasts:
    return CashForecasts(
        items=[
            CashForecast(horizon_days=horizon, point=point, lower=90, upper=110)
            for horizon in (7, 28, 84, 182)
        ]
    )


def prediction(*, issued_day: int = 0, decision_id: UUID = DECISION_ID):
    return create_cash_prediction(
        run_id=RUN_ID,
        decision_id=decision_id,
        decision_week=issued_day // 7,
        issued_day=issued_day,
        model_version_id=MODEL_ID,
        prompt_version="executive-v1",
        observation_reference=f"observation:{RUN_ID}:{issued_day}",
        assumptions=("Current acquisition efficiency persists.",),
        evidence_references=("world-model-v1",),
        uncertainty_source="world-model-parameter-sampling",
        confidence=0.6,
        forecasts=forecasts(),
        committed_at=COMMITTED_AT,
    )


def observation(day: int, cash: float = 105.0) -> ObservationSnapshot:
    return ObservationSnapshot(
        day=day,
        cash=cash,
        observed_at=COMMITTED_AT,
    )


def test_prediction_commitment_has_exact_immutable_targets() -> None:
    entry = prediction()

    assert [target.target_day for target in entry.targets] == [7, 28, 84, 182]
    assert entry == prediction()
    with pytest.raises(ValidationError, match="frozen"):
        entry.targets[0].point = 999  # type: ignore[misc]


def test_maturity_boundaries_do_not_score_against_the_wrong_day() -> None:
    entry = prediction()
    target = entry.targets[0]

    assert prediction_status(target, current_day=6) is PredictionStatus.PENDING
    assert prediction_status(target, current_day=7) is PredictionStatus.DUE
    assert mature_cash_predictions(
        (entry,),
        observation(8),
        observation_reference="observation:day-8",
    ) == ()


def test_multiple_older_predictions_can_mature_in_the_same_week() -> None:
    day_zero = prediction()
    day_twenty_one = prediction(issued_day=21, decision_id=uuid4())

    outcomes = mature_cash_predictions(
        (day_zero, day_twenty_one),
        observation(28),
        observation_reference="observation:day-28",
    )

    assert len(outcomes) == 2
    assert {outcome.score.target_id for outcome in outcomes} == {
        day_zero.targets[1].id,
        day_twenty_one.targets[0].id,
    }
    assert all(outcome.actual.observed_day == 28 for outcome in outcomes)


def test_repeated_maturation_is_idempotent_and_conflicts_are_rejected() -> None:
    entry = prediction()
    first = mature_cash_predictions(
        (entry,),
        observation(7),
        observation_reference="observation:day-7",
    )
    replay = mature_cash_predictions(
        (entry,),
        observation(7),
        observation_reference="observation:day-7",
        existing_outcomes=first,
    )

    assert len(first) == 1
    assert replay == ()
    assert (
        prediction_status(entry.targets[0], current_day=7, outcome=first[0])
        is PredictionStatus.MATURED
    )

    later_poll = observation(7).model_copy(
        update={"observed_at": COMMITTED_AT + timedelta(seconds=30)}
    )
    assert mature_cash_predictions(
        (entry,),
        later_poll,
        observation_reference="observation:day-7",
        existing_outcomes=first,
    ) == ()

    with pytest.raises(ConflictError, match="different outcome"):
        mature_cash_predictions(
            (entry,),
            observation(7, cash=120),
            observation_reference="observation:day-7-corrected",
            existing_outcomes=first,
        )


def test_cash_scoring_matches_point_interval_and_wis_formulas() -> None:
    entry = prediction()
    inside = mature_cash_predictions(
        (entry,),
        observation(7, cash=105),
        observation_reference="observation:day-7",
    )[0].score

    assert inside.signed_error == 5
    assert inside.absolute_error == 5
    assert inside.absolute_percentage_error == pytest.approx(5 / 105 * 100)
    assert inside.interval_hit is True
    assert inside.interval_width == 20
    assert inside.interval_score == pytest.approx(20)
    assert inside.weighted_interval_score == pytest.approx(2)

    target = entry.targets[0]
    outside_actual = mature_cash_predictions(
        (entry,),
        observation(7, cash=120),
        observation_reference="observation:day-7-outside",
    )[0].actual
    outside = score_cash_prediction(target, outside_actual)
    assert outside.interval_hit is False
    assert outside.interval_score == pytest.approx(420)
    assert outside.weighted_interval_score == pytest.approx(41 / 3)


def test_zero_actual_has_finite_normalized_error_and_no_percentage_error() -> None:
    outcome = mature_cash_predictions(
        (prediction(),),
        observation(7, cash=0),
        observation_reference="observation:day-7-zero",
    )[0]

    assert outcome.score.absolute_percentage_error is None
    assert outcome.score.normalized_absolute_error == 100


def test_matured_prediction_attributes_a_later_policy_change_separately() -> None:
    entry = prediction()
    outcome = mature_cash_predictions(
        (entry,),
        observation(28, cash=70),
        observation_reference="observation:day-28",
    )[0]

    def committed_decision(*, decision_id: UUID, week: int, development: float):
        return DecisionRecord.model_construct(
            id=decision_id,
            week=week,
            action_plan=ActionPlan(
                name=f"week-{week}",
                strategy_family="continuation" if development == 0 else "new_program",
                rationale="test policy",
                commands=[
                    ActionCommand(
                        tool="set_daily_spend",
                        arguments={"development": development},
                        idempotency_key=f"week-{week}",
                    )
                ],
            ),
            candidate_evaluations=[],
            actual_outcome=observation((week + 1) * 7),
        )

    origin = committed_decision(decision_id=DECISION_ID, week=0, development=0)
    held = committed_decision(decision_id=uuid4(), week=1, development=0)
    changed = committed_decision(decision_id=uuid4(), week=2, development=10_000)
    attributed = attribute_prediction_policy_path(
        outcome,
        entry=entry,
        decisions=(origin, held, changed),
    )

    assert attributed.attribution is PredictionOutcomeAttribution.POLICY_PATH_DIVERGED
    assert attributed.policy_divergence_week == 2


def test_matured_prediction_remains_model_evidence_when_policy_is_held() -> None:
    entry = prediction()
    outcome = mature_cash_predictions(
        (entry,),
        observation(28, cash=70),
        observation_reference="observation:day-28",
    )[0]
    plan = ActionPlan(
        name="hold",
        strategy_family="continuation",
        rationale="hold controls",
        commands=[
            ActionCommand(
                tool="set_daily_spend",
                arguments={"development": 0},
                idempotency_key="origin",
            )
        ],
    )
    decisions = tuple(
        DecisionRecord.model_construct(
            id=DECISION_ID if week == 0 else uuid4(),
            week=week,
            action_plan=plan.model_copy(
                update={
                    "commands": [
                        plan.commands[0].model_copy(
                            update={"idempotency_key": f"week-{week}"}
                        )
                    ]
                },
                deep=True,
            ),
            candidate_evaluations=[],
            actual_outcome=observation((week + 1) * 7),
        )
        for week in range(4)
    )

    attributed = attribute_prediction_policy_path(
        outcome,
        entry=entry,
        decisions=decisions,
    )

    assert attributed.attribution is PredictionOutcomeAttribution.MODEL_PERFORMANCE
    assert attributed.policy_divergence_week is None
