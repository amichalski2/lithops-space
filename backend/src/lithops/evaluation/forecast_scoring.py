"""Proper deterministic scores for one point and 95% interval cash forecast."""

from __future__ import annotations

from lithops.domain.predictions import (
    CashPredictionTarget,
    PredictionActual,
    PredictionScore,
)
from lithops.evaluation.interval_math import interval_score as _interval_score
from lithops.evaluation.interval_math import (
    weighted_interval_score as _weighted_interval_score,
)


def score_cash_prediction(
    target: CashPredictionTarget,
    actual: PredictionActual,
    *,
    normalization_floor: float = 1.0,
) -> PredictionScore:
    if actual.target_id != target.id:
        raise ValueError("actual does not belong to the supplied prediction target")
    if normalization_floor <= 0:
        raise ValueError("normalization_floor must be positive")

    signed_error = actual.cash - target.point
    absolute_error = abs(signed_error)
    percentage_error = (
        absolute_error / abs(actual.cash) * 100.0 if actual.cash != 0 else None
    )
    normalized_error = absolute_error / max(abs(actual.cash), normalization_floor)
    interval_width = target.upper - target.lower
    alpha = 1.0 - target.interval_probability
    interval_score = _interval_score(
        lower=target.lower,
        upper=target.upper,
        actual=actual.cash,
        alpha=alpha,
    )
    weighted_interval_score = _weighted_interval_score(
        point=target.point,
        lower=target.lower,
        upper=target.upper,
        actual=actual.cash,
        interval_probability=target.interval_probability,
    )

    return PredictionScore(
        target_id=target.id,
        signed_error=signed_error,
        absolute_error=absolute_error,
        absolute_percentage_error=percentage_error,
        normalized_absolute_error=normalized_error,
        interval_hit=target.lower <= actual.cash <= target.upper,
        interval_width=interval_width,
        interval_score=interval_score,
        weighted_interval_score=weighted_interval_score,
        scored_at=actual.observed_at,
    )
