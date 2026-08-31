"""Clock rules for pending, due, matured, and explicitly invalidated predictions."""

from uuid import UUID

from lithops.domain.predictions import (
    CashPredictionTarget,
    PredictionOutcome,
    PredictionStatus,
)


def prediction_status(
    target: CashPredictionTarget,
    *,
    current_day: int,
    outcome: PredictionOutcome | None = None,
    invalidated_target_ids: frozenset[UUID] = frozenset(),
) -> PredictionStatus:
    if target.id in invalidated_target_ids:
        return PredictionStatus.INVALIDATED
    if outcome is not None:
        if outcome.target_id != target.id:
            raise ValueError("outcome does not belong to prediction target")
        return PredictionStatus.MATURED
    if current_day >= target.target_day:
        return PredictionStatus.DUE
    return PredictionStatus.PENDING


def is_exactly_maturing(target: CashPredictionTarget, observed_day: int) -> bool:
    """Never substitute a later snapshot for the target day's missing actual."""

    return target.target_day == observed_day
