"""Append-only persistence contract for predictions and their later outcomes."""

from typing import Protocol
from uuid import UUID

from lithops.domain.predictions import PredictionLedgerEntry, PredictionOutcome


class PredictionRepository(Protocol):
    async def append_prediction(
        self,
        prediction: PredictionLedgerEntry,
    ) -> PredictionLedgerEntry: ...

    async def get_prediction(self, prediction_id: UUID) -> PredictionLedgerEntry: ...

    async def list_predictions(self, run_id: UUID) -> list[PredictionLedgerEntry]: ...

    async def get_prediction_outcome(self, target_id: UUID) -> PredictionOutcome | None: ...

    async def list_prediction_outcomes(self, run_id: UUID) -> list[PredictionOutcome]: ...

    async def append_prediction_outcome(
        self,
        outcome: PredictionOutcome,
    ) -> PredictionOutcome: ...
