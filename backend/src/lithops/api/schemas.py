"""Read models composed for the simulation cockpit."""

from pydantic import BaseModel, ConfigDict

from lithops.domain.evaluation import ModelHealthSignal
from lithops.domain.models import DecisionRecord, RunRecord
from lithops.domain.predictions import PredictionLedgerEntry, PredictionOutcome
from lithops.domain.world_model import WorldModelVersion


class PredictionView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction: PredictionLedgerEntry
    outcomes: list[PredictionOutcome]


class DecisionExplanation(BaseModel):
    """Everything needed to audit what Lithops believed and why it acted."""

    model_config = ConfigDict(extra="forbid")

    decision: DecisionRecord
    world_model: WorldModelVersion
    prediction: PredictionView
    model_health_signals: list[ModelHealthSignal]


class RunReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: RunRecord
    decision_count: int
    prediction_count: int
    matured_outcome_count: int
    world_model_version: int | None
    latest_model_health: ModelHealthSignal | None
