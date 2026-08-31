"""Combined learning-artifact contract required by the weekly state machine."""

from typing import Protocol

from lithops.domain.ports.model_challenge_repository import ModelChallengeRepository
from lithops.domain.ports.model_health_repository import ModelHealthRepository
from lithops.domain.ports.prediction_repository import PredictionRepository
from lithops.domain.ports.world_model_repository import WorldModelRepository


class LearningRepository(
    PredictionRepository,
    WorldModelRepository,
    ModelHealthRepository,
    ModelChallengeRepository,
    Protocol,
):
    pass
