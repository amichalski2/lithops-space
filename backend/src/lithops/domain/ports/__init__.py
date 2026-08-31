"""Technology-independent contracts used by Lithops application services."""

from lithops.domain.ports.benchmark import BenchmarkPort
from lithops.domain.ports.executable_model import ExecutableCompanyModel
from lithops.domain.ports.learning_repository import LearningRepository
from lithops.domain.ports.model_challenge_repository import ModelChallengeRepository
from lithops.domain.ports.model_health_repository import ModelHealthRepository
from lithops.domain.ports.model_provider import StructuredModelProvider
from lithops.domain.ports.prediction_repository import PredictionRepository
from lithops.domain.ports.repository import RunRepository
from lithops.domain.ports.strategy_repository import StrategyRepository
from lithops.domain.ports.world_model_repository import WorldModelRepository

__all__ = [
    "BenchmarkPort",
    "ExecutableCompanyModel",
    "LearningRepository",
    "ModelHealthRepository",
    "ModelChallengeRepository",
    "PredictionRepository",
    "RunRepository",
    "StrategyRepository",
    "StructuredModelProvider",
    "WorldModelRepository",
]
