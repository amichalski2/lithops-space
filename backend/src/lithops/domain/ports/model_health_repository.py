"""Append-only persistence contract for deterministic model-health evaluations."""

from typing import Protocol
from uuid import UUID

from lithops.domain.evaluation import ModelHealthSignal


class ModelHealthRepository(Protocol):
    async def append_model_health_signal(
        self,
        signal: ModelHealthSignal,
    ) -> ModelHealthSignal: ...

    async def get_model_health_signal(self, signal_id: UUID) -> ModelHealthSignal: ...

    async def list_model_health_signals(self, run_id: UUID) -> list[ModelHealthSignal]: ...
