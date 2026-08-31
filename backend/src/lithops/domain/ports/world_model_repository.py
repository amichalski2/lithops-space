"""Append-only persistence contract for versioned company world models."""

from typing import Protocol
from uuid import UUID

from lithops.domain.world_model import WorldModelRelationship, WorldModelVersion


class WorldModelRepository(Protocol):
    async def get_latest_world_model(self, run_id: UUID) -> WorldModelVersion | None: ...

    async def get_world_model(self, model_id: UUID) -> WorldModelVersion: ...

    async def list_world_models(self, run_id: UUID) -> list[WorldModelVersion]: ...

    async def list_world_model_relationships(
        self,
        model_id: UUID,
    ) -> list[WorldModelRelationship]: ...

    async def append_world_model(
        self,
        world_model: WorldModelVersion,
        *,
        expected_latest_version: int | None,
    ) -> WorldModelVersion: ...
