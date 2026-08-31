"""Immutable lineage for composable causal world-model components."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelComponentScope(StrEnum):
    LEAD_ARRIVAL = "lead_arrival"
    CONVERSION = "conversion"
    QUALITY_DYNAMICS = "quality_dynamics"
    RETENTION = "retention"
    PRICING_REVENUE = "pricing_revenue"
    CAPACITY_COST = "capacity_cost"
    RESIDUAL_CASH = "residual_cash"


class ComponentAssignment(BaseModel):
    """One fitted component selected for one causal responsibility."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scope: ModelComponentScope
    artifact_id: UUID
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fitted_model_id: UUID
    fitted_state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class WorldModelAssembly(BaseModel):
    """Content-addressed manifest used to reproduce a composed prediction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    protocol_version: str = Field(default="1.0", pattern=r"^\d+\.\d+$")
    components: tuple[ComponentAssignment, ...] = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @staticmethod
    def _content_identity(
        components: tuple[ComponentAssignment, ...],
        protocol_version: str,
    ) -> tuple[str, UUID]:
        payload: dict[str, Any] = {
            "protocol_version": protocol_version,
            "components": [item.model_dump(mode="json") for item in components],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        identity = uuid5(NAMESPACE_URL, f"lithops:world-model-assembly:{content_hash}")
        return content_hash, identity

    @classmethod
    def create(
        cls,
        *,
        components: tuple[ComponentAssignment, ...],
        protocol_version: str = "1.0",
    ) -> WorldModelAssembly:
        ordered = tuple(sorted(components, key=lambda item: item.scope.value))
        content_hash, identity = cls._content_identity(ordered, protocol_version)
        return cls(
            id=identity,
            protocol_version=protocol_version,
            components=ordered,
            content_hash=content_hash,
        )

    @model_validator(mode="after")
    def validate_identity(self) -> WorldModelAssembly:
        scopes = [item.scope for item in self.components]
        if len(scopes) != len(set(scopes)):
            raise ValueError("world-model assembly contains duplicate component scopes")
        if tuple(sorted(scopes, key=lambda item: item.value)) != tuple(scopes):
            raise ValueError("world-model assembly components must use canonical scope order")
        expected_hash, expected_id = self._content_identity(
            self.components,
            self.protocol_version,
        )
        if expected_hash != self.content_hash or expected_id != self.id:
            raise ValueError("world-model assembly identity does not match its components")
        return self
