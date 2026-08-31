"""Secret-free pointer to the canonical persisted weekly experiment state."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lithops.domain.experiment_contracts import (
    CAPABILITY_CATALOG_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
    EXPERIMENT_PROTOCOL_VERSION,
    OBSERVATION_CONTRACT_VERSION,
)
from lithops.domain.models import RunRecord, RunStatus, utc_now


class ExperimentCheckpoint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["lithops-experiment-checkpoint-v2"] = (
        CHECKPOINT_SCHEMA_VERSION
    )
    run_id: UUID
    benchmark_session_id: str | None = None
    benchmark: str = "ceobench"
    provider: str = "openrouter"
    model: str = Field(min_length=1)
    benchmark_seed: int
    target_weeks: int = Field(ge=1)
    executive_authority_v2: bool = False
    observation_contract_version: str = OBSERVATION_CONTRACT_VERSION
    experiment_protocol_version: str = EXPERIMENT_PROTOCOL_VERSION
    capability_catalog_version: str = CAPABILITY_CATALOG_VERSION
    current_day: int = Field(ge=0)
    completed_weeks: int = Field(ge=0)
    run_status: RunStatus
    run_version: int = Field(ge=0)
    world_model_version: int | None = Field(default=None, ge=1)
    last_decision_id: UUID | None = None
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_week_boundary(self) -> ExperimentCheckpoint:
        if self.current_day != self.completed_weeks * 7:
            raise ValueError("checkpoint must describe a committed seven-day boundary")
        if self.completed_weeks > self.target_weeks:
            raise ValueError("checkpoint exceeds the experiment target")
        return self

    @classmethod
    def from_run(
        cls,
        run: RunRecord,
        *,
        provider: str = "openrouter",
        model: str,
        benchmark_seed: int,
        target_weeks: int,
        world_model_version: int | None,
        executive_authority_v2: bool = False,
        observation_contract_version: str = OBSERVATION_CONTRACT_VERSION,
        experiment_protocol_version: str = EXPERIMENT_PROTOCOL_VERSION,
        capability_catalog_version: str = CAPABILITY_CATALOG_VERSION,
    ) -> ExperimentCheckpoint:
        if run.horizon_days != target_weeks * 7:
            raise ValueError("run horizon does not match the experiment target")
        return cls(
            run_id=run.id,
            benchmark_session_id=run.benchmark_session_id,
            provider=provider,
            model=model,
            benchmark_seed=benchmark_seed,
            target_weeks=target_weeks,
            executive_authority_v2=executive_authority_v2,
            observation_contract_version=observation_contract_version,
            experiment_protocol_version=experiment_protocol_version,
            capability_catalog_version=capability_catalog_version,
            current_day=run.current_day,
            completed_weeks=run.current_day // 7,
            run_status=run.status,
            run_version=run.version,
            world_model_version=world_model_version,
            last_decision_id=run.last_decision_id,
            updated_at=run.updated_at,
        )

    def assert_compatible(
        self,
        run: RunRecord,
        *,
        provider: str,
        model: str,
        benchmark_seed: int,
        target_weeks: int,
        executive_authority_v2: bool = False,
        observation_contract_version: str = OBSERVATION_CONTRACT_VERSION,
        experiment_protocol_version: str = EXPERIMENT_PROTOCOL_VERSION,
        capability_catalog_version: str = CAPABILITY_CATALOG_VERSION,
    ) -> None:
        expected = {
            "run_id": str(run.id),
            "provider": provider,
            "model": model,
            "benchmark_seed": benchmark_seed,
            "target_weeks": target_weeks,
            "executive_authority_v2": executive_authority_v2,
            "observation_contract_version": observation_contract_version,
            "experiment_protocol_version": experiment_protocol_version,
            "capability_catalog_version": capability_catalog_version,
            "horizon_days": target_weeks * 7,
        }
        actual = {
            "run_id": str(self.run_id),
            "provider": self.provider,
            "model": self.model,
            "benchmark_seed": self.benchmark_seed,
            "target_weeks": self.target_weeks,
            "executive_authority_v2": self.executive_authority_v2,
            "observation_contract_version": self.observation_contract_version,
            "experiment_protocol_version": self.experiment_protocol_version,
            "capability_catalog_version": self.capability_catalog_version,
            "horizon_days": run.horizon_days,
        }
        mismatches = [key for key, value in expected.items() if actual[key] != value]
        if mismatches:
            raise ValueError(
                "checkpoint is incompatible with requested experiment: "
                + ", ".join(mismatches)
            )
