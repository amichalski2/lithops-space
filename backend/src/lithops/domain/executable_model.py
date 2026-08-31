"""Provider-neutral contracts for executable, versioned company models."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from lithops.domain.component_program import ConversionComponentProgram
from lithops.domain.economics import AccountingPeriod
from lithops.domain.models import utc_now


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class ModelRuntimeKind(StrEnum):
    SANDBOXED_PYTHON = "sandboxed_python"
    TRUSTED_BASELINE = "trusted_baseline"
    TYPED_COMPONENT_ASSEMBLY = "typed_component_assembly"


class ModelEntrypoint(StrEnum):
    FIT = "fit"
    PREDICT = "predict"
    DIAGNOSTICS = "diagnostics"


class AssertionOperator(StrEnum):
    EQUALS = "equals"
    APPROX = "approx"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"


class ModelFeatureRequirement(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(
        min_length=1,
        max_length=120,
        pattern=r"^(history|state|action)\.[a-z][a-z0-9_]*$",
    )
    unit: str = Field(min_length=1, max_length=80)
    required: bool = True


class ModelArtifactAssertion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(min_length=1, max_length=240, pattern=r"^[a-zA-Z0-9_.]+$")
    operator: AssertionOperator
    expected: float
    tolerance: float = Field(default=1e-9, ge=0.0, le=1_000_000.0)

    @field_validator("operator", mode="before")
    @classmethod
    def normalize_equal_alias(cls, value: Any) -> Any:
        return AssertionOperator.EQUALS if value == "equal" else value

    @model_validator(mode="after")
    def validate_assertion_dsl(self) -> ModelArtifactAssertion:
        if self.path == "result" or self.path.startswith("result."):
            raise ValueError(
                "assertion paths are relative to the returned object; omit the result prefix"
            )
        return self


class ModelArtifactTestCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=120, pattern=r"^[a-z][a-z0-9_]*$")
    entrypoint: ModelEntrypoint
    arguments: dict[str, JsonValue]
    assertions: tuple[ModelArtifactAssertion, ...] = Field(min_length=1, max_length=40)


class ModelArtifact(BaseModel):
    """Immutable source artifact; its ID and hash are derived from canonical content."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    name: str = Field(min_length=1, max_length=120, pattern=r"^[a-z][a-z0-9_-]*$")
    protocol_version: str = Field(default="1.0", pattern=r"^\d+\.\d+$")
    runtime_kind: ModelRuntimeKind
    scope: str = Field(min_length=1, max_length=120)
    hypothesis: str = Field(min_length=1, max_length=2_000)
    authoring_agent: str = Field(min_length=1, max_length=120)
    provider: str = Field(min_length=1, max_length=120)
    model_name: str = Field(min_length=1, max_length=160)
    prompt_version: str = Field(min_length=1, max_length=120)
    source_code: str | None = Field(default=None, max_length=100_000)
    component_program: ConversionComponentProgram | None = None
    trusted_entrypoint: str | None = Field(
        default=None,
        max_length=240,
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_.:]*$",
    )
    dependencies: tuple[str, ...] = Field(default=(), max_length=30)
    required_features: tuple[ModelFeatureRequirement, ...] = ()
    required_priors: tuple[str, ...] = Field(default=(), max_length=30)
    tests: tuple[ModelArtifactTestCase, ...] = Field(default=(), max_length=100)
    limitations: tuple[str, ...] = Field(default=(), max_length=40)
    parent_artifact_id: UUID | None = None
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime = Field(default_factory=utc_now)

    @classmethod
    def create(cls, **values: Any) -> ModelArtifact:
        draft = cls.model_construct(
            id=UUID(int=0),
            content_hash="0" * 64,
            **values,
        )
        payload = draft.model_dump(
            exclude={"id", "content_hash", "created_at"},
            mode="json",
        )
        content_hash = _sha256(payload)
        return cls(
            **values,
            content_hash=content_hash,
            id=uuid5(NAMESPACE_URL, f"lithops:model-artifact:{content_hash}"),
        )

    @model_validator(mode="after")
    def validate_runtime_and_hash(self) -> ModelArtifact:
        if self.runtime_kind == ModelRuntimeKind.SANDBOXED_PYTHON:
            if not self.source_code or not self.source_code.strip():
                raise ValueError("sandboxed Python artifacts require source code")
            if self.trusted_entrypoint is not None:
                raise ValueError("sandboxed Python artifacts cannot use a trusted entrypoint")
            if self.component_program is not None:
                raise ValueError("sandboxed Python artifacts cannot embed component programs")
        elif self.runtime_kind == ModelRuntimeKind.TRUSTED_BASELINE:
            if not self.trusted_entrypoint:
                raise ValueError("trusted baseline artifacts require a trusted entrypoint")
            if self.source_code is not None:
                raise ValueError("trusted baseline artifacts cannot embed generated source code")
            if self.component_program is not None:
                raise ValueError("trusted baseline artifacts cannot embed component programs")
        elif self.runtime_kind == ModelRuntimeKind.TYPED_COMPONENT_ASSEMBLY:
            if self.component_program is None:
                raise ValueError("typed component artifacts require a component program")
            if self.source_code is not None or self.trusted_entrypoint is not None:
                raise ValueError(
                    "typed component artifacts cannot embed source or trusted entrypoints"
                )

        dependency_names = list(self.dependencies)
        if len(dependency_names) != len(set(dependency_names)):
            raise ValueError("artifact dependencies must be unique")
        feature_names = [feature.name for feature in self.required_features]
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("artifact feature names must be unique")
        if len(self.required_priors) != len(set(self.required_priors)):
            raise ValueError("artifact prior names must be unique")
        if any(not item or not item.replace("_", "").isalnum() for item in self.required_priors):
            raise ValueError("artifact prior names must use letters, digits, and underscores")
        test_names = [test.name for test in self.tests]
        if len(test_names) != len(set(test_names)):
            raise ValueError("artifact test names must be unique")
        if any(not item.strip() for item in self.limitations):
            raise ValueError("artifact limitations cannot be empty")

        values = self.model_dump(
            exclude={"id", "content_hash", "created_at"},
            mode="json",
        )
        expected_hash = _sha256(values)
        expected_id = uuid5(NAMESPACE_URL, f"lithops:model-artifact:{expected_hash}")
        if self.content_hash != expected_hash:
            raise ValueError("model artifact content hash does not match canonical content")
        if self.id != expected_id:
            raise ValueError("model artifact ID does not match its content hash")
        return self


class CompanyModelFitRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    observation_ids: tuple[str, ...] = Field(min_length=1)
    training_start_day: int = Field(ge=0)
    training_end_day: int = Field(ge=0)
    history: tuple[dict[str, JsonValue], ...] = Field(min_length=1)
    prior: dict[str, JsonValue] = Field(default_factory=dict)
    seed: int = 0

    @model_validator(mode="after")
    def validate_training_window(self) -> CompanyModelFitRequest:
        if self.training_end_day < self.training_start_day:
            raise ValueError("training end day cannot precede the start day")
        if len(self.observation_ids) != len(set(self.observation_ids)):
            raise ValueError("fit observation IDs must be unique")
        return self


class FittedModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    artifact_id: UUID
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    observation_ids: tuple[str, ...] = Field(min_length=1)
    training_start_day: int = Field(ge=0)
    training_end_day: int = Field(ge=0)
    fitted_state: dict[str, JsonValue]
    state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime = Field(default_factory=utc_now)

    @classmethod
    def create(
        cls,
        *,
        artifact: ModelArtifact,
        request: CompanyModelFitRequest,
        fitted_state: dict[str, JsonValue],
    ) -> FittedModel:
        state_payload = {
            "artifact_id": str(artifact.id),
            "artifact_hash": artifact.content_hash,
            "observation_ids": list(request.observation_ids),
            "training_start_day": request.training_start_day,
            "training_end_day": request.training_end_day,
            "fitted_state": fitted_state,
        }
        state_hash = _sha256(state_payload)
        return cls(
            id=uuid5(NAMESPACE_URL, f"lithops:fitted-model:{state_hash}"),
            artifact_id=artifact.id,
            artifact_hash=artifact.content_hash,
            observation_ids=request.observation_ids,
            training_start_day=request.training_start_day,
            training_end_day=request.training_end_day,
            fitted_state=fitted_state,
            state_hash=state_hash,
        )

    @model_validator(mode="after")
    def validate_fitted_state(self) -> FittedModel:
        if self.training_end_day < self.training_start_day:
            raise ValueError("fitted-model end day cannot precede the start day")
        payload = {
            "artifact_id": str(self.artifact_id),
            "artifact_hash": self.artifact_hash,
            "observation_ids": list(self.observation_ids),
            "training_start_day": self.training_start_day,
            "training_end_day": self.training_end_day,
            "fitted_state": self.fitted_state,
        }
        expected_hash = _sha256(payload)
        expected_id = uuid5(NAMESPACE_URL, f"lithops:fitted-model:{expected_hash}")
        if self.state_hash != expected_hash or self.id != expected_id:
            raise ValueError("fitted-model identity does not match canonical state")
        return self


class CompanyModelPredictRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    fitted_model: FittedModel
    state: dict[str, JsonValue]
    action: dict[str, JsonValue]
    # Optional sequence of the actions that were actually committed after the
    # origin.  Planning leaves this empty and therefore asks a hold-action
    # counterfactual.  Temporal backtests populate it so a 28/84-day fold is not
    # scored as though the first weekly decision stayed in force forever.
    policy_action_path: tuple[dict[str, JsonValue], ...] = Field(
        default=(),
        max_length=52,
    )
    horizons_days: tuple[int, ...] = Field(min_length=1, max_length=30)
    n_rollouts: int = Field(default=1_000, ge=1, le=100_000)
    seed: int = 0

    @model_validator(mode="after")
    def validate_horizons(self) -> CompanyModelPredictRequest:
        if any(day < 1 for day in self.horizons_days):
            raise ValueError("prediction horizons must be positive")
        if tuple(sorted(set(self.horizons_days))) != self.horizons_days:
            raise ValueError("prediction horizons must be unique and ordered")
        if self.policy_action_path:
            if any(day % 7 for day in self.horizons_days):
                raise ValueError("policy action paths require whole-week horizons")
            required_weeks = max(self.horizons_days) // 7
            if len(self.policy_action_path) != required_weeks:
                raise ValueError(
                    "policy action path must contain one committed action per forecast week"
                )
            if self.policy_action_path[0] != self.action:
                raise ValueError("the first policy action must equal the committed action")
        return self


class ModelOutcomeSample(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rollout_index: int = Field(ge=0)
    horizon_days: int = Field(ge=1)
    cash: float
    revenue_weekly: float = Field(ge=0.0)
    customers: float = Field(ge=0.0)
    churn_rate: float = Field(ge=0.0, le=1.0)
    # Component-local observables are optional for legacy sandbox artifacts. Trusted
    # assemblies emit them so a conversion hypothesis can be scored on conversion,
    # rather than only through diluted whole-company outcomes.
    weekly_leads: float | None = Field(default=None, ge=0.0)
    weekly_conversions: float | None = Field(default=None, ge=0.0)
    accounting: AccountingPeriod


class ModelOutcomeDistribution(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_id: UUID
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fitted_model_id: UUID
    horizons_days: tuple[int, ...] = Field(min_length=1)
    n_rollouts: int = Field(ge=1)
    samples: tuple[ModelOutcomeSample, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_distribution_shape(self) -> ModelOutcomeDistribution:
        expected = {
            (rollout_index, horizon)
            for rollout_index in range(self.n_rollouts)
            for horizon in self.horizons_days
        }
        actual = {(sample.rollout_index, sample.horizon_days) for sample in self.samples}
        if actual != expected or len(actual) != len(self.samples):
            raise ValueError("outcome samples must contain each rollout and horizon exactly once")
        return self
