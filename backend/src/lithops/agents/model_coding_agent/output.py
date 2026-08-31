"""Strict provider-facing schema for agent-authored executable model artifacts."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from lithops.domain.executable_model import (
    ModelArtifact,
    ModelArtifactTestCase,
    ModelFeatureRequirement,
    ModelRuntimeKind,
)
from lithops.domain.model_challenge import HypothesisFamily

_FEATURES_ADAPTER = TypeAdapter(tuple[ModelFeatureRequirement, ...])
_TESTS_ADAPTER = TypeAdapter(tuple[ModelArtifactTestCase, ...])


class ModelArtifactDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    name: str = Field(min_length=1, max_length=120, pattern=r"^[a-z][a-z0-9_-]*$")
    family: HypothesisFamily
    scope: str = Field(min_length=1, max_length=120)
    hypothesis: str = Field(min_length=1, max_length=2_000)
    source_lines: tuple[str, ...] = Field(min_length=3, max_length=4_000)
    required_features_json: str = Field(min_length=2, max_length=30_000)
    required_priors: tuple[str, ...] = Field(default=(), max_length=20)
    tests_json: str = Field(min_length=2, max_length=60_000)
    limitations: tuple[str, ...] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def validate_draft(self) -> ModelArtifactDraft:
        if any("\n" in line or "\r" in line for line in self.source_lines):
            raise ValueError("model draft source lines cannot contain embedded newlines")
        if len(self.source_code) > 100_000:
            raise ValueError("model draft source exceeds 100000 characters")
        feature_names = [feature.name for feature in self.required_features]
        if not 1 <= len(feature_names) <= 40:
            raise ValueError("model draft requires between 1 and 40 features")
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("model draft feature names must be unique")
        if len(self.required_priors) != len(set(self.required_priors)):
            raise ValueError("model draft prior names must be unique")
        if any(not item.strip() for item in self.limitations):
            raise ValueError("model draft limitations cannot be empty")
        required_functions = ("def fit(", "def predict(", "def diagnostics(")
        missing = [name for name in required_functions if name not in self.source_code]
        if missing:
            raise ValueError("model draft is missing required executable entrypoints")
        test_entrypoints = {test.entrypoint.value for test in self.tests}
        if not 1 <= len(self.tests) <= 30:
            raise ValueError("model draft requires between 1 and 30 tests")
        if not {"fit", "predict"}.issubset(test_entrypoints):
            raise ValueError("model draft requires both fit and predict tests")
        return self

    @property
    def source_code(self) -> str:
        return "\n".join(self.source_lines).strip() + "\n"

    @property
    def required_features(self) -> tuple[ModelFeatureRequirement, ...]:
        try:
            return _FEATURES_ADAPTER.validate_json(self.required_features_json)
        except ValueError as exc:
            raise ValueError(
                f"required_features_json must contain valid feature declarations: {exc}"
            ) from exc

    @property
    def tests(self) -> tuple[ModelArtifactTestCase, ...]:
        try:
            return _TESTS_ADAPTER.validate_json(self.tests_json)
        except ValueError as exc:
            raise ValueError(
                f"tests_json must contain valid executable test cases: {exc}"
            ) from exc

    def to_artifact(
        self,
        *,
        authoring_agent: str,
        provider: str,
        model_name: str,
        prompt_version: str,
        parent_artifact_id: UUID,
    ) -> ModelArtifact:
        return ModelArtifact.create(
            name=self.name,
            protocol_version="1.1",
            runtime_kind=ModelRuntimeKind.SANDBOXED_PYTHON,
            scope=self.scope,
            hypothesis=self.hypothesis,
            authoring_agent=authoring_agent,
            provider=provider,
            model_name=model_name,
            prompt_version=prompt_version,
            source_code=self.source_code,
            dependencies=(),
            required_features=self.required_features,
            required_priors=self.required_priors,
            tests=self.tests,
            limitations=self.limitations,
            parent_artifact_id=parent_artifact_id,
        )
