"""Runtime port for versioned executable company-model artifacts."""

from __future__ import annotations

from typing import Protocol

from lithops.domain.executable_model import (
    CompanyModelFitRequest,
    CompanyModelPredictRequest,
    FittedModel,
    ModelArtifact,
    ModelOutcomeDistribution,
)


class ExecutableCompanyModel(Protocol):
    @property
    def artifact(self) -> ModelArtifact: ...

    def fit(self, request: CompanyModelFitRequest) -> FittedModel: ...

    def predict(self, request: CompanyModelPredictRequest) -> ModelOutcomeDistribution: ...

    def diagnostics(self, fitted_model: FittedModel) -> dict: ...
