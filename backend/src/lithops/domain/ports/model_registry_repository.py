"""Persistence boundary for executable-model lineage and activation."""

from typing import Protocol
from uuid import UUID

from lithops.domain.executable_model import FittedModel, ModelArtifact
from lithops.domain.model_challenge import ModelChallengePackage, ModelChallengeRecord
from lithops.domain.model_registry import (
    ActiveModelAssignment,
    ModelArtifactAuthoringReceipt,
    ModelPromotionDecision,
    SandboxExecutionRecord,
    TemporalEvaluationFold,
)


class ModelRegistryRepository(Protocol):
    async def get_model_challenge(
        self, challenge_id: UUID
    ) -> ModelChallengeRecord | None: ...

    async def save_model_challenge(
        self, challenge: ModelChallengeRecord
    ) -> ModelChallengeRecord: ...

    async def append_model_challenge_package(
        self, package: ModelChallengePackage
    ) -> ModelChallengePackage: ...

    async def get_model_challenge_package(
        self, challenge_id: UUID
    ) -> ModelChallengePackage | None: ...

    async def append_model_artifact(self, artifact: ModelArtifact) -> ModelArtifact: ...

    async def get_model_artifact(self, artifact_id: UUID) -> ModelArtifact: ...

    async def append_model_artifact_authoring_receipt(
        self,
        receipt: ModelArtifactAuthoringReceipt,
    ) -> ModelArtifactAuthoringReceipt: ...

    async def list_model_artifact_authoring_receipts(
        self,
        run_id: UUID,
        challenge_id: UUID,
    ) -> list[ModelArtifactAuthoringReceipt]: ...

    async def append_fitted_model(self, run_id: UUID, fitted: FittedModel) -> FittedModel: ...

    async def get_fitted_model(self, run_id: UUID, fitted_model_id: UUID) -> FittedModel: ...

    async def append_sandbox_execution(
        self, execution: SandboxExecutionRecord
    ) -> SandboxExecutionRecord: ...

    async def list_sandbox_executions(self, run_id: UUID) -> list[SandboxExecutionRecord]: ...

    async def append_temporal_evaluation_fold(
        self, fold: TemporalEvaluationFold
    ) -> TemporalEvaluationFold: ...

    async def list_temporal_evaluation_folds(
        self,
        run_id: UUID,
        artifact_id: UUID | None = None,
        challenge_id: UUID | None = None,
    ) -> list[TemporalEvaluationFold]: ...

    async def append_model_promotion_decision(
        self, decision: ModelPromotionDecision
    ) -> ModelPromotionDecision: ...

    async def get_model_promotion_decision(
        self, decision_id: UUID
    ) -> ModelPromotionDecision: ...

    async def get_model_promotion_decision_for_challenge(
        self,
        run_id: UUID,
        challenge_id: UUID,
    ) -> ModelPromotionDecision | None: ...

    async def activate_model(
        self,
        assignment: ActiveModelAssignment,
        *,
        expected_previous_sequence: int | None,
    ) -> ActiveModelAssignment: ...

    async def get_active_model(self, run_id: UUID) -> ActiveModelAssignment | None: ...

    async def list_model_activations(self, run_id: UUID) -> list[ActiveModelAssignment]: ...
