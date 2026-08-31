from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import httpx

from lithops.domain.errors import (
    ConflictError,
    NotFoundError,
    OperationInProgressError,
    RepositoryError,
)
from lithops.domain.evaluation import ModelHealthSignal
from lithops.domain.executable_model import FittedModel, ModelArtifact
from lithops.domain.insights import InsightRecord
from lithops.domain.model_challenge import (
    HypothesisBacktestResult,
    ModelBuilderCallReceipt,
    ModelBuilderProposal,
    ModelChallengeDecision,
    ModelChallengePackage,
    ModelChallengeRecord,
    ModelChallengeStatus,
)
from lithops.domain.model_registry import (
    ActiveModelAssignment,
    ModelArtifactAuthoringReceipt,
    ModelPromotionDecision,
    PromotionDisposition,
    SandboxExecutionRecord,
    TemporalEvaluationFold,
)
from lithops.domain.models import (
    ActionReceipt,
    CandidateEvaluationRecord,
    DecisionRecord,
    EventRecord,
    OperationStatus,
    RunLease,
    RunRecord,
    StepOperation,
    StepResult,
    utc_now,
)
from lithops.domain.predictions import PredictionLedgerEntry, PredictionOutcome
from lithops.domain.strategy import (
    CandidateEvaluationSet,
    CommitmentReview,
    ExecutiveChoice,
    ExperimentOutcome,
    StrategicPortfolioRevision,
)
from lithops.domain.world_model import WorldModelRelationship, WorldModelVersion


class InMemoryRunRepository:
    def __init__(self) -> None:
        self._runs: dict[UUID, RunRecord] = {}
        self._decisions: dict[UUID, DecisionRecord] = {}
        self._decision_by_week: dict[tuple[UUID, int], UUID] = {}
        self._receipts: dict[UUID, ActionReceipt] = {}
        self._receipt_by_key: dict[tuple[UUID, str], UUID] = {}
        self._events: dict[UUID, list[EventRecord]] = {}
        self._operations: dict[tuple[UUID, str], StepOperation] = {}
        self._run_leases: dict[UUID, RunLease] = {}
        self._world_models: dict[UUID, WorldModelVersion] = {}
        self._world_models_by_run: dict[UUID, list[UUID]] = {}
        self._predictions: dict[UUID, PredictionLedgerEntry] = {}
        self._predictions_by_run: dict[UUID, list[UUID]] = {}
        self._prediction_outcomes: dict[UUID, PredictionOutcome] = {}
        self._model_health_signals: dict[UUID, ModelHealthSignal] = {}
        self._model_health_signals_by_run: dict[UUID, list[UUID]] = {}
        self._model_challenges: dict[UUID, ModelChallengeRecord] = {}
        self._model_challenge_packages: dict[UUID, ModelChallengePackage] = {}
        self._model_builder_proposals: dict[UUID, ModelBuilderProposal] = {}
        self._hypothesis_backtests: dict[UUID, HypothesisBacktestResult] = {}
        self._model_builder_calls: dict[UUID, ModelBuilderCallReceipt] = {}
        self._model_challenge_decisions: dict[UUID, ModelChallengeDecision] = {}
        self._model_challenge_decision_by_challenge: dict[UUID, UUID] = {}
        self._model_artifacts: dict[UUID, ModelArtifact] = {}
        self._model_artifact_authoring_receipts: dict[
            tuple[UUID, UUID, str], ModelArtifactAuthoringReceipt
        ] = {}
        self._fitted_models: dict[tuple[UUID, UUID], FittedModel] = {}
        self._sandbox_executions: dict[UUID, SandboxExecutionRecord] = {}
        self._sandbox_execution_by_key: dict[tuple[UUID, str], UUID] = {}
        self._temporal_evaluation_folds: dict[UUID, TemporalEvaluationFold] = {}
        self._promotion_decisions: dict[UUID, ModelPromotionDecision] = {}
        self._promotion_decision_by_challenge: dict[tuple[UUID, UUID], UUID] = {}
        self._model_activations: dict[UUID, ActiveModelAssignment] = {}
        self._model_activations_by_run: dict[UUID, list[UUID]] = {}
        self._activation_by_promotion: dict[UUID, UUID] = {}
        self._portfolio_revisions: dict[UUID, StrategicPortfolioRevision] = {}
        self._portfolio_revisions_by_run: dict[UUID, list[UUID]] = {}
        self._experiment_outcomes: dict[UUID, ExperimentOutcome] = {}
        self._experiment_outcomes_by_run: dict[UUID, list[UUID]] = {}
        self._commitment_reviews: dict[UUID, CommitmentReview] = {}
        self._commitment_reviews_by_run: dict[UUID, list[UUID]] = {}
        self._candidate_evaluation_sets: dict[UUID, CandidateEvaluationSet] = {}
        self._evaluation_set_by_week: dict[tuple[UUID, int], UUID] = {}
        self._executive_choices: dict[UUID, ExecutiveChoice] = {}
        self._executive_choice_by_week: dict[tuple[UUID, int], UUID] = {}
        self._insight_records: dict[UUID, InsightRecord] = {}
        self._insight_records_by_run: dict[UUID, list[UUID]] = {}
        self._lock = asyncio.Lock()

    async def append_model_artifact(self, artifact: ModelArtifact) -> ModelArtifact:
        async with self._lock:
            existing = self._model_artifacts.get(artifact.id)
            if existing is not None:
                if existing.content_hash != artifact.content_hash:
                    raise ConflictError(f"model artifact ID conflict: {artifact.id}")
                return existing.model_copy(deep=True)
            self._model_artifacts[artifact.id] = artifact.model_copy(deep=True)
            return artifact.model_copy(deep=True)

    async def get_model_artifact(self, artifact_id: UUID) -> ModelArtifact:
        async with self._lock:
            try:
                return self._model_artifacts[artifact_id].model_copy(deep=True)
            except KeyError as exc:
                raise NotFoundError(f"model artifact not found: {artifact_id}") from exc

    async def append_model_artifact_authoring_receipt(
        self,
        receipt: ModelArtifactAuthoringReceipt,
    ) -> ModelArtifactAuthoringReceipt:
        async with self._lock:
            artifact = self._model_artifacts.get(receipt.artifact_id)
            if artifact is None or artifact.content_hash != receipt.artifact_hash:
                raise ConflictError("authoring receipt does not resolve to its artifact")
            key = (receipt.run_id, receipt.challenge_id, receipt.author_key)
            existing = self._model_artifact_authoring_receipts.get(key)
            if existing is not None:
                if existing.model_dump(exclude={"created_at"}) != receipt.model_dump(
                    exclude={"created_at"}
                ):
                    raise ConflictError("artifact author already completed this challenge")
                return existing.model_copy(deep=True)
            self._model_artifact_authoring_receipts[key] = receipt.model_copy(deep=True)
            return receipt.model_copy(deep=True)

    async def list_model_artifact_authoring_receipts(
        self,
        run_id: UUID,
        challenge_id: UUID,
    ) -> list[ModelArtifactAuthoringReceipt]:
        async with self._lock:
            return [
                receipt.model_copy(deep=True)
                for (receipt_run, receipt_challenge, _), receipt in sorted(
                    self._model_artifact_authoring_receipts.items(),
                    key=lambda item: item[0][2],
                )
                if receipt_run == run_id and receipt_challenge == challenge_id
            ]

    async def append_fitted_model(self, run_id: UUID, fitted: FittedModel) -> FittedModel:
        async with self._lock:
            artifact = self._model_artifacts.get(fitted.artifact_id)
            if artifact is None or artifact.content_hash != fitted.artifact_hash:
                raise ConflictError("fitted model does not resolve to the stored artifact")
            key = (run_id, fitted.id)
            existing = self._fitted_models.get(key)
            if existing is not None:
                if existing.state_hash != fitted.state_hash:
                    raise ConflictError(f"fitted model ID conflict: {fitted.id}")
                return existing.model_copy(deep=True)
            self._fitted_models[key] = fitted.model_copy(deep=True)
            return fitted.model_copy(deep=True)

    async def get_fitted_model(self, run_id: UUID, fitted_model_id: UUID) -> FittedModel:
        async with self._lock:
            try:
                return self._fitted_models[(run_id, fitted_model_id)].model_copy(deep=True)
            except KeyError as exc:
                raise NotFoundError(f"fitted model not found: {fitted_model_id}") from exc

    async def append_sandbox_execution(
        self,
        execution: SandboxExecutionRecord,
    ) -> SandboxExecutionRecord:
        async with self._lock:
            artifact = self._model_artifacts.get(execution.artifact_id)
            if artifact is None or artifact.content_hash != execution.artifact_hash:
                raise ConflictError("sandbox execution does not resolve to its artifact")
            if execution.fitted_model_id is not None:
                fitted = self._fitted_models.get((execution.run_id, execution.fitted_model_id))
                if fitted is None or fitted.artifact_id != execution.artifact_id:
                    raise ConflictError("sandbox execution does not resolve to its fitted model")
            key = (execution.run_id, execution.idempotency_key)
            existing_id = self._sandbox_execution_by_key.get(key)
            if existing_id is not None:
                existing = self._sandbox_executions[existing_id]
                if existing != execution:
                    raise ConflictError(
                        f"sandbox execution key conflict: {execution.idempotency_key}"
                    )
                return existing.model_copy(deep=True)
            self._sandbox_executions[execution.id] = execution.model_copy(deep=True)
            self._sandbox_execution_by_key[key] = execution.id
            return execution.model_copy(deep=True)

    async def list_sandbox_executions(self, run_id: UUID) -> list[SandboxExecutionRecord]:
        async with self._lock:
            return [
                item.model_copy(deep=True)
                for item in sorted(
                    (item for item in self._sandbox_executions.values() if item.run_id == run_id),
                    key=lambda item: (item.created_at, item.idempotency_key),
                )
            ]

    async def append_temporal_evaluation_fold(
        self,
        fold: TemporalEvaluationFold,
    ) -> TemporalEvaluationFold:
        async with self._lock:
            fitted = self._fitted_models.get((fold.run_id, fold.fitted_model_id))
            if (
                fitted is None
                or fitted.artifact_id != fold.artifact_id
                or fitted.artifact_hash != fold.artifact_hash
            ):
                raise ConflictError("evaluation fold does not resolve to its fitted model")
            existing = self._temporal_evaluation_folds.get(fold.id)
            if existing is not None:
                if existing.model_dump(exclude={"created_at"}) != fold.model_dump(
                    exclude={"created_at"}
                ):
                    raise ConflictError(f"evaluation fold ID conflict: {fold.id}")
                return existing.model_copy(deep=True)
            self._temporal_evaluation_folds[fold.id] = fold.model_copy(deep=True)
            return fold.model_copy(deep=True)

    async def list_temporal_evaluation_folds(
        self,
        run_id: UUID,
        artifact_id: UUID | None = None,
        challenge_id: UUID | None = None,
    ) -> list[TemporalEvaluationFold]:
        async with self._lock:
            return [
                item.model_copy(deep=True)
                for item in sorted(
                    (
                        item
                        for item in self._temporal_evaluation_folds.values()
                        if item.run_id == run_id
                        and (artifact_id is None or item.artifact_id == artifact_id)
                        and (
                            challenge_id is None
                            or item.challenge_id == challenge_id
                        )
                    ),
                    key=lambda item: (item.fold_index, item.id),
                )
            ]

    async def append_model_promotion_decision(
        self,
        decision: ModelPromotionDecision,
    ) -> ModelPromotionDecision:
        async with self._lock:
            champion = self._fitted_models.get((decision.run_id, decision.champion_fitted_model_id))
            if champion is None or champion.artifact_id != decision.champion_artifact_id:
                raise ConflictError("promotion champion lineage does not resolve")
            if decision.candidate_fitted_model_id is not None:
                candidate = self._fitted_models.get(
                    (decision.run_id, decision.candidate_fitted_model_id)
                )
                if candidate is None or candidate.artifact_id != decision.candidate_artifact_id:
                    raise ConflictError("promotion candidate lineage does not resolve")
            for fold_id in decision.evaluation_fold_ids:
                fold = self._temporal_evaluation_folds.get(fold_id)
                if fold is None or fold.run_id != decision.run_id:
                    raise ConflictError("promotion evaluation fold does not resolve")
                if (
                    decision.candidate_artifact_id is not None
                    and fold.artifact_id != decision.candidate_artifact_id
                ):
                    raise ConflictError("promotion fold does not evaluate the candidate")
            existing = self._promotion_decisions.get(decision.id)
            if existing is not None:
                if existing != decision:
                    raise ConflictError(f"promotion decision ID conflict: {decision.id}")
                return existing.model_copy(deep=True)
            challenge_key = (decision.run_id, decision.challenge_id)
            prior_decision_id = self._promotion_decision_by_challenge.get(challenge_key)
            if prior_decision_id is not None:
                raise ConflictError("executable-model challenge already has a promotion decision")
            self._promotion_decisions[decision.id] = decision.model_copy(deep=True)
            self._promotion_decision_by_challenge[challenge_key] = decision.id
            return decision.model_copy(deep=True)

    async def get_model_promotion_decision(
        self,
        decision_id: UUID,
    ) -> ModelPromotionDecision:
        async with self._lock:
            try:
                return self._promotion_decisions[decision_id].model_copy(deep=True)
            except KeyError as exc:
                raise NotFoundError(f"promotion decision not found: {decision_id}") from exc

    async def get_model_promotion_decision_for_challenge(
        self,
        run_id: UUID,
        challenge_id: UUID,
    ) -> ModelPromotionDecision | None:
        async with self._lock:
            decision_id = self._promotion_decision_by_challenge.get((run_id, challenge_id))
            if decision_id is None:
                return None
            return self._promotion_decisions[decision_id].model_copy(deep=True)

    async def activate_model(
        self,
        assignment: ActiveModelAssignment,
        *,
        expected_previous_sequence: int | None,
    ) -> ActiveModelAssignment:
        async with self._lock:
            existing = self._model_activations.get(assignment.id)
            if existing is not None:
                if existing != assignment:
                    raise ConflictError(f"model activation ID conflict: {assignment.id}")
                return existing.model_copy(deep=True)
            previous_ids = self._model_activations_by_run.get(assignment.run_id, [])
            previous = self._model_activations[previous_ids[-1]] if previous_ids else None
            previous_sequence = previous.sequence if previous else None
            if previous_sequence != expected_previous_sequence:
                raise ConflictError(
                    "active-model sequence conflict: "
                    f"expected {expected_previous_sequence}, got {previous_sequence}"
                )
            expected_sequence = 1 if previous is None else previous.sequence + 1
            if assignment.sequence != expected_sequence:
                raise ConflictError(
                    f"active-model sequence must be {expected_sequence}, got {assignment.sequence}"
                )
            decision = self._promotion_decisions.get(assignment.promotion_decision_id)
            if decision is None or decision.run_id != assignment.run_id:
                raise ConflictError("activation does not resolve to a promotion decision")
            if decision.disposition != PromotionDisposition.PROMOTED:
                raise ConflictError("only a promoted candidate can become active")
            if (
                decision.candidate_artifact_id != assignment.artifact_id
                or decision.candidate_fitted_model_id != assignment.fitted_model_id
            ):
                raise ConflictError("activation target differs from the promoted candidate")
            fitted = self._fitted_models.get((assignment.run_id, assignment.fitted_model_id))
            artifact = self._model_artifacts.get(assignment.artifact_id)
            if (
                fitted is None
                or artifact is None
                or fitted.state_hash != assignment.fitted_state_hash
                or artifact.content_hash != assignment.artifact_hash
                or fitted.artifact_id != assignment.artifact_id
            ):
                raise ConflictError("activation content hashes do not resolve")
            if assignment.promotion_decision_id in self._activation_by_promotion:
                raise ConflictError("promotion decision is already activated")
            self._model_activations[assignment.id] = assignment.model_copy(deep=True)
            self._model_activations_by_run.setdefault(assignment.run_id, []).append(assignment.id)
            self._activation_by_promotion[assignment.promotion_decision_id] = assignment.id
            return assignment.model_copy(deep=True)

    async def get_active_model(self, run_id: UUID) -> ActiveModelAssignment | None:
        async with self._lock:
            assignment_ids = self._model_activations_by_run.get(run_id, [])
            if not assignment_ids:
                return None
            return self._model_activations[assignment_ids[-1]].model_copy(deep=True)

    async def list_model_activations(self, run_id: UUID) -> list[ActiveModelAssignment]:
        async with self._lock:
            return [
                self._model_activations[item_id].model_copy(deep=True)
                for item_id in self._model_activations_by_run.get(run_id, [])
            ]

    async def get_latest_world_model(self, run_id: UUID) -> WorldModelVersion | None:
        async with self._lock:
            model_ids = self._world_models_by_run.get(run_id, [])
            if not model_ids:
                return None
            return self._world_models[model_ids[-1]].model_copy(deep=True)

    async def get_world_model(self, model_id: UUID) -> WorldModelVersion:
        async with self._lock:
            try:
                return self._world_models[model_id].model_copy(deep=True)
            except KeyError as exc:
                raise NotFoundError(f"world model not found: {model_id}") from exc

    async def list_world_models(self, run_id: UUID) -> list[WorldModelVersion]:
        async with self._lock:
            return [
                self._world_models[model_id].model_copy(deep=True)
                for model_id in self._world_models_by_run.get(run_id, [])
            ]

    async def list_world_model_relationships(
        self,
        model_id: UUID,
    ) -> list[WorldModelRelationship]:
        model = await self.get_world_model(model_id)
        return [
            relationship.model_copy(deep=True)
            for relationship in sorted(model.relationships, key=lambda item: item.key)
        ]

    async def append_world_model(
        self,
        world_model: WorldModelVersion,
        *,
        expected_latest_version: int | None,
    ) -> WorldModelVersion:
        async with self._lock:
            existing = self._world_models.get(world_model.id)
            if existing is not None:
                if existing != world_model:
                    raise ConflictError(f"world model ID conflict: {world_model.id}")
                return existing.model_copy(deep=True)

            model_ids = self._world_models_by_run.get(world_model.run_id, [])
            latest = self._world_models[model_ids[-1]] if model_ids else None
            latest_version = latest.version if latest else None
            if latest_version != expected_latest_version:
                raise ConflictError(
                    "world-model version conflict: "
                    f"expected {expected_latest_version}, got {latest_version}"
                )
            expected_new_version = 1 if latest is None else latest.version + 1
            if world_model.version != expected_new_version:
                raise ConflictError(
                    f"world-model version must be {expected_new_version}, got {world_model.version}"
                )
            self._world_models[world_model.id] = world_model.model_copy(deep=True)
            self._world_models_by_run.setdefault(world_model.run_id, []).append(world_model.id)
            return world_model.model_copy(deep=True)

    async def append_prediction(
        self,
        prediction: PredictionLedgerEntry,
    ) -> PredictionLedgerEntry:
        async with self._lock:
            existing = self._predictions.get(prediction.id)
            if existing is not None:
                if existing != prediction:
                    raise ConflictError(f"prediction ID conflict: {prediction.id}")
                return existing.model_copy(deep=True)
            self._predictions[prediction.id] = prediction.model_copy(deep=True)
            self._predictions_by_run.setdefault(prediction.run_id, []).append(prediction.id)
            return prediction.model_copy(deep=True)

    async def get_prediction(self, prediction_id: UUID) -> PredictionLedgerEntry:
        async with self._lock:
            try:
                return self._predictions[prediction_id].model_copy(deep=True)
            except KeyError as exc:
                raise NotFoundError(f"prediction not found: {prediction_id}") from exc

    async def list_predictions(self, run_id: UUID) -> list[PredictionLedgerEntry]:
        async with self._lock:
            return [
                self._predictions[prediction_id].model_copy(deep=True)
                for prediction_id in self._predictions_by_run.get(run_id, [])
            ]

    async def get_prediction_outcome(self, target_id: UUID) -> PredictionOutcome | None:
        async with self._lock:
            outcome = self._prediction_outcomes.get(target_id)
            return outcome.model_copy(deep=True) if outcome else None

    async def list_prediction_outcomes(self, run_id: UUID) -> list[PredictionOutcome]:
        async with self._lock:
            return [
                outcome.model_copy(deep=True)
                for outcome in self._prediction_outcomes.values()
                if outcome.run_id == run_id
            ]

    async def append_prediction_outcome(
        self,
        outcome: PredictionOutcome,
    ) -> PredictionOutcome:
        async with self._lock:
            existing = self._prediction_outcomes.get(outcome.target_id)
            if existing is not None:
                if existing != outcome:
                    raise ConflictError(f"prediction target outcome conflict: {outcome.target_id}")
                return existing.model_copy(deep=True)
            self._prediction_outcomes[outcome.target_id] = outcome.model_copy(deep=True)
            return outcome.model_copy(deep=True)

    async def append_portfolio_revision(
        self,
        revision: StrategicPortfolioRevision,
    ) -> StrategicPortfolioRevision:
        async with self._lock:
            existing = self._portfolio_revisions.get(revision.id)
            if existing is not None:
                if existing.model_dump(exclude={"created_at"}) != revision.model_dump(
                    exclude={"created_at"}
                ):
                    raise ConflictError(f"portfolio revision conflict: {revision.id}")
                return existing.model_copy(deep=True)
            revision_ids = self._portfolio_revisions_by_run.get(revision.run_id, [])
            latest = (
                self._portfolio_revisions[revision_ids[-1]] if revision_ids else None
            )
            if latest is None:
                if revision.revision != 1:
                    raise ConflictError(
                        "portfolio history must start at revision 1: "
                        f"got revision {revision.revision}"
                    )
            else:
                if revision.revision != latest.revision + 1:
                    raise ConflictError(
                        "portfolio revision must extend the latest revision: "
                        f"latest={latest.revision}, got={revision.revision}"
                    )
                if revision.portfolio.prior_portfolio_hash != latest.portfolio_hash:
                    raise ConflictError(
                        f"portfolio hash chain broken at revision {revision.revision}"
                    )
            self._portfolio_revisions[revision.id] = revision.model_copy(deep=True)
            self._portfolio_revisions_by_run.setdefault(revision.run_id, []).append(
                revision.id
            )
            return revision.model_copy(deep=True)

    async def list_portfolio_revisions(
        self,
        run_id: UUID,
    ) -> list[StrategicPortfolioRevision]:
        async with self._lock:
            return [
                self._portfolio_revisions[revision_id].model_copy(deep=True)
                for revision_id in self._portfolio_revisions_by_run.get(run_id, [])
            ]

    async def get_latest_portfolio_revision(
        self,
        run_id: UUID,
    ) -> StrategicPortfolioRevision | None:
        async with self._lock:
            revision_ids = self._portfolio_revisions_by_run.get(run_id, [])
            if not revision_ids:
                return None
            return self._portfolio_revisions[revision_ids[-1]].model_copy(deep=True)

    async def append_experiment_outcome(
        self,
        outcome: ExperimentOutcome,
    ) -> ExperimentOutcome:
        async with self._lock:
            existing = self._experiment_outcomes.get(outcome.id)
            if existing is not None:
                if existing != outcome:
                    raise ConflictError(f"experiment outcome conflict: {outcome.id}")
                return existing.model_copy(deep=True)
            self._experiment_outcomes[outcome.id] = outcome.model_copy(deep=True)
            self._experiment_outcomes_by_run.setdefault(outcome.run_id, []).append(
                outcome.id
            )
            return outcome.model_copy(deep=True)

    async def list_experiment_outcomes(self, run_id: UUID) -> list[ExperimentOutcome]:
        async with self._lock:
            return [
                self._experiment_outcomes[outcome_id].model_copy(deep=True)
                for outcome_id in self._experiment_outcomes_by_run.get(run_id, [])
            ]

    async def list_commitment_experiment_outcomes(
        self,
        run_id: UUID,
        commitment_id: str,
    ) -> list[ExperimentOutcome]:
        async with self._lock:
            return [
                self._experiment_outcomes[outcome_id].model_copy(deep=True)
                for outcome_id in self._experiment_outcomes_by_run.get(run_id, [])
                if self._experiment_outcomes[outcome_id].commitment_id == commitment_id
            ]

    async def append_commitment_review(
        self,
        review: CommitmentReview,
    ) -> CommitmentReview:
        async with self._lock:
            existing = self._commitment_reviews.get(review.id)
            if existing is not None:
                if existing.model_dump(exclude={"created_at"}) != review.model_dump(
                    exclude={"created_at"}
                ):
                    raise ConflictError(f"commitment review conflict: {review.id}")
                return existing.model_copy(deep=True)
            self._commitment_reviews[review.id] = review.model_copy(deep=True)
            self._commitment_reviews_by_run.setdefault(review.run_id, []).append(
                review.id
            )
            return review.model_copy(deep=True)

    async def list_commitment_reviews(
        self,
        run_id: UUID,
        commitment_id: str | None = None,
    ) -> list[CommitmentReview]:
        async with self._lock:
            return [
                self._commitment_reviews[review_id].model_copy(deep=True)
                for review_id in self._commitment_reviews_by_run.get(run_id, [])
                if commitment_id is None
                or self._commitment_reviews[review_id].commitment_id == commitment_id
            ]

    async def append_candidate_evaluation_set(
        self,
        evaluation_set: CandidateEvaluationSet,
    ) -> CandidateEvaluationSet:
        async with self._lock:
            existing = self._candidate_evaluation_sets.get(evaluation_set.id)
            if existing is not None:
                if existing.set_hash != evaluation_set.set_hash:
                    raise ConflictError(
                        f"candidate evaluation set conflict: {evaluation_set.id}"
                    )
                return existing.model_copy(deep=True)
            week_key = (evaluation_set.run_id, evaluation_set.week)
            if week_key in self._evaluation_set_by_week:
                raise ConflictError(
                    "candidate evaluation set already exists for week "
                    f"{evaluation_set.week}"
                )
            self._candidate_evaluation_sets[evaluation_set.id] = (
                evaluation_set.model_copy(deep=True)
            )
            self._evaluation_set_by_week[week_key] = evaluation_set.id
            return evaluation_set.model_copy(deep=True)

    async def get_candidate_evaluation_set(
        self,
        run_id: UUID,
        week: int,
    ) -> CandidateEvaluationSet | None:
        async with self._lock:
            set_id = self._evaluation_set_by_week.get((run_id, week))
            if set_id is None:
                return None
            return self._candidate_evaluation_sets[set_id].model_copy(deep=True)

    async def append_executive_choice(self, choice: ExecutiveChoice) -> ExecutiveChoice:
        async with self._lock:
            existing = self._executive_choices.get(choice.id)
            if existing is not None:
                if existing.model_dump(exclude={"created_at"}) != choice.model_dump(
                    exclude={"created_at"}
                ):
                    raise ConflictError(f"executive choice conflict: {choice.id}")
                return existing.model_copy(deep=True)
            week_key = (choice.run_id, choice.week)
            if week_key in self._executive_choice_by_week:
                raise ConflictError(
                    f"executive choice already exists for week {choice.week}"
                )
            self._executive_choices[choice.id] = choice.model_copy(deep=True)
            self._executive_choice_by_week[week_key] = choice.id
            return choice.model_copy(deep=True)

    async def get_executive_choice(
        self,
        run_id: UUID,
        week: int,
    ) -> ExecutiveChoice | None:
        async with self._lock:
            choice_id = self._executive_choice_by_week.get((run_id, week))
            if choice_id is None:
                return None
            return self._executive_choices[choice_id].model_copy(deep=True)

    async def append_insight_record(self, insight: InsightRecord) -> InsightRecord:
        async with self._lock:
            existing = self._insight_records.get(insight.id)
            if existing is not None:
                return existing.model_copy(deep=True)
            self._insight_records[insight.id] = insight.model_copy(deep=True)
            self._insight_records_by_run.setdefault(insight.run_id, []).append(
                insight.id
            )
            return insight.model_copy(deep=True)

    async def list_insight_records(self, run_id: UUID) -> list[InsightRecord]:
        async with self._lock:
            return [
                self._insight_records[insight_id].model_copy(deep=True)
                for insight_id in self._insight_records_by_run.get(run_id, [])
            ]

    async def append_model_health_signal(
        self,
        signal: ModelHealthSignal,
    ) -> ModelHealthSignal:
        async with self._lock:
            existing = self._model_health_signals.get(signal.id)
            if existing is not None:
                if existing != signal:
                    raise ConflictError(f"model-health signal ID conflict: {signal.id}")
                return existing.model_copy(deep=True)
            self._model_health_signals[signal.id] = signal.model_copy(deep=True)
            self._model_health_signals_by_run.setdefault(signal.run_id, []).append(signal.id)
            return signal.model_copy(deep=True)

    async def get_model_health_signal(self, signal_id: UUID) -> ModelHealthSignal:
        async with self._lock:
            try:
                return self._model_health_signals[signal_id].model_copy(deep=True)
            except KeyError as exc:
                raise NotFoundError(f"model-health signal not found: {signal_id}") from exc

    async def list_model_health_signals(self, run_id: UUID) -> list[ModelHealthSignal]:
        async with self._lock:
            return [
                self._model_health_signals[signal_id].model_copy(deep=True)
                for signal_id in self._model_health_signals_by_run.get(run_id, [])
            ]

    async def get_model_challenge(self, challenge_id: UUID) -> ModelChallengeRecord | None:
        async with self._lock:
            challenge = self._model_challenges.get(challenge_id)
            return challenge.model_copy(deep=True) if challenge else None

    async def save_model_challenge(self, challenge: ModelChallengeRecord) -> ModelChallengeRecord:
        async with self._lock:
            existing = self._model_challenges.get(challenge.id)
            if existing is not None:
                identity = (
                    existing.run_id,
                    existing.health_signal_id,
                    existing.base_model_version_id,
                    existing.requested_builders,
                    existing.created_at,
                )
                incoming_identity = (
                    challenge.run_id,
                    challenge.health_signal_id,
                    challenge.base_model_version_id,
                    challenge.requested_builders,
                    challenge.created_at,
                )
                if identity != incoming_identity:
                    raise ConflictError(f"model challenge ID conflict: {challenge.id}")
                if existing.status in {
                    ModelChallengeStatus.COMPLETED,
                    ModelChallengeStatus.FAILED,
                }:
                    return existing.model_copy(deep=True)
            self._model_challenges[challenge.id] = challenge.model_copy(deep=True)
            return challenge.model_copy(deep=True)

    async def append_model_challenge_package(
        self, package: ModelChallengePackage
    ) -> ModelChallengePackage:
        async with self._lock:
            existing = self._model_challenge_packages.get(package.challenge_id)
            if existing is not None:
                if existing != package:
                    raise ConflictError(f"model challenge package conflict: {package.challenge_id}")
                return existing.model_copy(deep=True)
            self._model_challenge_packages[package.challenge_id] = package.model_copy(deep=True)
            return package.model_copy(deep=True)

    async def get_model_challenge_package(self, challenge_id: UUID) -> ModelChallengePackage | None:
        async with self._lock:
            package = self._model_challenge_packages.get(challenge_id)
            return package.model_copy(deep=True) if package else None

    async def append_model_builder_proposal(
        self, proposal: ModelBuilderProposal
    ) -> ModelBuilderProposal:
        async with self._lock:
            existing = self._model_builder_proposals.get(proposal.id)
            if existing is not None:
                if existing != proposal:
                    raise ConflictError(f"model-builder proposal conflict: {proposal.id}")
                return existing.model_copy(deep=True)
            self._model_builder_proposals[proposal.id] = proposal.model_copy(deep=True)
            return proposal.model_copy(deep=True)

    async def list_model_builder_proposals(self, challenge_id: UUID) -> list[ModelBuilderProposal]:
        async with self._lock:
            return sorted(
                (
                    item.model_copy(deep=True)
                    for item in self._model_builder_proposals.values()
                    if item.challenge_id == challenge_id
                ),
                key=lambda item: (item.builder_name, str(item.id)),
            )

    async def append_hypothesis_backtest(
        self, result: HypothesisBacktestResult
    ) -> HypothesisBacktestResult:
        async with self._lock:
            existing = self._hypothesis_backtests.get(result.id)
            if existing is not None:
                if existing != result:
                    raise ConflictError(f"hypothesis backtest conflict: {result.id}")
                return existing.model_copy(deep=True)
            self._hypothesis_backtests[result.id] = result.model_copy(deep=True)
            return result.model_copy(deep=True)

    async def list_hypothesis_backtests(self, challenge_id: UUID) -> list[HypothesisBacktestResult]:
        async with self._lock:
            return sorted(
                (
                    item.model_copy(deep=True)
                    for item in self._hypothesis_backtests.values()
                    if item.challenge_id == challenge_id
                ),
                key=lambda item: (item.candidate_score, str(item.id)),
            )

    async def append_model_builder_call(
        self, receipt: ModelBuilderCallReceipt
    ) -> ModelBuilderCallReceipt:
        async with self._lock:
            existing = self._model_builder_calls.get(receipt.id)
            if existing is not None:
                if existing != receipt:
                    raise ConflictError(f"model-builder call conflict: {receipt.id}")
                return existing.model_copy(deep=True)
            self._model_builder_calls[receipt.id] = receipt.model_copy(deep=True)
            return receipt.model_copy(deep=True)

    async def list_model_builder_calls(self, challenge_id: UUID) -> list[ModelBuilderCallReceipt]:
        async with self._lock:
            return sorted(
                (
                    item.model_copy(deep=True)
                    for item in self._model_builder_calls.values()
                    if item.challenge_id == challenge_id
                ),
                key=lambda item: (item.builder_name, item.attempt),
            )

    async def append_model_challenge_decision(
        self, decision: ModelChallengeDecision
    ) -> ModelChallengeDecision:
        async with self._lock:
            decision_id = self._model_challenge_decision_by_challenge.get(decision.challenge_id)
            if decision_id is not None:
                existing = self._model_challenge_decisions[decision_id]
                if existing != decision:
                    raise ConflictError(
                        f"model-challenge decision conflict: {decision.challenge_id}"
                    )
                return existing.model_copy(deep=True)
            self._model_challenge_decisions[decision.id] = decision.model_copy(deep=True)
            self._model_challenge_decision_by_challenge[decision.challenge_id] = decision.id
            return decision.model_copy(deep=True)

    async def get_model_challenge_decision(
        self, challenge_id: UUID
    ) -> ModelChallengeDecision | None:
        async with self._lock:
            decision_id = self._model_challenge_decision_by_challenge.get(challenge_id)
            if decision_id is None:
                return None
            return self._model_challenge_decisions[decision_id].model_copy(deep=True)

    async def create_run(self, run: RunRecord) -> RunRecord:
        async with self._lock:
            if run.id in self._runs:
                raise ConflictError(f"run already exists: {run.id}")
            self._runs[run.id] = run.model_copy(deep=True)
            return run.model_copy(deep=True)

    async def get_run(self, run_id: UUID) -> RunRecord:
        async with self._lock:
            try:
                return self._runs[run_id].model_copy(deep=True)
            except KeyError as exc:
                raise NotFoundError(f"run not found: {run_id}") from exc

    async def save_run(self, run: RunRecord, *, expected_version: int) -> RunRecord:
        async with self._lock:
            current = self._runs.get(run.id)
            if current is None:
                raise NotFoundError(f"run not found: {run.id}")
            if current.version != expected_version:
                raise ConflictError(
                    f"run version conflict: expected {expected_version}, got {current.version}"
                )
            saved = run.model_copy(
                update={"version": expected_version + 1, "updated_at": utc_now()},
                deep=True,
            )
            self._runs[run.id] = saved
            return saved.model_copy(deep=True)

    async def claim_run_lease(
        self,
        run_id: UUID,
        owner_id: str,
        *,
        now: datetime,
        ttl_seconds: int,
    ) -> RunLease | None:
        async with self._lock:
            if run_id not in self._runs:
                raise NotFoundError(f"run not found: {run_id}")
            existing = self._run_leases.get(run_id)
            if existing is not None and existing.expires_at > now and existing.owner_id != owner_id:
                return None
            lease = RunLease(
                run_id=run_id,
                owner_id=owner_id,
                acquired_at=now,
                heartbeat_at=now,
                expires_at=now + timedelta(seconds=ttl_seconds),
            )
            self._run_leases[run_id] = lease
            return lease.model_copy(deep=True)

    async def renew_run_lease(
        self,
        run_id: UUID,
        token: UUID,
        *,
        now: datetime,
        ttl_seconds: int,
    ) -> RunLease | None:
        async with self._lock:
            existing = self._run_leases.get(run_id)
            if existing is None or existing.token != token or existing.expires_at <= now:
                return None
            renewed = existing.model_copy(
                update={
                    "heartbeat_at": now,
                    "expires_at": now + timedelta(seconds=ttl_seconds),
                },
                deep=True,
            )
            self._run_leases[run_id] = renewed
            return renewed.model_copy(deep=True)

    async def release_run_lease(self, run_id: UUID, token: UUID) -> bool:
        async with self._lock:
            existing = self._run_leases.get(run_id)
            if existing is None or existing.token != token:
                return False
            del self._run_leases[run_id]
            return True

    async def get_decision_for_week(self, run_id: UUID, week: int) -> DecisionRecord | None:
        async with self._lock:
            decision_id = self._decision_by_week.get((run_id, week))
            if decision_id is None:
                return None
            return self._decisions[decision_id].model_copy(deep=True)

    async def get_decision(self, run_id: UUID, decision_id: UUID) -> DecisionRecord:
        async with self._lock:
            decision = self._decisions.get(decision_id)
            if decision is None or decision.run_id != run_id:
                raise NotFoundError(f"decision not found: {decision_id}")
            return decision.model_copy(deep=True)

    async def list_decisions(self, run_id: UUID) -> list[DecisionRecord]:
        async with self._lock:
            decisions = [
                decision for decision in self._decisions.values() if decision.run_id == run_id
            ]
            return [
                decision.model_copy(deep=True)
                for decision in sorted(decisions, key=lambda item: item.week)
            ]

    async def list_candidate_simulations(
        self,
        run_id: UUID,
        decision_id: UUID,
    ) -> list[CandidateEvaluationRecord]:
        decision = await self.get_decision(run_id, decision_id)
        return [
            candidate.model_copy(deep=True)
            for candidate in sorted(
                decision.candidate_evaluations,
                key=lambda item: item.robust_utility,
                reverse=True,
            )
        ]

    async def save_decision(self, decision: DecisionRecord) -> DecisionRecord:
        async with self._lock:
            key = (decision.run_id, decision.week)
            existing_id = self._decision_by_week.get(key)
            if existing_id is not None:
                return self._decisions[existing_id].model_copy(deep=True)
            self._decisions[decision.id] = decision.model_copy(deep=True)
            self._decision_by_week[key] = decision.id
            return decision.model_copy(deep=True)

    async def update_decision(self, decision: DecisionRecord) -> DecisionRecord:
        async with self._lock:
            if decision.id not in self._decisions:
                raise NotFoundError(f"decision not found: {decision.id}")
            self._decisions[decision.id] = decision.model_copy(deep=True)
            return decision.model_copy(deep=True)

    async def get_receipt(self, run_id: UUID, idempotency_key: str) -> ActionReceipt | None:
        async with self._lock:
            receipt_id = self._receipt_by_key.get((run_id, idempotency_key))
            if receipt_id is None:
                return None
            return self._receipts[receipt_id].model_copy(deep=True)

    async def save_receipt(self, receipt: ActionReceipt) -> ActionReceipt:
        async with self._lock:
            key = (receipt.run_id, receipt.idempotency_key)
            existing_id = self._receipt_by_key.get(key)
            if existing_id is not None:
                return self._receipts[existing_id].model_copy(deep=True)
            self._receipts[receipt.id] = receipt.model_copy(deep=True)
            self._receipt_by_key[key] = receipt.id
            return receipt.model_copy(deep=True)

    async def list_receipts(self, decision_id: UUID) -> list[ActionReceipt]:
        async with self._lock:
            return [
                receipt.model_copy(deep=True)
                for receipt in self._receipts.values()
                if receipt.decision_id == decision_id
            ]

    async def append_event(self, event: EventRecord) -> EventRecord:
        async with self._lock:
            events = self._events.setdefault(event.run_id, [])
            existing = next((item for item in events if item.id == event.id), None)
            if existing is not None:
                if existing.model_copy(update={"sequence": None}) != event:
                    raise ConflictError(f"event ID conflict: {event.id}")
                return existing.model_copy(deep=True)
            saved = event.model_copy(update={"sequence": len(events) + 1}, deep=True)
            events.append(saved)
            return saved.model_copy(deep=True)

    async def list_events(self, run_id: UUID) -> list[EventRecord]:
        async with self._lock:
            return [event.model_copy(deep=True) for event in self._events.get(run_id, [])]

    async def get_operation(self, run_id: UUID, request_id: str) -> StepOperation | None:
        async with self._lock:
            operation = self._operations.get((run_id, request_id))
            return operation.model_copy(deep=True) if operation else None

    async def start_operation(self, operation: StepOperation) -> StepOperation:
        async with self._lock:
            key = (operation.run_id, operation.request_id)
            existing = self._operations.get(key)
            if existing is None:
                self._operations[key] = operation.model_copy(deep=True)
                return operation.model_copy(deep=True)
            if existing.status == OperationStatus.COMPLETED:
                return existing.model_copy(deep=True)
            if existing.status == OperationStatus.STARTED:
                raise OperationInProgressError(
                    f"operation already in progress: {operation.request_id}"
                )
            restarted = existing.model_copy(
                update={
                    "status": OperationStatus.STARTED,
                    "error": None,
                    "attempts": existing.attempts + 1,
                    "updated_at": utc_now(),
                },
                deep=True,
            )
            self._operations[key] = restarted
            return restarted.model_copy(deep=True)

    async def complete_operation(
        self, run_id: UUID, request_id: str, result: StepResult
    ) -> StepOperation:
        return await self._finish_operation(
            run_id,
            request_id,
            status=OperationStatus.COMPLETED,
            result=result.model_dump(mode="json"),
            error=None,
        )

    async def fail_operation(self, run_id: UUID, request_id: str, error: str) -> StepOperation:
        return await self._finish_operation(
            run_id,
            request_id,
            status=OperationStatus.FAILED,
            result=None,
            error=error,
        )

    async def _finish_operation(
        self,
        run_id: UUID,
        request_id: str,
        *,
        status: OperationStatus,
        result: dict[str, Any] | None,
        error: str | None,
    ) -> StepOperation:
        async with self._lock:
            key = (run_id, request_id)
            existing = self._operations.get(key)
            if existing is None:
                raise NotFoundError(f"operation not found: {request_id}")
            saved = existing.model_copy(
                update={
                    "status": status,
                    "result": result,
                    "error": error,
                    "updated_at": utc_now(),
                },
                deep=True,
            )
            self._operations[key] = saved
            return saved.model_copy(deep=True)


class SupabaseRunRepository:
    """PostgREST repository intended for server-side use with a secret key."""

    def __init__(
        self,
        *,
        url: str,
        secret_key: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = f"{url.rstrip('/')}/rest/v1"
        self._headers = {
            "apikey": secret_key,
            "Authorization": f"Bearer {secret_key}",
            "Content-Type": "application/json",
        }
        self._client = client or httpx.AsyncClient(timeout=30)

    async def append_model_artifact(self, artifact: ModelArtifact) -> ModelArtifact:
        try:
            existing = await self.get_model_artifact(artifact.id)
        except NotFoundError:
            existing = None
        if existing is not None:
            if existing.content_hash != artifact.content_hash:
                raise ConflictError(f"model artifact ID conflict: {artifact.id}")
            return existing
        try:
            rows = await self._request(
                "POST",
                "lithops_model_artifacts",
                json={
                    "id": str(artifact.id),
                    "content_hash": artifact.content_hash,
                    "runtime_kind": artifact.runtime_kind,
                    "parent_artifact_id": (
                        str(artifact.parent_artifact_id)
                        if artifact.parent_artifact_id is not None
                        else None
                    ),
                    "payload": artifact.model_dump(mode="json"),
                    "created_at": artifact.created_at.isoformat(),
                },
                prefer="return=representation",
            )
        except ConflictError:
            return await self.get_model_artifact(artifact.id)
        return ModelArtifact.model_validate(rows[0]["payload"])

    async def get_model_artifact(self, artifact_id: UUID) -> ModelArtifact:
        rows = await self._request(
            "GET",
            "lithops_model_artifacts",
            params={"id": f"eq.{artifact_id}", "select": "payload"},
        )
        if not rows:
            raise NotFoundError(f"model artifact not found: {artifact_id}")
        return ModelArtifact.model_validate(rows[0]["payload"])

    async def append_model_artifact_authoring_receipt(
        self,
        receipt: ModelArtifactAuthoringReceipt,
    ) -> ModelArtifactAuthoringReceipt:
        artifact = await self.get_model_artifact(receipt.artifact_id)
        if artifact.content_hash != receipt.artifact_hash:
            raise ConflictError("authoring receipt does not resolve to its artifact")
        return await self._append_challenge_artifact(
            table="lithops_model_artifact_authoring_receipts",
            identity_column="id",
            identity=receipt.id,
            payload_model=receipt,
            row={
                "id": str(receipt.id),
                "challenge_id": str(receipt.challenge_id),
                "run_id": str(receipt.run_id),
                "author_key": receipt.author_key,
                "artifact_id": str(receipt.artifact_id),
                "artifact_hash": receipt.artifact_hash,
                "input_hash": receipt.input_hash,
                "payload": receipt.model_dump(mode="json"),
                "created_at": receipt.created_at.isoformat(),
            },
            model_type=ModelArtifactAuthoringReceipt,
        )

    async def list_model_artifact_authoring_receipts(
        self,
        run_id: UUID,
        challenge_id: UUID,
    ) -> list[ModelArtifactAuthoringReceipt]:
        rows = await self._request(
            "GET",
            "lithops_model_artifact_authoring_receipts",
            params={
                "run_id": f"eq.{run_id}",
                "challenge_id": f"eq.{challenge_id}",
                "select": "payload",
                "order": "author_key.asc",
            },
        )
        return [ModelArtifactAuthoringReceipt.model_validate(row["payload"]) for row in rows]

    async def append_fitted_model(self, run_id: UUID, fitted: FittedModel) -> FittedModel:
        artifact = await self.get_model_artifact(fitted.artifact_id)
        if artifact.content_hash != fitted.artifact_hash:
            raise ConflictError("fitted model does not resolve to the stored artifact")
        rows = await self._request(
            "GET",
            "lithops_fitted_models",
            params={
                "run_id": f"eq.{run_id}",
                "id": f"eq.{fitted.id}",
                "select": "payload",
            },
        )
        if rows:
            existing = FittedModel.model_validate(rows[0]["payload"])
            if existing.state_hash != fitted.state_hash:
                raise ConflictError(f"fitted model ID conflict: {fitted.id}")
            return existing
        try:
            rows = await self._request(
                "POST",
                "lithops_fitted_models",
                json={
                    "id": str(fitted.id),
                    "run_id": str(run_id),
                    "artifact_id": str(fitted.artifact_id),
                    "artifact_hash": fitted.artifact_hash,
                    "state_hash": fitted.state_hash,
                    "training_start_day": fitted.training_start_day,
                    "training_end_day": fitted.training_end_day,
                    "payload": fitted.model_dump(mode="json"),
                    "created_at": fitted.created_at.isoformat(),
                },
                prefer="return=representation",
            )
        except ConflictError:
            return await self.get_fitted_model(run_id, fitted.id)
        return FittedModel.model_validate(rows[0]["payload"])

    async def get_fitted_model(self, run_id: UUID, fitted_model_id: UUID) -> FittedModel:
        rows = await self._request(
            "GET",
            "lithops_fitted_models",
            params={
                "run_id": f"eq.{run_id}",
                "id": f"eq.{fitted_model_id}",
                "select": "payload",
            },
        )
        if not rows:
            raise NotFoundError(f"fitted model not found: {fitted_model_id}")
        return FittedModel.model_validate(rows[0]["payload"])

    async def append_sandbox_execution(
        self,
        execution: SandboxExecutionRecord,
    ) -> SandboxExecutionRecord:
        artifact = await self.get_model_artifact(execution.artifact_id)
        if artifact.content_hash != execution.artifact_hash:
            raise ConflictError("sandbox execution does not resolve to its artifact")
        if execution.fitted_model_id is not None:
            fitted = await self.get_fitted_model(
                execution.run_id,
                execution.fitted_model_id,
            )
            if fitted.artifact_id != execution.artifact_id:
                raise ConflictError("sandbox execution does not resolve to its fitted model")
        rows = await self._request(
            "GET",
            "lithops_sandbox_executions",
            params={
                "run_id": f"eq.{execution.run_id}",
                "idempotency_key": f"eq.{execution.idempotency_key}",
                "select": "payload",
            },
        )
        if rows:
            existing = SandboxExecutionRecord.model_validate(rows[0]["payload"])
            if existing != execution:
                raise ConflictError(f"sandbox execution key conflict: {execution.idempotency_key}")
            return existing
        try:
            rows = await self._request(
                "POST",
                "lithops_sandbox_executions",
                json={
                    "id": str(execution.id),
                    "run_id": str(execution.run_id),
                    "idempotency_key": execution.idempotency_key,
                    "artifact_id": str(execution.artifact_id),
                    "fitted_model_id": (
                        str(execution.fitted_model_id)
                        if execution.fitted_model_id is not None
                        else None
                    ),
                    "operation": execution.operation,
                    "status": execution.status,
                    "input_hash": execution.input_hash,
                    "output_hash": execution.output_hash,
                    "payload": execution.model_dump(mode="json"),
                    "created_at": execution.created_at.isoformat(),
                },
                prefer="return=representation",
            )
        except ConflictError:
            rows = await self._request(
                "GET",
                "lithops_sandbox_executions",
                params={
                    "run_id": f"eq.{execution.run_id}",
                    "idempotency_key": f"eq.{execution.idempotency_key}",
                    "select": "payload",
                },
            )
            if not rows:
                raise
            existing = SandboxExecutionRecord.model_validate(rows[0]["payload"])
            if existing != execution:
                raise ConflictError(
                    f"sandbox execution key conflict: {execution.idempotency_key}"
                ) from None
            return existing
        return SandboxExecutionRecord.model_validate(rows[0]["payload"])

    async def list_sandbox_executions(self, run_id: UUID) -> list[SandboxExecutionRecord]:
        rows = await self._request(
            "GET",
            "lithops_sandbox_executions",
            params={
                "run_id": f"eq.{run_id}",
                "select": "payload",
                "order": "created_at.asc",
            },
        )
        return [SandboxExecutionRecord.model_validate(row["payload"]) for row in rows]

    async def append_temporal_evaluation_fold(
        self,
        fold: TemporalEvaluationFold,
    ) -> TemporalEvaluationFold:
        fitted = await self.get_fitted_model(fold.run_id, fold.fitted_model_id)
        if fitted.artifact_id != fold.artifact_id or fitted.artifact_hash != fold.artifact_hash:
            raise ConflictError("evaluation fold does not resolve to its fitted model")
        return await self._append_challenge_artifact(
            table="lithops_temporal_evaluation_folds",
            identity_column="id",
            identity=fold.id,
            payload_model=fold,
            row={
                "id": str(fold.id),
                "run_id": str(fold.run_id),
                "artifact_id": str(fold.artifact_id),
                "fitted_model_id": str(fold.fitted_model_id),
                "challenge_id": str(fold.challenge_id),
                "fold_index": fold.fold_index,
                "evaluation_seed": fold.evaluation_seed,
                "total_score": fold.total_score,
                "invariant_gate_passed": fold.invariant_gate_passed,
                "payload": fold.model_dump(mode="json"),
                "created_at": fold.created_at.isoformat(),
            },
            model_type=TemporalEvaluationFold,
        )

    async def list_temporal_evaluation_folds(
        self,
        run_id: UUID,
        artifact_id: UUID | None = None,
        challenge_id: UUID | None = None,
    ) -> list[TemporalEvaluationFold]:
        params = {
            "run_id": f"eq.{run_id}",
            "select": "payload",
            "order": "fold_index.asc",
        }
        if artifact_id is not None:
            params["artifact_id"] = f"eq.{artifact_id}"
        if challenge_id is not None:
            params["challenge_id"] = f"eq.{challenge_id}"
        rows = await self._request_all_pages(
            "lithops_temporal_evaluation_folds",
            params=params,
        )
        return [TemporalEvaluationFold.model_validate(row["payload"]) for row in rows]

    async def append_model_promotion_decision(
        self,
        decision: ModelPromotionDecision,
    ) -> ModelPromotionDecision:
        champion = await self.get_fitted_model(
            decision.run_id,
            decision.champion_fitted_model_id,
        )
        if champion.artifact_id != decision.champion_artifact_id:
            raise ConflictError("promotion champion lineage does not resolve")
        if decision.candidate_fitted_model_id is not None:
            candidate = await self.get_fitted_model(
                decision.run_id,
                decision.candidate_fitted_model_id,
            )
            if candidate.artifact_id != decision.candidate_artifact_id:
                raise ConflictError("promotion candidate lineage does not resolve")
        folds = {
            fold.id: fold
            for fold in await self.list_temporal_evaluation_folds(
                decision.run_id,
                challenge_id=decision.challenge_id,
            )
        }
        for fold_id in decision.evaluation_fold_ids:
            fold = folds.get(fold_id)
            if fold is None:
                raise ConflictError("promotion evaluation fold does not resolve")
            if (
                decision.candidate_artifact_id is not None
                and fold.artifact_id != decision.candidate_artifact_id
            ):
                raise ConflictError("promotion fold does not evaluate the candidate")
        return await self._append_challenge_artifact(
            table="lithops_model_promotion_decisions",
            identity_column="id",
            identity=decision.id,
            payload_model=decision,
            row={
                "id": str(decision.id),
                "challenge_id": str(decision.challenge_id),
                "run_id": str(decision.run_id),
                "decision_day": decision.decision_day,
                "champion_artifact_id": str(decision.champion_artifact_id),
                "champion_fitted_model_id": str(decision.champion_fitted_model_id),
                "candidate_artifact_id": (
                    str(decision.candidate_artifact_id)
                    if decision.candidate_artifact_id is not None
                    else None
                ),
                "candidate_fitted_model_id": (
                    str(decision.candidate_fitted_model_id)
                    if decision.candidate_fitted_model_id is not None
                    else None
                ),
                "disposition": decision.disposition,
                "payload": decision.model_dump(mode="json"),
                "created_at": decision.created_at.isoformat(),
            },
            model_type=ModelPromotionDecision,
        )

    async def get_model_promotion_decision(
        self,
        decision_id: UUID,
    ) -> ModelPromotionDecision:
        rows = await self._request(
            "GET",
            "lithops_model_promotion_decisions",
            params={"id": f"eq.{decision_id}", "select": "payload"},
        )
        if not rows:
            raise NotFoundError(f"promotion decision not found: {decision_id}")
        return ModelPromotionDecision.model_validate(rows[0]["payload"])

    async def get_model_promotion_decision_for_challenge(
        self,
        run_id: UUID,
        challenge_id: UUID,
    ) -> ModelPromotionDecision | None:
        rows = await self._request(
            "GET",
            "lithops_model_promotion_decisions",
            params={
                "run_id": f"eq.{run_id}",
                "challenge_id": f"eq.{challenge_id}",
                "select": "payload",
                "limit": "1",
            },
        )
        return ModelPromotionDecision.model_validate(rows[0]["payload"]) if rows else None

    async def activate_model(
        self,
        assignment: ActiveModelAssignment,
        *,
        expected_previous_sequence: int | None,
    ) -> ActiveModelAssignment:
        rows = await self._request(
            "GET",
            "lithops_active_model_assignments",
            params={"id": f"eq.{assignment.id}", "select": "payload"},
        )
        if rows:
            existing = ActiveModelAssignment.model_validate(rows[0]["payload"])
            if existing != assignment:
                raise ConflictError(f"model activation ID conflict: {assignment.id}")
            return existing
        rows = await self._request(
            "GET",
            "lithops_active_model_assignments",
            params={
                "promotion_decision_id": f"eq.{assignment.promotion_decision_id}",
                "select": "payload",
            },
        )
        if rows:
            raise ConflictError("promotion decision is already activated")
        previous = await self.get_active_model(assignment.run_id)
        previous_sequence = previous.sequence if previous else None
        if previous_sequence != expected_previous_sequence:
            raise ConflictError(
                "active-model sequence conflict: "
                f"expected {expected_previous_sequence}, got {previous_sequence}"
            )
        expected_sequence = 1 if previous is None else previous.sequence + 1
        if assignment.sequence != expected_sequence:
            raise ConflictError(
                f"active-model sequence must be {expected_sequence}, got {assignment.sequence}"
            )
        decision = await self.get_model_promotion_decision(assignment.promotion_decision_id)
        if decision.disposition != PromotionDisposition.PROMOTED:
            raise ConflictError("only a promoted candidate can become active")
        if (
            decision.run_id != assignment.run_id
            or decision.candidate_artifact_id != assignment.artifact_id
            or decision.candidate_fitted_model_id != assignment.fitted_model_id
        ):
            raise ConflictError("activation target differs from the promoted candidate")
        artifact = await self.get_model_artifact(assignment.artifact_id)
        fitted = await self.get_fitted_model(assignment.run_id, assignment.fitted_model_id)
        if (
            artifact.content_hash != assignment.artifact_hash
            or fitted.state_hash != assignment.fitted_state_hash
            or fitted.artifact_id != assignment.artifact_id
        ):
            raise ConflictError("activation content hashes do not resolve")
        try:
            rows = await self._request(
                "POST",
                "lithops_active_model_assignments",
                json={
                    "id": str(assignment.id),
                    "run_id": str(assignment.run_id),
                    "sequence": assignment.sequence,
                    "artifact_id": str(assignment.artifact_id),
                    "fitted_model_id": str(assignment.fitted_model_id),
                    "promotion_decision_id": str(assignment.promotion_decision_id),
                    "payload": assignment.model_dump(mode="json"),
                    "created_at": assignment.activated_at.isoformat(),
                },
                prefer="return=representation",
            )
        except ConflictError as exc:
            current = await self.get_active_model(assignment.run_id)
            if current != assignment:
                raise exc from None
            return current
        return ActiveModelAssignment.model_validate(rows[0]["payload"])

    async def get_active_model(self, run_id: UUID) -> ActiveModelAssignment | None:
        rows = await self._request(
            "GET",
            "lithops_active_model_assignments",
            params={
                "run_id": f"eq.{run_id}",
                "select": "payload",
                "order": "sequence.desc",
                "limit": "1",
            },
        )
        return ActiveModelAssignment.model_validate(rows[0]["payload"]) if rows else None

    async def list_model_activations(self, run_id: UUID) -> list[ActiveModelAssignment]:
        rows = await self._request(
            "GET",
            "lithops_active_model_assignments",
            params={
                "run_id": f"eq.{run_id}",
                "select": "payload",
                "order": "sequence.asc",
            },
        )
        return [ActiveModelAssignment.model_validate(row["payload"]) for row in rows]

    async def get_latest_world_model(self, run_id: UUID) -> WorldModelVersion | None:
        rows = await self._request(
            "GET",
            "lithops_world_models",
            params={
                "run_id": f"eq.{run_id}",
                "select": "payload",
                "order": "version.desc",
                "limit": "1",
            },
        )
        return WorldModelVersion.model_validate(rows[0]["payload"]) if rows else None

    async def get_world_model(self, model_id: UUID) -> WorldModelVersion:
        rows = await self._request(
            "GET",
            "lithops_world_models",
            params={"id": f"eq.{model_id}", "select": "payload"},
        )
        if not rows:
            raise NotFoundError(f"world model not found: {model_id}")
        return WorldModelVersion.model_validate(rows[0]["payload"])

    async def list_world_models(self, run_id: UUID) -> list[WorldModelVersion]:
        rows = await self._request(
            "GET",
            "lithops_world_models",
            params={
                "run_id": f"eq.{run_id}",
                "select": "payload",
                "order": "version.asc",
            },
        )
        return [WorldModelVersion.model_validate(row["payload"]) for row in rows]

    async def list_world_model_relationships(
        self,
        model_id: UUID,
    ) -> list[WorldModelRelationship]:
        rows = await self._request(
            "GET",
            "lithops_world_model_relationships",
            params={
                "world_model_id": f"eq.{model_id}",
                "select": "payload",
                "order": "relationship_key.asc",
            },
        )
        return [WorldModelRelationship.model_validate(row["payload"]) for row in rows]

    async def append_world_model(
        self,
        world_model: WorldModelVersion,
        *,
        expected_latest_version: int | None,
    ) -> WorldModelVersion:
        rows = await self._request(
            "GET",
            "lithops_world_models",
            params={"id": f"eq.{world_model.id}", "select": "payload"},
        )
        if rows:
            existing = WorldModelVersion.model_validate(rows[0]["payload"])
            if existing != world_model:
                raise ConflictError(f"world model ID conflict: {world_model.id}")
            return existing

        latest = await self.get_latest_world_model(world_model.run_id)
        latest_version = latest.version if latest else None
        if latest_version != expected_latest_version:
            raise ConflictError(
                "world-model version conflict: "
                f"expected {expected_latest_version}, got {latest_version}"
            )
        expected_new_version = 1 if latest is None else latest.version + 1
        if world_model.version != expected_new_version:
            raise ConflictError(
                f"world-model version must be {expected_new_version}, got {world_model.version}"
            )
        try:
            rows = await self._request(
                "POST",
                "lithops_world_models",
                json=self._world_model_row(world_model),
                prefer="return=representation",
            )
        except ConflictError as exc:
            try:
                existing = await self.get_world_model(world_model.id)
            except NotFoundError:
                raise exc from None
            if existing != world_model:
                raise
            return existing
        return WorldModelVersion.model_validate(rows[0]["payload"])

    async def append_prediction(
        self,
        prediction: PredictionLedgerEntry,
    ) -> PredictionLedgerEntry:
        rows = await self._request(
            "GET",
            "lithops_predictions",
            params={"id": f"eq.{prediction.id}", "select": "payload"},
        )
        if rows:
            existing = PredictionLedgerEntry.model_validate(rows[0]["payload"])
            if existing != prediction:
                raise ConflictError(f"prediction ID conflict: {prediction.id}")
            return existing
        try:
            rows = await self._request(
                "POST",
                "lithops_predictions",
                json=self._prediction_row(prediction),
                prefer="return=representation",
            )
        except ConflictError as exc:
            try:
                existing = await self.get_prediction(prediction.id)
            except NotFoundError:
                raise exc from None
            if existing != prediction:
                raise exc from None
            return existing
        return PredictionLedgerEntry.model_validate(rows[0]["payload"])

    async def get_prediction(self, prediction_id: UUID) -> PredictionLedgerEntry:
        rows = await self._request(
            "GET",
            "lithops_predictions",
            params={"id": f"eq.{prediction_id}", "select": "payload"},
        )
        if not rows:
            raise NotFoundError(f"prediction not found: {prediction_id}")
        return PredictionLedgerEntry.model_validate(rows[0]["payload"])

    async def list_predictions(self, run_id: UUID) -> list[PredictionLedgerEntry]:
        rows = await self._request(
            "GET",
            "lithops_predictions",
            params={
                "run_id": f"eq.{run_id}",
                "select": "payload",
                "order": "issued_day.asc",
            },
        )
        return [PredictionLedgerEntry.model_validate(row["payload"]) for row in rows]

    async def get_prediction_outcome(self, target_id: UUID) -> PredictionOutcome | None:
        rows = await self._request(
            "GET",
            "lithops_prediction_outcomes",
            params={"target_id": f"eq.{target_id}", "select": "payload"},
        )
        return PredictionOutcome.model_validate(rows[0]["payload"]) if rows else None

    async def list_prediction_outcomes(self, run_id: UUID) -> list[PredictionOutcome]:
        rows = await self._request(
            "GET",
            "lithops_prediction_outcomes",
            params={
                "run_id": f"eq.{run_id}",
                "select": "payload",
                "order": "observed_day.asc",
            },
        )
        return [PredictionOutcome.model_validate(row["payload"]) for row in rows]

    async def append_prediction_outcome(
        self,
        outcome: PredictionOutcome,
    ) -> PredictionOutcome:
        existing = await self.get_prediction_outcome(outcome.target_id)
        if existing is not None:
            if existing != outcome:
                raise ConflictError(f"prediction target outcome conflict: {outcome.target_id}")
            return existing
        try:
            rows = await self._request(
                "POST",
                "lithops_prediction_outcomes",
                json=self._prediction_outcome_row(outcome),
                prefer="return=representation",
            )
        except ConflictError:
            existing = await self.get_prediction_outcome(outcome.target_id)
            if existing is None or existing != outcome:
                raise
            return existing
        return PredictionOutcome.model_validate(rows[0]["payload"])

    async def append_portfolio_revision(
        self,
        revision: StrategicPortfolioRevision,
    ) -> StrategicPortfolioRevision:
        rows = await self._request(
            "GET",
            "lithops_strategic_portfolio_revisions",
            params={"id": f"eq.{revision.id}", "select": "payload"},
        )
        if rows:
            existing = StrategicPortfolioRevision.model_validate(rows[0]["payload"])
            if existing.model_dump(exclude={"created_at"}) != revision.model_dump(
                exclude={"created_at"}
            ):
                raise ConflictError(f"portfolio revision conflict: {revision.id}")
            return existing
        latest = await self.get_latest_portfolio_revision(revision.run_id)
        if latest is None:
            if revision.revision != 1:
                raise ConflictError(
                    "portfolio history must start at revision 1: "
                    f"got revision {revision.revision}"
                )
        else:
            if revision.revision != latest.revision + 1:
                raise ConflictError(
                    "portfolio revision must extend the latest revision: "
                    f"latest={latest.revision}, got={revision.revision}"
                )
            if revision.portfolio.prior_portfolio_hash != latest.portfolio_hash:
                raise ConflictError(
                    f"portfolio hash chain broken at revision {revision.revision}"
                )
        try:
            rows = await self._request(
                "POST",
                "lithops_strategic_portfolio_revisions",
                json={
                    "id": str(revision.id),
                    "run_id": str(revision.run_id),
                    "decision_id": (
                        str(revision.decision_id) if revision.decision_id else None
                    ),
                    "week": revision.week,
                    "revision": revision.revision,
                    "portfolio_hash": revision.portfolio_hash,
                    "prior_portfolio_hash": revision.portfolio.prior_portfolio_hash,
                    "payload": revision.model_dump(mode="json"),
                },
                prefer="return=representation",
            )
        except ConflictError as exc:
            rows = await self._request(
                "GET",
                "lithops_strategic_portfolio_revisions",
                params={"id": f"eq.{revision.id}", "select": "payload"},
            )
            if not rows:
                raise exc from None
            existing = StrategicPortfolioRevision.model_validate(rows[0]["payload"])
            if existing.model_dump(exclude={"created_at"}) != revision.model_dump(
                exclude={"created_at"}
            ):
                raise exc from None
            return existing
        return StrategicPortfolioRevision.model_validate(rows[0]["payload"])

    async def list_portfolio_revisions(
        self,
        run_id: UUID,
    ) -> list[StrategicPortfolioRevision]:
        rows = await self._request(
            "GET",
            "lithops_strategic_portfolio_revisions",
            params={
                "run_id": f"eq.{run_id}",
                "select": "payload",
                "order": "revision.asc",
            },
        )
        return [
            StrategicPortfolioRevision.model_validate(row["payload"]) for row in rows
        ]

    async def get_latest_portfolio_revision(
        self,
        run_id: UUID,
    ) -> StrategicPortfolioRevision | None:
        rows = await self._request(
            "GET",
            "lithops_strategic_portfolio_revisions",
            params={
                "run_id": f"eq.{run_id}",
                "select": "payload",
                "order": "revision.desc",
                "limit": "1",
            },
        )
        if not rows:
            return None
        return StrategicPortfolioRevision.model_validate(rows[0]["payload"])

    async def append_experiment_outcome(
        self,
        outcome: ExperimentOutcome,
    ) -> ExperimentOutcome:
        rows = await self._request(
            "GET",
            "lithops_experiment_outcomes",
            params={"id": f"eq.{outcome.id}", "select": "payload"},
        )
        if rows:
            existing = ExperimentOutcome.model_validate(rows[0]["payload"])
            if existing != outcome:
                raise ConflictError(f"experiment outcome conflict: {outcome.id}")
            return existing
        try:
            rows = await self._request(
                "POST",
                "lithops_experiment_outcomes",
                json={
                    "id": str(outcome.id),
                    "run_id": str(outcome.run_id),
                    "commitment_id": outcome.commitment_id,
                    "hypothesis_id": outcome.hypothesis_id,
                    "outcome_status": outcome.outcome_status.value,
                    "started_week": outcome.started_week,
                    "measured_week": outcome.measured_week,
                    "payload": outcome.model_dump(mode="json"),
                },
                prefer="return=representation",
            )
        except ConflictError as exc:
            rows = await self._request(
                "GET",
                "lithops_experiment_outcomes",
                params={"id": f"eq.{outcome.id}", "select": "payload"},
            )
            if not rows:
                raise exc from None
            existing = ExperimentOutcome.model_validate(rows[0]["payload"])
            if existing != outcome:
                raise exc from None
            return existing
        return ExperimentOutcome.model_validate(rows[0]["payload"])

    async def list_experiment_outcomes(self, run_id: UUID) -> list[ExperimentOutcome]:
        rows = await self._request(
            "GET",
            "lithops_experiment_outcomes",
            params={
                "run_id": f"eq.{run_id}",
                "select": "payload",
                "order": "started_week.asc,commitment_id.asc",
            },
        )
        return [ExperimentOutcome.model_validate(row["payload"]) for row in rows]

    async def list_commitment_experiment_outcomes(
        self,
        run_id: UUID,
        commitment_id: str,
    ) -> list[ExperimentOutcome]:
        rows = await self._request(
            "GET",
            "lithops_experiment_outcomes",
            params={
                "run_id": f"eq.{run_id}",
                "commitment_id": f"eq.{commitment_id}",
                "select": "payload",
                "order": "started_week.asc",
            },
        )
        return [ExperimentOutcome.model_validate(row["payload"]) for row in rows]

    async def append_commitment_review(
        self,
        review: CommitmentReview,
    ) -> CommitmentReview:
        rows = await self._request(
            "GET",
            "lithops_commitment_reviews",
            params={"id": f"eq.{review.id}", "select": "payload"},
        )
        if rows:
            existing = CommitmentReview.model_validate(rows[0]["payload"])
            if existing.model_dump(exclude={"created_at"}) != review.model_dump(
                exclude={"created_at"}
            ):
                raise ConflictError(f"commitment review conflict: {review.id}")
            return existing
        try:
            rows = await self._request(
                "POST",
                "lithops_commitment_reviews",
                json={
                    "id": str(review.id),
                    "run_id": str(review.run_id),
                    "commitment_id": review.commitment_id,
                    "week": review.week,
                    "verdict": review.verdict.value,
                    "payload": review.model_dump(mode="json"),
                },
                prefer="return=representation",
            )
        except ConflictError as exc:
            rows = await self._request(
                "GET",
                "lithops_commitment_reviews",
                params={"id": f"eq.{review.id}", "select": "payload"},
            )
            if not rows:
                raise exc from None
            existing = CommitmentReview.model_validate(rows[0]["payload"])
            if existing.model_dump(exclude={"created_at"}) != review.model_dump(
                exclude={"created_at"}
            ):
                raise exc from None
            return existing
        return CommitmentReview.model_validate(rows[0]["payload"])

    async def list_commitment_reviews(
        self,
        run_id: UUID,
        commitment_id: str | None = None,
    ) -> list[CommitmentReview]:
        params = {
            "run_id": f"eq.{run_id}",
            "select": "payload",
            "order": "week.asc,commitment_id.asc",
        }
        if commitment_id is not None:
            params["commitment_id"] = f"eq.{commitment_id}"
        rows = await self._request("GET", "lithops_commitment_reviews", params=params)
        return [CommitmentReview.model_validate(row["payload"]) for row in rows]

    async def append_candidate_evaluation_set(
        self,
        evaluation_set: CandidateEvaluationSet,
    ) -> CandidateEvaluationSet:
        rows = await self._request(
            "GET",
            "lithops_candidate_evaluation_sets",
            params={"id": f"eq.{evaluation_set.id}", "select": "payload"},
        )
        if rows:
            existing = CandidateEvaluationSet.model_validate(rows[0]["payload"])
            if existing.set_hash != evaluation_set.set_hash:
                raise ConflictError(
                    f"candidate evaluation set conflict: {evaluation_set.id}"
                )
            return existing
        try:
            rows = await self._request(
                "POST",
                "lithops_candidate_evaluation_sets",
                json={
                    "id": str(evaluation_set.id),
                    "run_id": str(evaluation_set.run_id),
                    "week": evaluation_set.week,
                    "set_hash": evaluation_set.set_hash,
                    "payload": evaluation_set.model_dump(mode="json"),
                },
                prefer="return=representation",
            )
        except ConflictError as exc:
            rows = await self._request(
                "GET",
                "lithops_candidate_evaluation_sets",
                params={"id": f"eq.{evaluation_set.id}", "select": "payload"},
            )
            if not rows:
                raise exc from None
            existing = CandidateEvaluationSet.model_validate(rows[0]["payload"])
            if existing.set_hash != evaluation_set.set_hash:
                raise exc from None
            return existing
        return CandidateEvaluationSet.model_validate(rows[0]["payload"])

    async def get_candidate_evaluation_set(
        self,
        run_id: UUID,
        week: int,
    ) -> CandidateEvaluationSet | None:
        rows = await self._request(
            "GET",
            "lithops_candidate_evaluation_sets",
            params={
                "run_id": f"eq.{run_id}",
                "week": f"eq.{week}",
                "select": "payload",
            },
        )
        if not rows:
            return None
        return CandidateEvaluationSet.model_validate(rows[0]["payload"])

    async def append_executive_choice(self, choice: ExecutiveChoice) -> ExecutiveChoice:
        rows = await self._request(
            "GET",
            "lithops_executive_choices",
            params={"id": f"eq.{choice.id}", "select": "payload"},
        )
        if rows:
            existing = ExecutiveChoice.model_validate(rows[0]["payload"])
            if existing.model_dump(exclude={"created_at"}) != choice.model_dump(
                exclude={"created_at"}
            ):
                raise ConflictError(f"executive choice conflict: {choice.id}")
            return existing
        try:
            rows = await self._request(
                "POST",
                "lithops_executive_choices",
                json={
                    "id": str(choice.id),
                    "run_id": str(choice.run_id),
                    "week": choice.week,
                    "evaluation_set_id": str(choice.evaluation_set_id),
                    "selected_candidate_id": choice.selected_candidate_id,
                    "payload": choice.model_dump(mode="json"),
                },
                prefer="return=representation",
            )
        except ConflictError as exc:
            rows = await self._request(
                "GET",
                "lithops_executive_choices",
                params={"id": f"eq.{choice.id}", "select": "payload"},
            )
            if not rows:
                raise exc from None
            existing = ExecutiveChoice.model_validate(rows[0]["payload"])
            if existing.model_dump(exclude={"created_at"}) != choice.model_dump(
                exclude={"created_at"}
            ):
                raise exc from None
            return existing
        return ExecutiveChoice.model_validate(rows[0]["payload"])

    async def get_executive_choice(
        self,
        run_id: UUID,
        week: int,
    ) -> ExecutiveChoice | None:
        rows = await self._request(
            "GET",
            "lithops_executive_choices",
            params={
                "run_id": f"eq.{run_id}",
                "week": f"eq.{week}",
                "select": "payload",
            },
        )
        if not rows:
            return None
        return ExecutiveChoice.model_validate(rows[0]["payload"])

    async def append_insight_record(self, insight: InsightRecord) -> InsightRecord:
        rows = await self._request(
            "GET",
            "lithops_insight_records",
            params={"id": f"eq.{insight.id}", "select": "payload"},
        )
        if rows:
            return InsightRecord.model_validate(rows[0]["payload"])
        try:
            rows = await self._request(
                "POST",
                "lithops_insight_records",
                json={
                    "id": str(insight.id),
                    "run_id": str(insight.run_id),
                    "week": insight.week,
                    "tool": insight.tool,
                    "target_group": insight.target_group,
                    "parse_status": insight.parse_status.value,
                    "payload": insight.model_dump(mode="json"),
                },
                prefer="return=representation",
            )
        except ConflictError:
            rows = await self._request(
                "GET",
                "lithops_insight_records",
                params={"id": f"eq.{insight.id}", "select": "payload"},
            )
            if not rows:
                raise
            return InsightRecord.model_validate(rows[0]["payload"])
        return InsightRecord.model_validate(rows[0]["payload"])

    async def list_insight_records(self, run_id: UUID) -> list[InsightRecord]:
        rows = await self._request(
            "GET",
            "lithops_insight_records",
            params={
                "run_id": f"eq.{run_id}",
                "select": "payload",
                "order": "week.asc,tool.asc",
            },
        )
        return [InsightRecord.model_validate(row["payload"]) for row in rows]

    async def append_model_health_signal(
        self,
        signal: ModelHealthSignal,
    ) -> ModelHealthSignal:
        rows = await self._request(
            "GET",
            "lithops_model_health_signals",
            params={"id": f"eq.{signal.id}", "select": "payload"},
        )
        if rows:
            existing = ModelHealthSignal.model_validate(rows[0]["payload"])
            if existing != signal:
                raise ConflictError(f"model-health signal ID conflict: {signal.id}")
            return existing
        try:
            rows = await self._request(
                "POST",
                "lithops_model_health_signals",
                json=self._model_health_row(signal),
                prefer="return=representation",
            )
        except ConflictError as exc:
            try:
                existing = await self.get_model_health_signal(signal.id)
            except NotFoundError:
                raise exc from None
            if existing != signal:
                raise exc from None
            return existing
        return ModelHealthSignal.model_validate(rows[0]["payload"])

    async def get_model_health_signal(self, signal_id: UUID) -> ModelHealthSignal:
        rows = await self._request(
            "GET",
            "lithops_model_health_signals",
            params={"id": f"eq.{signal_id}", "select": "payload"},
        )
        if not rows:
            raise NotFoundError(f"model-health signal not found: {signal_id}")
        return ModelHealthSignal.model_validate(rows[0]["payload"])

    async def list_model_health_signals(self, run_id: UUID) -> list[ModelHealthSignal]:
        rows = await self._request(
            "GET",
            "lithops_model_health_signals",
            params={
                "run_id": f"eq.{run_id}",
                "select": "payload",
                "order": "evaluated_day.asc",
            },
        )
        return [ModelHealthSignal.model_validate(row["payload"]) for row in rows]

    async def get_model_challenge(self, challenge_id: UUID) -> ModelChallengeRecord | None:
        rows = await self._request(
            "GET",
            "lithops_model_challenges",
            params={"id": f"eq.{challenge_id}", "select": "payload"},
        )
        return ModelChallengeRecord.model_validate(rows[0]["payload"]) if rows else None

    async def save_model_challenge(self, challenge: ModelChallengeRecord) -> ModelChallengeRecord:
        existing = await self.get_model_challenge(challenge.id)
        if existing is not None:
            identity = (
                existing.run_id,
                existing.health_signal_id,
                existing.base_model_version_id,
                existing.requested_builders,
                existing.created_at,
            )
            incoming_identity = (
                challenge.run_id,
                challenge.health_signal_id,
                challenge.base_model_version_id,
                challenge.requested_builders,
                challenge.created_at,
            )
            if identity != incoming_identity:
                raise ConflictError(f"model challenge ID conflict: {challenge.id}")
            if existing.status in {
                ModelChallengeStatus.COMPLETED,
                ModelChallengeStatus.FAILED,
            }:
                return existing
            rows = await self._request(
                "PATCH",
                "lithops_model_challenges",
                params={"id": f"eq.{challenge.id}", "select": "payload"},
                json=self._model_challenge_row(challenge),
                prefer="return=representation",
            )
        else:
            rows = await self._request(
                "POST",
                "lithops_model_challenges",
                json=self._model_challenge_row(challenge),
                prefer="return=representation",
            )
        return ModelChallengeRecord.model_validate(rows[0]["payload"])

    async def append_model_challenge_package(
        self, package: ModelChallengePackage
    ) -> ModelChallengePackage:
        return await self._append_challenge_artifact(
            table="lithops_model_challenge_packages",
            identity_column="challenge_id",
            identity=package.challenge_id,
            payload_model=package,
            row={
                "challenge_id": str(package.challenge_id),
                "run_id": str(package.run_id),
                "payload": package.model_dump(mode="json"),
                "created_at": package.created_at.isoformat(),
            },
            model_type=ModelChallengePackage,
        )

    async def get_model_challenge_package(self, challenge_id: UUID) -> ModelChallengePackage | None:
        rows = await self._request(
            "GET",
            "lithops_model_challenge_packages",
            params={"challenge_id": f"eq.{challenge_id}", "select": "payload"},
        )
        return ModelChallengePackage.model_validate(rows[0]["payload"]) if rows else None

    async def append_model_builder_proposal(
        self, proposal: ModelBuilderProposal
    ) -> ModelBuilderProposal:
        return await self._append_challenge_artifact(
            table="lithops_model_builder_proposals",
            identity_column="id",
            identity=proposal.id,
            payload_model=proposal,
            row={
                "id": str(proposal.id),
                "challenge_id": str(proposal.challenge_id),
                "builder_name": proposal.builder_name,
                "payload": proposal.model_dump(mode="json"),
                "created_at": proposal.created_at.isoformat(),
            },
            model_type=ModelBuilderProposal,
        )

    async def list_model_builder_proposals(self, challenge_id: UUID) -> list[ModelBuilderProposal]:
        rows = await self._request(
            "GET",
            "lithops_model_builder_proposals",
            params={
                "challenge_id": f"eq.{challenge_id}",
                "select": "payload",
                "order": "builder_name.asc",
            },
        )
        return [ModelBuilderProposal.model_validate(row["payload"]) for row in rows]

    async def append_hypothesis_backtest(
        self, result: HypothesisBacktestResult
    ) -> HypothesisBacktestResult:
        return await self._append_challenge_artifact(
            table="lithops_hypothesis_backtests",
            identity_column="id",
            identity=result.id,
            payload_model=result,
            row={
                "id": str(result.id),
                "challenge_id": str(result.challenge_id),
                "proposal_id": str(result.proposal_id),
                "supported": result.supported,
                "penalized_improvement": result.penalized_improvement,
                "payload": result.model_dump(mode="json"),
                "created_at": result.evaluated_at.isoformat(),
            },
            model_type=HypothesisBacktestResult,
        )

    async def list_hypothesis_backtests(self, challenge_id: UUID) -> list[HypothesisBacktestResult]:
        rows = await self._request(
            "GET",
            "lithops_hypothesis_backtests",
            params={
                "challenge_id": f"eq.{challenge_id}",
                "select": "payload",
                "order": "penalized_improvement.desc",
            },
        )
        return [HypothesisBacktestResult.model_validate(row["payload"]) for row in rows]

    async def append_model_builder_call(
        self, receipt: ModelBuilderCallReceipt
    ) -> ModelBuilderCallReceipt:
        return await self._append_challenge_artifact(
            table="lithops_model_builder_calls",
            identity_column="id",
            identity=receipt.id,
            payload_model=receipt,
            row={
                "id": str(receipt.id),
                "challenge_id": str(receipt.challenge_id),
                "builder_name": receipt.builder_name,
                "attempt": receipt.attempt,
                "status": receipt.status,
                "input_hash": receipt.input_hash,
                "payload": receipt.model_dump(mode="json"),
                "created_at": receipt.completed_at.isoformat(),
            },
            model_type=ModelBuilderCallReceipt,
        )

    async def list_model_builder_calls(self, challenge_id: UUID) -> list[ModelBuilderCallReceipt]:
        rows = await self._request(
            "GET",
            "lithops_model_builder_calls",
            params={
                "challenge_id": f"eq.{challenge_id}",
                "select": "payload",
                "order": "builder_name.asc",
            },
        )
        return sorted(
            (ModelBuilderCallReceipt.model_validate(row["payload"]) for row in rows),
            key=lambda item: (item.builder_name, item.attempt),
        )

    async def append_model_challenge_decision(
        self, decision: ModelChallengeDecision
    ) -> ModelChallengeDecision:
        existing = await self.get_model_challenge_decision(decision.challenge_id)
        if existing is not None:
            if existing != decision:
                raise ConflictError(f"model-challenge decision conflict: {decision.challenge_id}")
            return existing
        return await self._append_challenge_artifact(
            table="lithops_model_challenge_decisions",
            identity_column="id",
            identity=decision.id,
            payload_model=decision,
            row={
                "id": str(decision.id),
                "challenge_id": str(decision.challenge_id),
                "resolution": decision.resolution,
                "activated_model_version_id": (
                    str(decision.activated_model_version_id)
                    if decision.activated_model_version_id
                    else None
                ),
                "payload": decision.model_dump(mode="json"),
                "created_at": decision.decided_at.isoformat(),
            },
            model_type=ModelChallengeDecision,
        )

    async def get_model_challenge_decision(
        self, challenge_id: UUID
    ) -> ModelChallengeDecision | None:
        rows = await self._request(
            "GET",
            "lithops_model_challenge_decisions",
            params={"challenge_id": f"eq.{challenge_id}", "select": "payload"},
        )
        return ModelChallengeDecision.model_validate(rows[0]["payload"]) if rows else None

    async def create_run(self, run: RunRecord) -> RunRecord:
        rows = await self._request(
            "POST",
            "lithops_runs",
            json=self._run_row(run),
            prefer="return=representation",
        )
        return RunRecord.model_validate(rows[0]["payload"])

    async def get_run(self, run_id: UUID) -> RunRecord:
        rows = await self._request(
            "GET",
            "lithops_runs",
            params={"id": f"eq.{run_id}", "select": "payload"},
        )
        if not rows:
            raise NotFoundError(f"run not found: {run_id}")
        return RunRecord.model_validate(rows[0]["payload"])

    async def save_run(self, run: RunRecord, *, expected_version: int) -> RunRecord:
        saved = run.model_copy(
            update={"version": expected_version + 1, "updated_at": utc_now()},
            deep=True,
        )
        rows = await self._request(
            "PATCH",
            "lithops_runs",
            params={
                "id": f"eq.{run.id}",
                "version": f"eq.{expected_version}",
                "select": "payload",
            },
            json=self._run_row(saved),
            prefer="return=representation",
        )
        if not rows:
            raise ConflictError(f"run version conflict: {run.id}")
        return RunRecord.model_validate(rows[0]["payload"])

    async def claim_run_lease(
        self,
        run_id: UUID,
        owner_id: str,
        *,
        now: datetime,
        ttl_seconds: int,
    ) -> RunLease | None:
        rows = await self._request(
            "POST",
            "rpc/lithops_claim_run_lease",
            json={
                "p_run_id": str(run_id),
                "p_owner_id": owner_id,
                "p_token": str(uuid4()),
                "p_now": now.isoformat(),
                "p_expires_at": (now + timedelta(seconds=ttl_seconds)).isoformat(),
            },
        )
        return RunLease.model_validate(rows[0]) if rows else None

    async def renew_run_lease(
        self,
        run_id: UUID,
        token: UUID,
        *,
        now: datetime,
        ttl_seconds: int,
    ) -> RunLease | None:
        rows = await self._request(
            "POST",
            "rpc/lithops_renew_run_lease",
            json={
                "p_run_id": str(run_id),
                "p_token": str(token),
                "p_now": now.isoformat(),
                "p_expires_at": (now + timedelta(seconds=ttl_seconds)).isoformat(),
            },
        )
        return RunLease.model_validate(rows[0]) if rows else None

    async def release_run_lease(self, run_id: UUID, token: UUID) -> bool:
        rows = await self._request(
            "DELETE",
            "lithops_run_leases",
            params={
                "run_id": f"eq.{run_id}",
                "token": f"eq.{token}",
                "select": "run_id",
            },
            prefer="return=representation",
        )
        return bool(rows)

    async def get_decision_for_week(self, run_id: UUID, week: int) -> DecisionRecord | None:
        rows = await self._request(
            "GET",
            "lithops_decisions",
            params={
                "run_id": f"eq.{run_id}",
                "week": f"eq.{week}",
                "select": "payload",
            },
        )
        return DecisionRecord.model_validate(rows[0]["payload"]) if rows else None

    async def get_decision(self, run_id: UUID, decision_id: UUID) -> DecisionRecord:
        rows = await self._request(
            "GET",
            "lithops_decisions",
            params={
                "run_id": f"eq.{run_id}",
                "id": f"eq.{decision_id}",
                "select": "payload",
            },
        )
        if not rows:
            raise NotFoundError(f"decision not found: {decision_id}")
        return DecisionRecord.model_validate(rows[0]["payload"])

    async def list_decisions(self, run_id: UUID) -> list[DecisionRecord]:
        rows = await self._request(
            "GET",
            "lithops_decisions",
            params={
                "run_id": f"eq.{run_id}",
                "select": "payload",
                "order": "week.asc",
            },
        )
        return [DecisionRecord.model_validate(row["payload"]) for row in rows]

    async def list_candidate_simulations(
        self,
        run_id: UUID,
        decision_id: UUID,
    ) -> list[CandidateEvaluationRecord]:
        rows = await self._request(
            "GET",
            "lithops_candidate_simulations",
            params={
                "run_id": f"eq.{run_id}",
                "decision_id": f"eq.{decision_id}",
                "select": "payload",
                "order": "robust_utility.desc",
            },
        )
        return [CandidateEvaluationRecord.model_validate(row["payload"]) for row in rows]

    async def save_decision(self, decision: DecisionRecord) -> DecisionRecord:
        existing = await self.get_decision_for_week(decision.run_id, decision.week)
        if existing is not None:
            return existing
        try:
            rows = await self._request(
                "POST",
                "lithops_decisions",
                json=self._decision_row(decision),
                prefer="return=representation",
            )
        except ConflictError:
            existing = await self.get_decision_for_week(decision.run_id, decision.week)
            if existing is None:
                raise
            return existing
        return DecisionRecord.model_validate(rows[0]["payload"])

    async def update_decision(self, decision: DecisionRecord) -> DecisionRecord:
        rows = await self._request(
            "PATCH",
            "lithops_decisions",
            params={"id": f"eq.{decision.id}", "select": "payload"},
            json=self._decision_row(decision),
            prefer="return=representation",
        )
        if not rows:
            raise NotFoundError(f"decision not found: {decision.id}")
        return DecisionRecord.model_validate(rows[0]["payload"])

    async def get_receipt(self, run_id: UUID, idempotency_key: str) -> ActionReceipt | None:
        rows = await self._request(
            "GET",
            "lithops_action_receipts",
            params={
                "run_id": f"eq.{run_id}",
                "idempotency_key": f"eq.{idempotency_key}",
                "select": "payload",
            },
        )
        return ActionReceipt.model_validate(rows[0]["payload"]) if rows else None

    async def save_receipt(self, receipt: ActionReceipt) -> ActionReceipt:
        existing = await self.get_receipt(receipt.run_id, receipt.idempotency_key)
        if existing is not None:
            return existing
        try:
            rows = await self._request(
                "POST",
                "lithops_action_receipts",
                json=self._receipt_row(receipt),
                prefer="return=representation",
            )
        except ConflictError:
            existing = await self.get_receipt(receipt.run_id, receipt.idempotency_key)
            if existing is None:
                raise
            return existing
        return ActionReceipt.model_validate(rows[0]["payload"])

    async def list_receipts(self, decision_id: UUID) -> list[ActionReceipt]:
        rows = await self._request(
            "GET",
            "lithops_action_receipts",
            params={
                "decision_id": f"eq.{decision_id}",
                "select": "payload",
                "order": "created_at.asc",
            },
        )
        return [ActionReceipt.model_validate(row["payload"]) for row in rows]

    async def append_event(self, event: EventRecord) -> EventRecord:
        existing_rows = await self._request(
            "GET",
            "lithops_events",
            params={"event_id": f"eq.{event.id}", "select": "id,payload"},
        )
        if existing_rows:
            payload = existing_rows[0]["payload"]
            payload["sequence"] = existing_rows[0]["id"]
            existing = EventRecord.model_validate(payload)
            if existing.model_copy(update={"sequence": None}) != event:
                raise ConflictError(f"event ID conflict: {event.id}")
            return existing
        rows = await self._request(
            "POST",
            "lithops_events",
            json={
                "event_id": str(event.id),
                "run_id": str(event.run_id),
                "type": event.type,
                "payload": event.model_dump(mode="json"),
                "created_at": event.created_at.isoformat(),
            },
            prefer="return=representation",
        )
        payload = rows[0]["payload"]
        payload["sequence"] = rows[0]["id"]
        return EventRecord.model_validate(payload)

    async def list_events(self, run_id: UUID) -> list[EventRecord]:
        rows = await self._request(
            "GET",
            "lithops_events",
            params={
                "run_id": f"eq.{run_id}",
                "select": "id,payload",
                "order": "id.asc",
            },
        )
        events: list[EventRecord] = []
        for row in rows:
            payload = row["payload"]
            payload["sequence"] = row["id"]
            events.append(EventRecord.model_validate(payload))
        return events

    async def _append_challenge_artifact(
        self,
        *,
        table: str,
        identity_column: str,
        identity: UUID,
        payload_model: Any,
        row: dict[str, Any],
        model_type: Any,
    ) -> Any:
        rows = await self._request(
            "GET",
            table,
            params={identity_column: f"eq.{identity}", "select": "payload"},
        )
        if rows:
            existing = model_type.model_validate(rows[0]["payload"])
            if existing.model_dump(exclude={"created_at"}) != payload_model.model_dump(
                exclude={"created_at"}
            ):
                raise ConflictError(f"{table} identity conflict: {identity}")
            return existing
        try:
            rows = await self._request(
                "POST",
                table,
                json=row,
                prefer="return=representation",
            )
        except ConflictError as exc:
            rows = await self._request(
                "GET",
                table,
                params={identity_column: f"eq.{identity}", "select": "payload"},
            )
            if not rows:
                raise exc from None
            existing = model_type.model_validate(rows[0]["payload"])
            if existing.model_dump(exclude={"created_at"}) != payload_model.model_dump(
                exclude={"created_at"}
            ):
                raise exc from None
            return existing
        return model_type.model_validate(rows[0]["payload"])

    async def get_operation(self, run_id: UUID, request_id: str) -> StepOperation | None:
        rows = await self._request(
            "GET",
            "lithops_operations",
            params={
                "run_id": f"eq.{run_id}",
                "request_id": f"eq.{request_id}",
                "select": "payload",
            },
        )
        return StepOperation.model_validate(rows[0]["payload"]) if rows else None

    async def start_operation(self, operation: StepOperation) -> StepOperation:
        existing = await self.get_operation(operation.run_id, operation.request_id)
        if existing is not None:
            if existing.status == OperationStatus.COMPLETED:
                return existing
            if existing.status == OperationStatus.STARTED:
                raise OperationInProgressError(
                    f"operation already in progress: {operation.request_id}"
                )
            restarted = existing.model_copy(
                update={
                    "status": OperationStatus.STARTED,
                    "error": None,
                    "attempts": existing.attempts + 1,
                    "updated_at": utc_now(),
                },
                deep=True,
            )
            return await self._update_operation(restarted)

        try:
            rows = await self._request(
                "POST",
                "lithops_operations",
                json=self._operation_row(operation),
                prefer="return=representation",
            )
        except ConflictError:
            raise OperationInProgressError(
                f"operation already claimed: {operation.request_id}"
            ) from None
        return StepOperation.model_validate(rows[0]["payload"])

    async def complete_operation(
        self, run_id: UUID, request_id: str, result: StepResult
    ) -> StepOperation:
        operation = await self._required_operation(run_id, request_id)
        completed = operation.model_copy(
            update={
                "status": OperationStatus.COMPLETED,
                "result": result.model_dump(mode="json"),
                "error": None,
                "updated_at": utc_now(),
            },
            deep=True,
        )
        return await self._update_operation(completed)

    async def fail_operation(self, run_id: UUID, request_id: str, error: str) -> StepOperation:
        operation = await self._required_operation(run_id, request_id)
        failed = operation.model_copy(
            update={
                "status": OperationStatus.FAILED,
                "result": None,
                "error": error,
                "updated_at": utc_now(),
            },
            deep=True,
        )
        return await self._update_operation(failed)

    async def _required_operation(self, run_id: UUID, request_id: str) -> StepOperation:
        operation = await self.get_operation(run_id, request_id)
        if operation is None:
            raise NotFoundError(f"operation not found: {request_id}")
        return operation

    async def _update_operation(self, operation: StepOperation) -> StepOperation:
        rows = await self._request(
            "PATCH",
            "lithops_operations",
            params={
                "run_id": f"eq.{operation.run_id}",
                "request_id": f"eq.{operation.request_id}",
                "select": "payload",
            },
            json=self._operation_row(operation),
            prefer="return=representation",
        )
        if not rows:
            raise NotFoundError(f"operation not found: {operation.request_id}")
        return StepOperation.model_validate(rows[0]["payload"])

    async def _request(
        self,
        method: str,
        table: str,
        *,
        params: dict[str, str] | None = None,
        json: Any | None = None,
        prefer: str | None = None,
    ) -> list[dict[str, Any]]:
        headers = dict(self._headers)
        if prefer:
            headers["Prefer"] = prefer
        response = await self._client.request(
            method,
            f"{self._base_url}/{table}",
            params=params,
            json=json,
            headers=headers,
        )
        if response.status_code == 409:
            raise ConflictError(response.text)
        if response.status_code == 404:
            raise NotFoundError(response.text)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RepositoryError(
                f"Supabase request failed ({response.status_code}): {response.text}"
            ) from exc
        if not response.content:
            return []
        data = response.json()
        return data if isinstance(data, list) else [data]

    async def _request_all_pages(
        self,
        table: str,
        *,
        params: dict[str, str] | None = None,
        page_size: int = 1_000,
    ) -> list[dict[str, Any]]:
        """Read every PostgREST row despite the project's max-rows setting."""

        if page_size < 1:
            raise ValueError("PostgREST page size must be positive")
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            page_params = dict(params or {})
            page_params["limit"] = str(page_size)
            page_params["offset"] = str(offset)
            page = await self._request("GET", table, params=page_params)
            rows.extend(page)
            if len(page) < page_size:
                return rows
            offset += len(page)

    @staticmethod
    def _run_row(run: RunRecord) -> dict[str, Any]:
        return {
            "id": str(run.id),
            "status": run.status,
            "current_day": run.current_day,
            "benchmark_session_id": run.benchmark_session_id,
            "version": run.version,
            "payload": run.model_dump(mode="json"),
            "created_at": run.created_at.isoformat(),
            "updated_at": run.updated_at.isoformat(),
        }

    @staticmethod
    def _decision_row(decision: DecisionRecord) -> dict[str, Any]:
        return {
            "id": str(decision.id),
            "run_id": str(decision.run_id),
            "week": decision.week,
            "status": decision.status,
            "payload": decision.model_dump(mode="json"),
            "created_at": decision.created_at.isoformat(),
        }

    @staticmethod
    def _receipt_row(receipt: ActionReceipt) -> dict[str, Any]:
        return {
            "id": str(receipt.id),
            "run_id": str(receipt.run_id),
            "decision_id": str(receipt.decision_id),
            "idempotency_key": receipt.idempotency_key,
            "status": receipt.status,
            "payload": receipt.model_dump(mode="json"),
            "created_at": receipt.created_at.isoformat(),
        }

    @staticmethod
    def _operation_row(operation: StepOperation) -> dict[str, Any]:
        return {
            "id": str(operation.id),
            "run_id": str(operation.run_id),
            "request_id": operation.request_id,
            "status": operation.status,
            "payload": operation.model_dump(mode="json"),
            "created_at": operation.created_at.isoformat(),
            "updated_at": operation.updated_at.isoformat(),
        }

    @staticmethod
    def _world_model_row(world_model: WorldModelVersion) -> dict[str, Any]:
        return {
            "id": str(world_model.id),
            "run_id": str(world_model.run_id),
            "version": world_model.version,
            "source_observation_day": world_model.source_observation_day,
            "based_on_version_id": (
                str(world_model.based_on_version_id)
                if world_model.based_on_version_id is not None
                else None
            ),
            "update_method": world_model.update_method,
            "payload": world_model.model_dump(mode="json"),
            "created_at": world_model.created_at.isoformat(),
        }

    @staticmethod
    def _prediction_row(prediction: PredictionLedgerEntry) -> dict[str, Any]:
        return {
            "id": str(prediction.id),
            "run_id": str(prediction.run_id),
            "decision_id": str(prediction.decision_id),
            "model_version_id": str(prediction.model_version_id),
            "issued_day": prediction.issued_day,
            "payload": prediction.model_dump(mode="json"),
            "created_at": prediction.committed_at.isoformat(),
        }

    @staticmethod
    def _prediction_outcome_row(outcome: PredictionOutcome) -> dict[str, Any]:
        return {
            "id": str(outcome.id),
            "run_id": str(outcome.run_id),
            "ledger_entry_id": str(outcome.ledger_entry_id),
            "target_id": str(outcome.target_id),
            "observed_day": outcome.actual.observed_day,
            "payload": outcome.model_dump(mode="json"),
            "created_at": outcome.recorded_at.isoformat(),
        }

    @staticmethod
    def _model_health_row(signal: ModelHealthSignal) -> dict[str, Any]:
        return {
            "id": str(signal.id),
            "run_id": str(signal.run_id),
            "model_version_id": str(signal.model_version_id),
            "evaluated_day": signal.evaluated_day,
            "status": signal.status,
            "rebuild_recommended": signal.rebuild_recommended,
            "payload": signal.model_dump(mode="json"),
            "created_at": signal.evaluated_at.isoformat(),
        }

    @staticmethod
    def _model_challenge_row(challenge: ModelChallengeRecord) -> dict[str, Any]:
        return {
            "id": str(challenge.id),
            "run_id": str(challenge.run_id),
            "health_signal_id": str(challenge.health_signal_id),
            "base_model_version_id": str(challenge.base_model_version_id),
            "status": challenge.status,
            "payload": challenge.model_dump(mode="json"),
            "created_at": challenge.created_at.isoformat(),
            "updated_at": challenge.updated_at.isoformat(),
        }
