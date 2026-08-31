"""Deterministic evaluation and activation of agent-authored executable challengers."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from statistics import fmean
from typing import Protocol
from uuid import UUID

from pydantic import JsonValue

from lithops.domain.errors import ConflictError
from lithops.domain.executable_model import ModelArtifact, ModelRuntimeKind
from lithops.domain.model_challenge import (
    ModelChallengePackage,
    ModelChallengeRecord,
    ModelChallengeStatus,
)
from lithops.domain.model_registry import (
    ActiveModelAssignment,
    ModelArtifactAuthoringReceipt,
    ModelPromotionDecision,
    PromotionDisposition,
)
from lithops.domain.ports.executable_model import ExecutableCompanyModel
from lithops.domain.ports.model_registry_repository import ModelRegistryRepository
from lithops.domain.world_model import WorldModelVersion
from lithops.model_runtime import (
    ArtifactEvaluationResult,
    FixedBaselineModel,
    ModelStressCase,
    TemporalModelEvaluator,
    TemporalObservation,
    runtime_for_artifact,
)


class ExecutableArtifactAuthor(Protocol):
    async def author(
        self,
        *,
        package: ModelChallengePackage,
        parent_artifact: ModelArtifact,
    ) -> ModelArtifact: ...


@dataclass(frozen=True, slots=True)
class ExecutableChallengeResult:
    promotion: ModelPromotionDecision | None
    activation: ActiveModelAssignment | None
    candidate_evaluation: ArtifactEvaluationResult | None
    failure_reason_code: str | None = None
    failure_codes: tuple[str, ...] = ()
    operational_fallback_artifact_id: UUID | None = None


class ExecutableModelChallenge:
    """Author challengers, score unseen folds, and activate only the supported winner."""

    def __init__(
        self,
        *,
        repository: ModelRegistryRepository,
        authors: tuple[ExecutableArtifactAuthor, ...],
        evaluator: TemporalModelEvaluator | None = None,
        baseline: ExecutableCompanyModel | None = None,
        runtime_factory: Callable[[ModelArtifact], ExecutableCompanyModel] | None = None,
        author_timeout_seconds: float = 150.0,
    ) -> None:
        if not authors:
            raise ValueError("executable model challenge requires at least one author")
        if author_timeout_seconds <= 0:
            raise ValueError("executable model author timeout must be positive")
        self.repository = repository
        self.authors = authors
        self.evaluator = evaluator or TemporalModelEvaluator()
        self.baseline = baseline or FixedBaselineModel()
        self.runtime_factory = runtime_factory or runtime_for_artifact
        self.author_timeout_seconds = author_timeout_seconds
        author_keys = tuple(self._author_key(author, index) for index, author in enumerate(authors))
        if len(author_keys) != len(set(author_keys)):
            raise ValueError("executable model challenge author keys must be unique")

    async def run(
        self,
        *,
        package: ModelChallengePackage,
        observations: tuple[TemporalObservation, ...],
        world_model: WorldModelVersion,
        seed: int = 0,
    ) -> ExecutableChallengeResult:
        package = await self._canonical_package(package)
        completed = await self.repository.get_model_promotion_decision_for_challenge(
            package.run_id,
            package.challenge_id,
        )
        if completed is not None:
            await self._complete_challenge(package, completed)
            return ExecutableChallengeResult(
                promotion=completed,
                activation=await self._ensure_activation(completed),
                candidate_evaluation=None,
            )
        active = await self.repository.get_active_model(package.run_id)
        champion_runtime = await self._runtime_for_active(active)
        baseline_result = self._evaluate(
            package.run_id,
            package.challenge_id,
            self.baseline,
            observations,
            world_model,
            seed,
        )
        champion_result = (
            baseline_result
            if champion_runtime.artifact.id == self.baseline.artifact.id
            else self._evaluate(
                package.run_id,
                package.challenge_id,
                champion_runtime,
                observations,
                world_model,
                seed,
            )
        )
        await self._persist_evaluation(package.run_id, baseline_result)
        if champion_result is not baseline_result:
            await self._persist_evaluation(package.run_id, champion_result)
        if not champion_result.passed:
            if active is not None and baseline_result.passed:
                promotion = ModelPromotionDecision.create(
                    challenge_id=package.challenge_id,
                    run_id=package.run_id,
                    decision_day=package.health_signal.evaluated_day,
                    champion_artifact_id=champion_result.artifact.id,
                    champion_fitted_model_id=active.fitted_model_id,
                    candidate_artifact_id=baseline_result.artifact.id,
                    candidate_fitted_model_id=baseline_result.latest_fitted_model.id,
                    evaluation_fold_ids=tuple(
                        fold.id for fold in baseline_result.folds
                    ),
                    disposition=PromotionDisposition.PROMOTED,
                    reason_code="champion_failed_evaluation_baseline_recovery",
                    evidence={
                        "champion_failure_codes": list(champion_result.failure_codes),
                        "baseline_mean_score": baseline_result.mean_total_score,
                    },
                    created_at=package.created_at,
                )
                promotion = await self.repository.append_model_promotion_decision(
                    promotion
                )
                await self._complete_challenge(package, promotion)
                return ExecutableChallengeResult(
                    promotion=promotion,
                    activation=await self._ensure_activation(promotion),
                    candidate_evaluation=None,
                )
            failure_codes = tuple(
                dict.fromkeys(
                    (
                        *(f"champion:{code}" for code in champion_result.failure_codes),
                        *(f"baseline:{code}" for code in baseline_result.failure_codes),
                    )
                )
            )
            return ExecutableChallengeResult(
                promotion=None,
                activation=None,
                candidate_evaluation=None,
                failure_reason_code="active_champion_failed_evaluation",
                failure_codes=failure_codes,
                operational_fallback_artifact_id=self.baseline.artifact.id,
            )

        selected_authors = self._selected_authors(package)
        candidates: list[ArtifactEvaluationResult] = []
        author_failures: list[dict[str, str]] = []
        receipts = {
            receipt.author_key: receipt
            for receipt in await self.repository.list_model_artifact_authoring_receipts(
                package.run_id,
                package.challenge_id,
            )
        }
        pending_authors: list[tuple[int, ExecutableArtifactAuthor, str]] = []
        for index, author in selected_authors:
            author_key = self._author_key(author, index)
            if author_key not in receipts:
                pending_authors.append((index, author, author_key))

        async def author_candidate(
            index: int,
            author: ExecutableArtifactAuthor,
            author_key: str,
        ) -> tuple[int, str, ModelArtifact | None, dict[str, str] | None]:
            try:
                artifact = await asyncio.wait_for(
                    author.author(
                        package=package,
                        parent_artifact=champion_runtime.artifact,
                    ),
                    timeout=self.author_timeout_seconds,
                )
            except Exception as exc:
                message = str(exc).strip() or type(exc).__name__
                return (
                    index,
                    author_key,
                    None,
                    {
                        "author_key": author_key,
                        "error_code": type(exc).__name__,
                        "message": message[:500],
                    },
                )
            return index, author_key, artifact, None

        authored = await asyncio.gather(
            *(
                author_candidate(index, author, author_key)
                for index, author, author_key in pending_authors
            )
        )
        authored_by_key: dict[str, ModelArtifact] = {}
        for _, author_key, artifact, failure in authored:
            if failure is not None:
                author_failures.append(failure)
            elif artifact is not None:
                authored_by_key[author_key] = artifact

        for index, author in selected_authors:
            author_key = self._author_key(author, index)
            input_hash = self._authoring_input_hash(
                package,
                champion_runtime.artifact,
                author_key,
            )
            receipt = receipts.get(author_key)
            if receipt is not None:
                if receipt.input_hash != input_hash:
                    raise ValueError("persisted authoring input hash does not match challenge")
                artifact = await self.repository.get_model_artifact(receipt.artifact_id)
                if artifact.content_hash != receipt.artifact_hash:
                    raise ValueError("persisted authoring receipt artifact hash mismatch")
            else:
                artifact = authored_by_key.get(author_key)
                if artifact is None:
                    continue
                artifact = await self.repository.append_model_artifact(artifact)
                receipt = await self.repository.append_model_artifact_authoring_receipt(
                    ModelArtifactAuthoringReceipt.create(
                        challenge_id=package.challenge_id,
                        run_id=package.run_id,
                        author_key=author_key,
                        artifact_id=artifact.id,
                        artifact_hash=artifact.content_hash,
                        input_hash=input_hash,
                        created_at=package.created_at,
                    )
                )
                receipts[author_key] = receipt
            try:
                runtime = self.runtime_factory(artifact)
                result = self._evaluate(
                    package.run_id,
                    package.challenge_id,
                    runtime,
                    observations,
                    world_model,
                    seed,
                )
            except Exception as exc:
                message = str(exc).strip() or type(exc).__name__
                author_failures.append(
                    {
                        "author_key": author_key,
                        "error_code": type(exc).__name__,
                        "message": message[:500],
                    }
                )
                continue
            await self._persist_evaluation(package.run_id, result)
            candidates.append(result)
        if not candidates:
            promotion = self.evaluator.recommend_baseline_recovery(
                challenge_id=package.challenge_id,
                run_id=package.run_id,
                decision_day=package.health_signal.evaluated_day,
                champion=champion_result,
                baseline=baseline_result,
            )
            if promotion is None:
                promotion = ModelPromotionDecision.create(
                    challenge_id=package.challenge_id,
                    run_id=package.run_id,
                    decision_day=package.health_signal.evaluated_day,
                    champion_artifact_id=champion_result.artifact.id,
                    champion_fitted_model_id=champion_result.latest_fitted_model.id,
                    candidate_artifact_id=None,
                    candidate_fitted_model_id=None,
                    evaluation_fold_ids=(),
                    disposition=PromotionDisposition.NO_UPDATE,
                    reason_code="all_candidate_authors_failed",
                    evidence={"author_failures": author_failures},
                    created_at=package.created_at,
                )
            promotion = await self.repository.append_model_promotion_decision(promotion)
            await self._complete_challenge(package, promotion)
            activation = await self._ensure_activation(promotion)
            return ExecutableChallengeResult(
                promotion=promotion,
                activation=activation,
                candidate_evaluation=None,
            )
        passed_references = tuple(
            result
            for result in (champion_result, baseline_result)
            if result.passed
        )
        cash_eligible_candidates = tuple(
            candidate
            for candidate in candidates
            if candidate.passed
            and all(
                self.evaluator.cash_calibration_noninferior(candidate, reference)
                for reference in passed_references
            )
        )
        ranked_candidates = cash_eligible_candidates or tuple(candidates)
        candidate = min(
            ranked_candidates,
            key=lambda item: (
                0 if item.passed else 1,
                item.mean_total_score if item.passed else float("inf"),
                item.artifact.content_hash,
            ),
        )
        promotion = self.evaluator.recommend_promotion(
            challenge_id=package.challenge_id,
            run_id=package.run_id,
            decision_day=package.health_signal.evaluated_day,
            champion=champion_result,
            candidate=candidate,
            baseline=baseline_result,
        )
        promotion = await self.repository.append_model_promotion_decision(promotion)
        await self._complete_challenge(package, promotion)
        activation = await self._ensure_activation(promotion)
        return ExecutableChallengeResult(
            promotion=promotion,
            activation=activation,
            candidate_evaluation=candidate,
        )

    async def _ensure_activation(
        self,
        promotion: ModelPromotionDecision,
    ) -> ActiveModelAssignment | None:
        if promotion.disposition.value != "promoted":
            return None
        existing = next(
            (
                item
                for item in await self.repository.list_model_activations(promotion.run_id)
                if item.promotion_decision_id == promotion.id
            ),
            None,
        )
        if existing is not None:
            return existing
        if promotion.candidate_artifact_id is None or promotion.candidate_fitted_model_id is None:
            raise ValueError("promoted executable candidate has incomplete lineage")
        artifact = await self.repository.get_model_artifact(promotion.candidate_artifact_id)
        fitted = await self.repository.get_fitted_model(
            promotion.run_id,
            promotion.candidate_fitted_model_id,
        )
        active = await self.repository.get_active_model(promotion.run_id)
        previous_sequence = active.sequence if active is not None else None
        return await self.repository.activate_model(
            ActiveModelAssignment.create(
                run_id=promotion.run_id,
                sequence=1 if active is None else active.sequence + 1,
                artifact_id=artifact.id,
                artifact_hash=artifact.content_hash,
                fitted_model_id=fitted.id,
                fitted_state_hash=fitted.state_hash,
                promotion_decision_id=promotion.id,
            ),
            expected_previous_sequence=previous_sequence,
        )

    @staticmethod
    def _author_supports(
        author: ExecutableArtifactAuthor,
        package: ModelChallengePackage,
    ) -> bool:
        supports = getattr(author, "supports", None)
        if not callable(supports):
            return True
        return bool(supports(package))

    def _selected_authors(
        self,
        package: ModelChallengePackage,
    ) -> tuple[tuple[int, ExecutableArtifactAuthor], ...]:
        selected = tuple(
            (index, author)
            for index, author in enumerate(self.authors)
            if self._author_supports(author, package)
        )
        return selected or tuple(enumerate(self.authors))

    async def _canonical_package(
        self,
        package: ModelChallengePackage,
    ) -> ModelChallengePackage:
        """Seal author input once so retries cannot drift with a re-observation.

        A benchmark retry may reconstruct an equivalent snapshot with a different
        observation timestamp.  The challenge ID remains the same, so the first
        persisted package is the authoritative authoring and evaluation input.
        """

        selected_authors = self._selected_authors(package)
        author_keys = tuple(
            self._author_key(author, index)
            for index, author in selected_authors
        )

        # `lithops_model_challenge_packages.challenge_id` is a foreign key to
        # the lifecycle row.  The legacy challenge path created that parent,
        # but the executable path used to append the package directly.  The
        # in-memory repository hid the violation; PostgREST correctly rejected
        # every live challenge with a ConflictError before authoring began.
        challenge = await self.repository.get_model_challenge(package.challenge_id)
        if challenge is None:
            challenge = ModelChallengeRecord(
                id=package.challenge_id,
                run_id=package.run_id,
                health_signal_id=package.health_signal.id,
                base_model_version_id=package.active_model.id,
                requested_builders=author_keys,
                created_at=package.created_at,
                updated_at=package.created_at,
            )
        await self.repository.save_model_challenge(challenge)

        existing = await self.repository.get_model_challenge_package(
            package.challenge_id
        )
        if existing is not None:
            canonical = existing
        else:
            try:
                canonical = await self.repository.append_model_challenge_package(package)
            except ConflictError:
                # Another worker may have sealed the same deterministic challenge after
                # the read above. Resolve that winner instead of comparing timestamps.
                canonical = await self.repository.get_model_challenge_package(
                    package.challenge_id
                )
                if canonical is None:
                    raise
        challenge = await self.repository.get_model_challenge(package.challenge_id)
        if challenge is None:
            raise ValueError("executable challenge lifecycle record disappeared")
        if challenge.status is ModelChallengeStatus.TRIGGERED:
            await self.repository.save_model_challenge(
                challenge.model_copy(
                    update={
                        "status": ModelChallengeStatus.BUILDING,
                        "updated_at": canonical.created_at,
                    }
                )
            )
        return canonical

    async def _complete_challenge(
        self,
        package: ModelChallengePackage,
        promotion: ModelPromotionDecision,
    ) -> None:
        challenge = await self.repository.get_model_challenge(package.challenge_id)
        if challenge is None:
            raise ValueError("executable challenge lifecycle record is missing")
        if challenge.status is ModelChallengeStatus.COMPLETED:
            if challenge.decision_id != promotion.id:
                raise ConflictError(
                    f"model challenge completion conflict: {package.challenge_id}"
                )
            return
        await self.repository.save_model_challenge(
            challenge.model_copy(
                update={
                    "status": ModelChallengeStatus.COMPLETED,
                    "decision_id": promotion.id,
                    "updated_at": promotion.created_at,
                    "completed_at": promotion.created_at,
                }
            )
        )

    async def _runtime_for_active(
        self,
        active: ActiveModelAssignment | None,
    ) -> ExecutableCompanyModel:
        if active is None:
            return self.baseline
        artifact = await self.repository.get_model_artifact(active.artifact_id)
        if artifact.content_hash != active.artifact_hash:
            raise ValueError("active executable artifact hash mismatch")
        if artifact.runtime_kind is ModelRuntimeKind.TRUSTED_BASELINE:
            if artifact.id != self.baseline.artifact.id:
                raise ValueError("unknown trusted executable champion")
            return self.baseline
        return self.runtime_factory(artifact)

    def _evaluate(
        self,
        run_id: UUID,
        challenge_id: UUID,
        runtime: ExecutableCompanyModel,
        observations: tuple[TemporalObservation, ...],
        world_model: WorldModelVersion,
        seed: int,
    ) -> ArtifactEvaluationResult:
        return self.evaluator.evaluate(
            run_id=run_id,
            challenge_id=challenge_id,
            runtime=runtime,
            observations=observations,
            prior=self._prior(runtime.artifact, observations, world_model),
            stress_cases=self._stress_cases(observations),
            seed=seed,
        )

    @staticmethod
    def _stress_cases(
        observations: tuple[TemporalObservation, ...],
    ) -> tuple[ModelStressCase, ...]:
        latest = observations[-1]
        state = dict(latest.state)
        weekly_fixed_outflow = sum(
            float(state.get(name, 0.0))
            for name in (
                "marketing_spend",
                "development_spend",
                "targeted_development_spend",
                "operations_spend",
                "capacity_spend_weekly",
            )
            if isinstance(state.get(name, 0.0), int | float)
        )
        state["cash"] = max(1.0, min(float(state["cash"]), weekly_fixed_outflow * 2.0))
        action = dict(latest.action_from_previous)
        if not action:
            action = {
                "name": "stress_hold",
                "price_per_customer_weekly": float(
                    state.get(
                        "catalog_price_per_customer_weekly",
                        state.get("price_per_customer_weekly", 1.0),
                    )
                ),
                "marketing_spend": float(state.get("marketing_spend", 0.0)),
                "development_spend": float(state.get("development_spend", 0.0)),
                "operations_spend": float(state.get("operations_spend", 0.0)),
                "model_tier_a": int(state.get("model_tier_a", 1)),
                "model_tier_b": int(state.get("model_tier_b", 1)),
                "model_tier_c": int(state.get("model_tier_c", 1)),
            }
        action.setdefault(
            "price_per_customer_weekly",
            float(
                state.get(
                    "catalog_price_per_customer_weekly",
                    state.get("price_per_customer_weekly", 1.0),
                )
            ),
        )
        for name in ("marketing_spend", "development_spend", "operations_spend"):
            action.setdefault(name, float(state.get(name, 0.0)))
        for name in ("model_tier_a", "model_tier_b", "model_tier_c"):
            action.setdefault(name, int(state.get(name, 1)))
        action.setdefault("experiment_duration_weeks", 1.0)
        action.setdefault(
            "marketing_spend_after_experiment",
            float(state.get("marketing_spend", 0.0)),
        )
        action.setdefault(
            "development_spend_after_experiment",
            float(state.get("development_spend", 0.0)),
        )
        action.setdefault(
            "targeted_development_spend_weekly",
            float(state.get("targeted_development_spend", 0.0)),
        )
        action.setdefault("targeted_development_duration_weeks", 1.0)
        action.setdefault(
            "targeted_development_spend_after_experiment",
            float(state.get("targeted_development_spend", 0.0)),
        )
        action.setdefault("marketing_spend_start_after_weeks", 0.0)
        action.setdefault(
            "lead_promotion_monthly",
            float(state.get("lead_promotion_monthly", 0.0)),
        )
        action.setdefault("lead_promotion_duration_weeks", 1.0)
        action.setdefault(
            "lead_promotion_after_experiment",
            float(state.get("lead_promotion_monthly", 0.0)),
        )
        return (
            ModelStressCase(
                name="low_cash_runway",
                state=state,
                action=action,
                horizon_days=7,
            ),
        )

    async def _persist_evaluation(
        self,
        run_id: UUID,
        result: ArtifactEvaluationResult,
    ) -> None:
        await self.repository.append_model_artifact(result.artifact)
        for fitted in result.fitted_models:
            await self.repository.append_fitted_model(run_id, fitted)
        for fold in result.folds:
            await self.repository.append_temporal_evaluation_fold(fold)

    @staticmethod
    def _prior(
        artifact: ModelArtifact,
        observations: tuple[TemporalObservation, ...],
        world_model: WorldModelVersion,
    ) -> dict[str, JsonValue]:
        if artifact.runtime_kind in {
            ModelRuntimeKind.TRUSTED_BASELINE,
            ModelRuntimeKind.TYPED_COMPONENT_ASSEMBLY,
        }:
            return {"legacy_world_model": world_model.model_dump(mode="json")}
        cash = [float(item.state["cash"]) for item in observations]
        days = [item.day for item in observations]
        weekly_deltas = [
            (current_cash - previous_cash) / max((current_day - previous_day) / 7.0, 1.0)
            for previous_cash, current_cash, previous_day, current_day in zip(
                cash,
                cash[1:],
                days,
                days[1:],
                strict=False,
            )
        ]
        canonical: dict[str, JsonValue] = {
            "weekly_cash_delta": fmean(weekly_deltas) if weekly_deltas else -25_000.0,
            "marketing_cash_return": 0.5,
            "marketing_saturation_scale_weekly": 10_000.0,
            "price_elasticity": 0.2,
            "churn_sensitivity": 0.2,
            "quality_lag_weeks": 4.0,
        }
        missing = set(artifact.required_priors) - set(canonical)
        if missing:
            raise ValueError(
                "challenger requires unsupported priors: " + ", ".join(sorted(missing))
            )
        return {name: canonical[name] for name in artifact.required_priors}

    @staticmethod
    def _author_key(author: ExecutableArtifactAuthor, index: int) -> str:
        spec = getattr(author, "spec", None)
        name = getattr(spec, "name", None)
        version = getattr(spec, "version", None)
        if isinstance(name, str) and isinstance(version, str):
            return f"{name}:{version}"
        return f"{type(author).__module__}.{type(author).__qualname__}:{index}"

    @staticmethod
    def _authoring_input_hash(
        package: ModelChallengePackage,
        parent_artifact: ModelArtifact,
        author_key: str,
    ) -> str:
        payload = {
            "author_key": author_key,
            "package": package.model_dump(mode="json"),
            "parent_artifact_id": str(parent_artifact.id),
            "parent_artifact_hash": parent_artifact.content_hash,
        }
        encoded = json.dumps(
            payload,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
