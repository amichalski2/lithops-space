from __future__ import annotations

from typing import Any, cast

import httpx
import pytest
from lithops.domain.errors import ConflictError
from lithops.domain.executable_model import (
    CompanyModelFitRequest,
    FittedModel,
    ModelArtifact,
    ModelRuntimeKind,
)
from lithops.domain.model_registry import (
    ActiveModelAssignment,
    ModelArtifactAuthoringReceipt,
    ModelPromotionDecision,
    PromotionDisposition,
    SandboxExecutionRecord,
    SandboxExecutionStatus,
    SandboxOperation,
    TemporalEvaluationFold,
)
from lithops.domain.models import RunRecord
from lithops.infrastructure.persistence.repositories import (
    InMemoryRunRepository,
    SupabaseRunRepository,
)

from tests.contract.test_learning_repository_contract import StatefulPostgrestClient


class CappedPostgrestClient(StatefulPostgrestClient):
    """Emulate Supabase's 1,000-row response cap with offset pagination."""

    async def request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, str] | None = None,
        json: Any = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        table = url.rsplit("/", 1)[-1]
        if method != "GET" or table != "lithops_temporal_evaluation_folds":
            return await super().request(
                method,
                url,
                params=params,
                json=json,
                headers=headers,
            )
        filters = dict(params or {})
        offset = int(filters.pop("offset", "0"))
        requested_limit = int(filters.pop("limit", "1000"))
        response = await super().request(
            method,
            url,
            params=filters,
            json=json,
            headers=headers,
        )
        rows = response.json()
        limit = min(requested_limit, 1_000)
        request = httpx.Request(method, url)
        return httpx.Response(
            200,
            json=rows[offset : offset + limit],
            request=request,
        )


def _artifact(name: str, *, candidate: bool) -> ModelArtifact:
    values: dict[str, Any] = {
        "name": name,
        "runtime_kind": (
            ModelRuntimeKind.SANDBOXED_PYTHON if candidate else ModelRuntimeKind.TRUSTED_BASELINE
        ),
        "scope": "company",
        "hypothesis": f"{name} hypothesis",
        "authoring_agent": "registry-contract",
        "provider": "test",
        "model_name": "deterministic-test",
        "prompt_version": "registry-v1",
    }
    if candidate:
        values["source_code"] = (
            "def fit(history, prior, seed):\n    return prior\n\n"
            "def predict(fitted, state, action, horizons_days, n_samples, seed):\n"
            "    return {'samples': []}\n\n"
            "def diagnostics(fitted):\n    return {}\n"
        )
    else:
        values["trusted_entrypoint"] = "lithops.model_runtime.baseline:FixedBaselineCompanyModel"
    return ModelArtifact.create(**values)


def _fitted(artifact: ModelArtifact, observation_id: str) -> FittedModel:
    request = CompanyModelFitRequest(
        observation_ids=(observation_id,),
        training_start_day=0,
        training_end_day=7,
        history=({"day": 7, "cash": 1000.0},),
        prior={"weekly_cash_delta": -50.0},
        seed=7,
    )
    return FittedModel.create(
        artifact=artifact,
        request=request,
        fitted_state={"weekly_cash_delta": -50.0},
    )


def repository_cases():
    yield "memory", InMemoryRunRepository()
    client = StatefulPostgrestClient()
    yield (
        "supabase",
        SupabaseRunRepository(
            url="https://example.supabase.co",
            secret_key="test-secret",
            client=cast(httpx.AsyncClient, client),
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("backend,repository", list(repository_cases()))
async def test_executable_model_registry_is_immutable_and_replay_safe(
    backend: str,
    repository: InMemoryRunRepository | SupabaseRunRepository,
) -> None:
    del backend
    run = await repository.create_run(RunRecord())
    champion_artifact = _artifact("registry-baseline-v1", candidate=False)
    candidate_artifact = _artifact("registry-candidate-v1", candidate=True)
    champion_fitted = _fitted(champion_artifact, "obs-champion")
    candidate_fitted = _fitted(candidate_artifact, "obs-candidate")

    for artifact in (champion_artifact, candidate_artifact):
        assert await repository.append_model_artifact(artifact) == artifact
        assert await repository.append_model_artifact(artifact) == artifact

    authoring = ModelArtifactAuthoringReceipt.create(
        challenge_id=run.id,
        run_id=run.id,
        author_key="registry-contract:1.0",
        artifact_id=candidate_artifact.id,
        artifact_hash=candidate_artifact.content_hash,
        input_hash="d" * 64,
    )
    assert await repository.append_model_artifact_authoring_receipt(authoring) == authoring
    assert await repository.append_model_artifact_authoring_receipt(authoring) == authoring
    assert await repository.list_model_artifact_authoring_receipts(
        run.id,
        run.id,
    ) == [authoring]
    for fitted in (champion_fitted, candidate_fitted):
        assert await repository.append_fitted_model(run.id, fitted) == fitted
        assert await repository.append_fitted_model(run.id, fitted) == fitted

    denied = SandboxExecutionRecord.create(
        run_id=run.id,
        idempotency_key="candidate-v1:validate",
        artifact_id=candidate_artifact.id,
        artifact_hash=candidate_artifact.content_hash,
        operation=SandboxOperation.VALIDATE,
        status=SandboxExecutionStatus.DENIED,
        input_hash="a" * 64,
        policy_denial_codes=("import_forbidden",),
        runtime_ms=2,
        error_code="sandbox_policy_denied",
    )
    completed = SandboxExecutionRecord.create(
        run_id=run.id,
        idempotency_key="candidate-v1:fit",
        artifact_id=candidate_artifact.id,
        artifact_hash=candidate_artifact.content_hash,
        fitted_model_id=candidate_fitted.id,
        operation=SandboxOperation.FIT,
        status=SandboxExecutionStatus.COMPLETED,
        input_hash="b" * 64,
        output_hash="c" * 64,
        runtime_ms=5,
    )
    for execution in (denied, completed):
        assert await repository.append_sandbox_execution(execution) == execution
        assert await repository.append_sandbox_execution(execution) == execution
    assert await repository.list_sandbox_executions(run.id) == [denied, completed]

    fold = TemporalEvaluationFold.create(
        run_id=run.id,
        challenge_id=run.id,
        artifact_id=candidate_artifact.id,
        artifact_hash=candidate_artifact.content_hash,
        fitted_model_id=candidate_fitted.id,
        fold_index=0,
        evaluation_seed=11,
        training_start_day=0,
        training_end_day=7,
        holdout_start_day=14,
        holdout_end_day=21,
        sample_count=2,
        predictive_score=0.10,
        complexity_penalty=0.01,
        runtime_penalty=0.02,
        total_score=0.13,
        invariant_gate_passed=True,
        metrics={"cash_mae": 100.0},
    )
    assert await repository.append_temporal_evaluation_fold(fold) == fold
    assert await repository.append_temporal_evaluation_fold(fold) == fold

    promotion = ModelPromotionDecision.create(
        challenge_id=run.id,
        run_id=run.id,
        decision_day=21,
        champion_artifact_id=champion_artifact.id,
        champion_fitted_model_id=champion_fitted.id,
        candidate_artifact_id=candidate_artifact.id,
        candidate_fitted_model_id=candidate_fitted.id,
        evaluation_fold_ids=(fold.id,),
        disposition=PromotionDisposition.PROMOTED,
        reason_code="candidate_materially_better",
        evidence={"score_delta": 0.2},
    )
    assert await repository.append_model_promotion_decision(promotion) == promotion
    assert await repository.append_model_promotion_decision(promotion) == promotion
    assert (
        await repository.get_model_promotion_decision_for_challenge(
            run.id,
            run.id,
        )
        == promotion
    )

    assignment = ActiveModelAssignment.create(
        run_id=run.id,
        sequence=1,
        artifact_id=candidate_artifact.id,
        artifact_hash=candidate_artifact.content_hash,
        fitted_model_id=candidate_fitted.id,
        fitted_state_hash=candidate_fitted.state_hash,
        promotion_decision_id=promotion.id,
    )
    assert (
        await repository.activate_model(
            assignment,
            expected_previous_sequence=None,
        )
        == assignment
    )
    assert (
        await repository.activate_model(
            assignment,
            expected_previous_sequence=None,
        )
        == assignment
    )
    assert await repository.get_active_model(run.id) == assignment
    assert await repository.list_model_activations(run.id) == [assignment]
    assert await repository.get_model_artifact(assignment.artifact_id) == candidate_artifact
    assert await repository.get_fitted_model(run.id, assignment.fitted_model_id) == candidate_fitted

    duplicate_promotion = ActiveModelAssignment.create(
        run_id=run.id,
        sequence=2,
        artifact_id=candidate_artifact.id,
        artifact_hash=candidate_artifact.content_hash,
        fitted_model_id=candidate_fitted.id,
        fitted_state_hash=candidate_fitted.state_hash,
        promotion_decision_id=promotion.id,
    )
    with pytest.raises(ConflictError, match="already activated"):
        await repository.activate_model(
            duplicate_promotion,
            expected_previous_sequence=1,
        )


@pytest.mark.asyncio
async def test_promotion_resolves_fold_beyond_supabase_row_cap() -> None:
    client = CappedPostgrestClient()
    repository = SupabaseRunRepository(
        url="https://example.supabase.co",
        secret_key="test-secret",
        client=cast(httpx.AsyncClient, client),
    )
    run = await repository.create_run(RunRecord())
    champion_artifact = _artifact("pagination-baseline-v1", candidate=False)
    candidate_artifact = _artifact("pagination-candidate-v1", candidate=True)
    champion_fitted = _fitted(champion_artifact, "pagination-champion")
    candidate_fitted = _fitted(candidate_artifact, "pagination-candidate")
    for artifact in (champion_artifact, candidate_artifact):
        await repository.append_model_artifact(artifact)
    for fitted in (champion_fitted, candidate_fitted):
        await repository.append_fitted_model(run.id, fitted)

    challenge_id = run.id
    folds = [
        TemporalEvaluationFold.create(
            run_id=run.id,
            challenge_id=challenge_id,
            artifact_id=candidate_artifact.id,
            artifact_hash=candidate_artifact.content_hash,
            fitted_model_id=candidate_fitted.id,
            fold_index=index,
            evaluation_seed=91,
            training_start_day=0,
            training_end_day=7,
            holdout_start_day=14,
            holdout_end_day=14,
            sample_count=2,
            predictive_score=0.10,
            complexity_penalty=0.01,
            runtime_penalty=0.02,
            total_score=0.13,
            invariant_gate_passed=True,
            metrics={"horizon_days": 7},
        )
        for index in range(1_001)
    ]
    client.tables["lithops_temporal_evaluation_folds"] = [
        {
            "id": str(fold.id),
            "run_id": str(fold.run_id),
            "challenge_id": str(fold.challenge_id),
            "artifact_id": str(fold.artifact_id),
            "fitted_model_id": str(fold.fitted_model_id),
            "fold_index": fold.fold_index,
            "evaluation_seed": fold.evaluation_seed,
            "payload": fold.model_dump(mode="json"),
            "created_at": fold.created_at.isoformat(),
        }
        for fold in folds
    ]

    listed = await repository.list_temporal_evaluation_folds(
        run.id,
        challenge_id=challenge_id,
    )
    assert len(listed) == 1_001
    assert listed[-1].id == folds[-1].id

    promotion = ModelPromotionDecision.create(
        challenge_id=challenge_id,
        run_id=run.id,
        decision_day=266,
        champion_artifact_id=champion_artifact.id,
        champion_fitted_model_id=champion_fitted.id,
        candidate_artifact_id=candidate_artifact.id,
        candidate_fitted_model_id=candidate_fitted.id,
        evaluation_fold_ids=(folds[-1].id,),
        disposition=PromotionDisposition.PROMOTED,
        reason_code="candidate_materially_better",
    )
    assert await repository.append_model_promotion_decision(promotion) == promotion
