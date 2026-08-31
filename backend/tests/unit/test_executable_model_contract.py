from __future__ import annotations

from uuid import UUID

import pytest
from lithops.domain.executable_model import (
    CompanyModelFitRequest,
    FittedModel,
    ModelArtifact,
    ModelArtifactAssertion,
    ModelArtifactTestCase,
    ModelEntrypoint,
    ModelRuntimeKind,
)
from pydantic import ValidationError


def artifact() -> ModelArtifact:
    return ModelArtifact.create(
        name="cash-trend-v1",
        runtime_kind=ModelRuntimeKind.SANDBOXED_PYTHON,
        scope="cash",
        hypothesis="Recent operating cash delta persists over the next few weeks.",
        authoring_agent="cash_model_builder",
        provider="openrouter",
        model_name="qwen/qwen3-32b",
        prompt_version="cash-model-builder-v1",
        source_code=(
            "def fit(history, prior, seed):\n"
            "    return {'weekly_delta': prior['weekly_delta']}\n\n"
            "def predict(fitted, state, action, horizons_days, n_samples, seed):\n"
            "    return {'samples': []}\n\n"
            "def diagnostics(fitted):\n"
            "    return {'weekly_delta': fitted['weekly_delta']}\n"
        ),
        tests=(
            ModelArtifactTestCase(
                name="fit_preserves_prior",
                entrypoint=ModelEntrypoint.FIT,
                arguments={"history": [{"day": 0}], "prior": {"weekly_delta": -100}, "seed": 1},
                assertions=(
                    ModelArtifactAssertion(
                        path="weekly_delta",
                        operator="equals",
                        expected=-100,
                    ),
                ),
            ),
        ),
    )


def fit_request() -> CompanyModelFitRequest:
    return CompanyModelFitRequest(
        observation_ids=("observation:run:0", "observation:run:7"),
        training_start_day=0,
        training_end_day=7,
        history=({"day": 0, "cash": 1000}, {"day": 7, "cash": 900}),
        prior={"weekly_delta": -100},
        seed=4,
    )


def test_model_artifact_has_reproducible_content_identity() -> None:
    first = artifact()
    second = artifact()

    assert first.id == second.id
    assert first.content_hash == second.content_hash
    assert len(first.content_hash) == 64
    assert first.id != UUID(int=0)


def test_artifact_assertion_normalizes_only_equal_transport_alias() -> None:
    assertion = ModelArtifactAssertion(
        path="cash",
        operator="equal",
        expected=100,
    )

    assert assertion.operator.value == "equals"
    with pytest.raises(ValidationError):
        ModelArtifactAssertion(path="cash", operator="gte", expected=100)


def test_model_artifact_rejects_tampered_content_and_runtime_conflicts() -> None:
    original = artifact()
    tampered = original.model_dump()
    tampered["hypothesis"] = "Changed after hashing."

    with pytest.raises(ValidationError, match="content hash"):
        ModelArtifact.model_validate(tampered)

    with pytest.raises(ValidationError, match="trusted entrypoint"):
        ModelArtifact.create(
            name="broken-baseline",
            runtime_kind=ModelRuntimeKind.TRUSTED_BASELINE,
            scope="full_company",
            hypothesis="Invalid baseline definition.",
            authoring_agent="core",
            provider="deterministic",
            model_name="broken",
            prompt_version="not_applicable",
        )


def test_fitted_model_identity_covers_training_window_and_state() -> None:
    model_artifact = artifact()
    request = fit_request()
    first = FittedModel.create(
        artifact=model_artifact,
        request=request,
        fitted_state={"weekly_delta": -100},
    )
    second = FittedModel.create(
        artifact=model_artifact,
        request=request,
        fitted_state={"weekly_delta": -100},
    )

    assert first.id == second.id
    assert first.state_hash == second.state_hash

    tampered = first.model_dump()
    tampered["fitted_state"] = {"weekly_delta": 1_000_000}
    with pytest.raises(ValidationError, match="identity"):
        FittedModel.model_validate(tampered)


def test_fit_request_rejects_future_reversed_or_duplicate_evidence() -> None:
    with pytest.raises(ValidationError, match="cannot precede"):
        fit_request().model_copy(
            update={"training_start_day": 14, "training_end_day": 7}
        ).model_validate(
            fit_request()
            .model_copy(update={"training_start_day": 14, "training_end_day": 7})
            .model_dump()
        )

    with pytest.raises(ValidationError, match="must be unique"):
        CompanyModelFitRequest(
            observation_ids=("same", "same"),
            training_start_day=0,
            training_end_day=7,
            history=({"day": 0},),
        )
