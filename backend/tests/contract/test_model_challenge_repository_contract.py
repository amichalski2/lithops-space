from __future__ import annotations

from typing import cast

import httpx
import pytest
from lithops.application.model_challenge import ModelChallengeOrchestrator
from lithops.domain.model_challenge import (
    ModelChallengeResolution,
    ParameterDirection,
)
from lithops.infrastructure.persistence.repositories import (
    InMemoryRunRepository,
    SupabaseRunRepository,
)

from backend.tests.contract.test_learning_repository_contract import (
    StatefulPostgrestClient,
)
from backend.tests.unit.test_hypothesis_backtest import challenge_package
from backend.tests.unit.test_model_challenge_orchestrator import acquisition_builder


def repository_cases():
    yield "memory", InMemoryRunRepository()
    client = StatefulPostgrestClient()
    yield "supabase", SupabaseRunRepository(
        url="https://example.supabase.co",
        secret_key="test-secret",
        client=cast(httpx.AsyncClient, client),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("backend,repository", list(repository_cases()))
async def test_model_challenge_lifecycle_repository_contract(backend, repository) -> None:
    del backend
    package = challenge_package()
    await repository.append_world_model(package.active_model, expected_latest_version=None)
    orchestrator = ModelChallengeOrchestrator(
        repository=repository,
        builders=(
            acquisition_builder("acquisition_builder", ParameterDirection.DECREASE),
            acquisition_builder("challenger_builder", ParameterDirection.INCREASE),
        ),
    )

    first = await orchestrator.run(package)
    replay = await orchestrator.run(package)

    assert first == replay
    assert first.decision.resolution is ModelChallengeResolution.ACCEPTED
    assert await repository.get_model_challenge(package.challenge_id) == first.challenge
    assert await repository.get_model_challenge_package(package.challenge_id) == package
    assert len(await repository.list_model_builder_proposals(package.challenge_id)) == 2
    assert len(await repository.list_hypothesis_backtests(package.challenge_id)) == 2
    calls = await repository.list_model_builder_calls(package.challenge_id)
    assert len(calls) == 2
    assert all(call.input_hash and "secret" not in call.model_dump_json() for call in calls)
    assert (
        await repository.get_model_challenge_decision(package.challenge_id)
        == first.decision
    )
    events = await repository.list_events(package.run_id)
    assert {event.type for event in events} >= {
        "model_challenge.started",
        "model_builder.completed",
        "hypothesis.backtested",
        "model_challenge.completed",
    }
    assert len({event.id for event in events}) == len(events)
