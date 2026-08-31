from __future__ import annotations

import pytest
from lithops.agents.common import (
    AgentPermissionDenied,
    AgentRole,
    AgentTool,
    RoleScopedToolRegistry,
)
from lithops.domain.models import RunRecord
from lithops.infrastructure.persistence.repositories import InMemoryRunRepository


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool",
    [
        AgentTool.SET_PRICES,
        AgentTool.SET_DAILY_SPEND,
        AgentTool.ADVANCE_WEEK,
        AgentTool.EXECUTE_ACTION_PLAN,
        AgentTool.ACTIVATE_WORLD_MODEL,
    ],
)
async def test_candidate_builder_is_denied_before_privileged_handler(tool: AgentTool) -> None:
    repository = InMemoryRunRepository()
    run_id = (await repository.create_run(RunRecord())).id
    registry = RoleScopedToolRegistry(repository=repository)
    privileged_calls = 0

    async def privileged_handler(arguments):
        nonlocal privileged_calls
        privileged_calls += 1
        return arguments

    registry.register(tool, privileged_handler)
    with pytest.raises(AgentPermissionDenied, match=tool.value):
        await registry.invoke(
            run_id=run_id,
            correlation_id="challenge-1:builder-1",
            role=AgentRole.CANDIDATE_MODEL_BUILDER,
            agent_name="acquisition_builder",
            agent_version="1.0",
            tool=tool,
            arguments={"secret": "must-not-appear", "amount": 1_000},
        )

    assert privileged_calls == 0
    event = (await repository.list_events(run_id))[-1]
    assert event.type == "agent.permission_denied"
    assert event.payload["tool"] == tool.value
    assert event.payload["role"] == "candidate_model_builder"
    assert len(event.payload["input_hash"]) == 64
    assert "secret" not in str(event.payload)


@pytest.mark.asyncio
async def test_candidate_builder_can_request_a_backtest() -> None:
    repository = InMemoryRunRepository()
    run_id = (await repository.create_run(RunRecord())).id
    registry = RoleScopedToolRegistry(repository=repository)
    calls = 0

    async def backtest_handler(arguments):
        nonlocal calls
        calls += 1
        return {"supported": True, **arguments}

    registry.register(AgentTool.REQUEST_HYPOTHESIS_BACKTEST, backtest_handler)
    result = await registry.invoke(
        run_id=run_id,
        correlation_id="challenge-1:builder-1",
        role=AgentRole.CANDIDATE_MODEL_BUILDER,
        agent_name="acquisition_builder",
        agent_version="1.0",
        tool=AgentTool.REQUEST_HYPOTHESIS_BACKTEST,
        arguments={"proposal_id": "proposal-1"},
    )

    assert calls == 1
    assert result["supported"] is True
    assert await repository.list_events(run_id) == []
