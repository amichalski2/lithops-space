"""Role-scoped agent tool boundary with denial audit events."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any
from uuid import UUID

from lithops.domain.models import EventRecord
from lithops.domain.ports import RunRepository

ToolHandler = Callable[[dict[str, Any]], Awaitable[Any]]


class AgentRole(StrEnum):
    EXECUTIVE = "executive"
    WORLD_MODEL_BUILDER = "world_model_builder"
    CANDIDATE_MODEL_BUILDER = "candidate_model_builder"


class AgentTool(StrEnum):
    READ_COMPANY_STATE = "read_company_state"
    READ_PREDICTION_HISTORY = "read_prediction_history"
    READ_CHALLENGE_PACKAGE = "read_challenge_package"
    SUBMIT_MODEL_HYPOTHESIS = "submit_model_hypothesis"
    REQUEST_HYPOTHESIS_BACKTEST = "request_hypothesis_backtest"
    REQUEST_SIMULATION = "request_simulation"
    PREPARE_ACTION_PLAN = "prepare_action_plan"
    EXECUTE_ACTION_PLAN = "execute_action_plan"
    SET_PRICES = "set_prices"
    SET_DAILY_SPEND = "set_daily_spend"
    ADVANCE_WEEK = "advance_week"
    ACTIVATE_WORLD_MODEL = "activate_world_model"


ROLE_TOOL_ALLOWLIST: dict[AgentRole, frozenset[AgentTool]] = {
    AgentRole.EXECUTIVE: frozenset(
        {
            AgentTool.READ_COMPANY_STATE,
            AgentTool.READ_PREDICTION_HISTORY,
            AgentTool.REQUEST_SIMULATION,
            AgentTool.PREPARE_ACTION_PLAN,
            AgentTool.EXECUTE_ACTION_PLAN,
            AgentTool.ADVANCE_WEEK,
        }
    ),
    AgentRole.WORLD_MODEL_BUILDER: frozenset(
        {
            AgentTool.READ_COMPANY_STATE,
            AgentTool.READ_PREDICTION_HISTORY,
            AgentTool.READ_CHALLENGE_PACKAGE,
            AgentTool.SUBMIT_MODEL_HYPOTHESIS,
            AgentTool.REQUEST_HYPOTHESIS_BACKTEST,
        }
    ),
    AgentRole.CANDIDATE_MODEL_BUILDER: frozenset(
        {
            AgentTool.READ_CHALLENGE_PACKAGE,
            AgentTool.SUBMIT_MODEL_HYPOTHESIS,
            AgentTool.REQUEST_HYPOTHESIS_BACKTEST,
        }
    ),
}


class AgentPermissionDenied(PermissionError):
    pass


class RoleScopedToolRegistry:
    def __init__(self, *, repository: RunRepository) -> None:
        self.repository = repository
        self._handlers: dict[AgentTool, ToolHandler] = {}

    def register(self, tool: AgentTool, handler: ToolHandler) -> None:
        self._handlers[tool] = handler

    async def invoke(
        self,
        *,
        run_id: UUID,
        correlation_id: str,
        role: AgentRole,
        agent_name: str,
        agent_version: str,
        tool: AgentTool,
        arguments: dict[str, Any],
    ) -> Any:
        input_hash = hashlib.sha256(
            json.dumps(arguments, default=str, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
        if tool not in ROLE_TOOL_ALLOWLIST[role]:
            await self.repository.append_event(
                EventRecord(
                    run_id=run_id,
                    type="agent.permission_denied",
                    payload={
                        "correlation_id": correlation_id,
                        "role": role.value,
                        "agent_name": agent_name,
                        "agent_version": agent_version,
                        "tool": tool.value,
                        "input_hash": input_hash,
                        "reason_code": "tool_not_allowed_for_role",
                    },
                )
            )
            raise AgentPermissionDenied(f"{role.value} cannot invoke {tool.value}")
        handler = self._handlers.get(tool)
        if handler is None:
            raise LookupError(f"no handler registered for allowed tool: {tool.value}")
        return await handler(arguments)
