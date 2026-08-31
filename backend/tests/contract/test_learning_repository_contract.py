from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, cast
from uuid import uuid4

import httpx
import pytest
from lithops.application.step_run import RunManager, StaticDecisionEngine
from lithops.benchmark.fake import FakeBenchmarkAdapter
from lithops.domain.models import RunRecord, utc_now
from lithops.infrastructure.persistence.repositories import (
    InMemoryRunRepository,
    SupabaseRunRepository,
)


class StatefulPostgrestClient:
    """Small PostgREST double used to run one contract against both repositories."""

    def __init__(self) -> None:
        self.tables: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._event_sequence = 0

    async def request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, str] | None = None,
        json: Any = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        del headers
        table = url.rsplit("/", 1)[-1]
        request = httpx.Request(method, url)
        if method == "POST" and table == "lithops_claim_run_lease":
            leases = self.tables["lithops_run_leases"]
            current = next(
                (row for row in leases if row["run_id"] == json["p_run_id"]),
                None,
            )
            can_claim = (
                current is None
                or datetime.fromisoformat(current["expires_at"])
                <= datetime.fromisoformat(json["p_now"])
                or current["owner_id"] == json["p_owner_id"]
            )
            if not can_claim:
                return httpx.Response(200, json=[], request=request)
            row = {
                "run_id": json["p_run_id"],
                "owner_id": json["p_owner_id"],
                "token": json["p_token"],
                "acquired_at": json["p_now"],
                "heartbeat_at": json["p_now"],
                "expires_at": json["p_expires_at"],
            }
            self.tables["lithops_run_leases"] = [
                lease for lease in leases if lease["run_id"] != row["run_id"]
            ] + [row]
            return httpx.Response(200, json=[row], request=request)
        if method == "POST" and table == "lithops_renew_run_lease":
            current = next(
                (
                    row
                    for row in self.tables["lithops_run_leases"]
                    if row["run_id"] == json["p_run_id"]
                    and row["token"] == json["p_token"]
                    and datetime.fromisoformat(row["expires_at"])
                    > datetime.fromisoformat(json["p_now"])
                ),
                None,
            )
            if current is None:
                return httpx.Response(200, json=[], request=request)
            current["heartbeat_at"] = json["p_now"]
            current["expires_at"] = json["p_expires_at"]
            return httpx.Response(200, json=[current], request=request)
        if method == "POST":
            row = dict(json)
            if table == "lithops_events":
                self._event_sequence += 1
                row["id"] = self._event_sequence
            self.tables[table].append(row)
            self._materialize(table, row)
            return httpx.Response(201, json=[row], request=request)

        selected = [dict(row) for row in self.tables[table]]
        filters = params or {}
        for key, value in filters.items():
            if key in {"select", "order", "limit"} or not value.startswith("eq."):
                continue
            expected = value[3:]
            selected = [row for row in selected if str(row.get(key)) == expected]

        order = filters.get("order")
        if order:
            field, direction = order.split(".", 1)
            selected.sort(key=lambda row: row[field], reverse=direction == "desc")
        if "limit" in filters:
            selected = selected[: int(filters["limit"])]

        if method == "PATCH":
            matching_ids = {row.get("id") for row in selected}
            updated: list[dict[str, Any]] = []
            for index, row in enumerate(self.tables[table]):
                if row.get("id") in matching_ids:
                    replacement = {**row, **dict(json)}
                    self.tables[table][index] = replacement
                    self._materialize(table, replacement)
                    updated.append(replacement)
            return httpx.Response(200, json=updated, request=request)

        if method == "DELETE":
            selected_tokens = {row.get("token") for row in selected}
            self.tables[table] = [
                row for row in self.tables[table] if row.get("token") not in selected_tokens
            ]
            return httpx.Response(200, json=selected, request=request)

        return httpx.Response(200, json=selected, request=request)

    def _materialize(self, table: str, row: dict[str, Any]) -> None:
        if table == "lithops_world_models":
            self.tables["lithops_world_model_relationships"] = [
                existing
                for existing in self.tables["lithops_world_model_relationships"]
                if existing["world_model_id"] != row["id"]
            ]
            for relationship in row["payload"]["relationships"]:
                self.tables["lithops_world_model_relationships"].append(
                    {
                        "world_model_id": row["id"],
                        "run_id": row["run_id"],
                        "relationship_key": relationship["key"],
                        "payload": relationship,
                    }
                )
        if table == "lithops_decisions":
            self.tables["lithops_candidate_simulations"] = [
                existing
                for existing in self.tables["lithops_candidate_simulations"]
                if existing["decision_id"] != row["id"]
            ]
            for candidate in row["payload"]["candidate_evaluations"]:
                self.tables["lithops_candidate_simulations"].append(
                    {
                        "decision_id": row["id"],
                        "run_id": row["run_id"],
                        "strategy": candidate["strategy"],
                        "robust_utility": candidate["robust_utility"],
                        "payload": candidate,
                    }
                )


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
async def test_learning_artifact_repository_contract(backend, repository) -> None:
    manager = RunManager(
        repository=repository,
        benchmark=FakeBenchmarkAdapter(),
        decision_engine=StaticDecisionEngine(),
        planning_rollouts=20,
    )
    run = await manager.create_run()
    first = await manager.step_run(run.id, request_id=f"{backend}-week-0")
    await manager.step_run(run.id, request_id=f"{backend}-week-1")

    decisions = await repository.list_decisions(run.id)
    models = await repository.list_world_models(run.id)
    predictions = await repository.list_predictions(run.id)
    outcomes = await repository.list_prediction_outcomes(run.id)
    health_signals = await repository.list_model_health_signals(run.id)

    assert len(decisions) == 2
    assert len(models) == 2
    assert len(models[0].relationships) >= 3
    assert await repository.list_world_model_relationships(models[0].id) == sorted(
        models[0].relationships,
        key=lambda relationship: relationship.key,
    )
    assert len(predictions) == 2
    assert len(outcomes) == 1
    assert len(health_signals) == 1
    assert decisions[0].candidate_evaluations
    assert await repository.list_candidate_simulations(
        run.id,
        decisions[0].id,
    ) == sorted(
        decisions[0].candidate_evaluations,
        key=lambda candidate: candidate.robust_utility,
        reverse=True,
    )
    assert decisions[0].selection_reason
    assert decisions[0].prediction_id == predictions[0].id
    assert outcomes[0].ledger_entry_id == predictions[0].id
    assert await repository.get_decision(run.id, first.decision.id) == decisions[0]

    replay = await repository.append_model_health_signal(health_signals[0])
    assert replay == health_signals[0]
    assert len(await repository.list_model_health_signals(run.id)) == 1


def lease_repository_cases():
    yield "memory", InMemoryRunRepository()
    client = StatefulPostgrestClient()
    yield "supabase", SupabaseRunRepository(
        url="https://example.supabase.co",
        secret_key="test-secret",
        client=cast(httpx.AsyncClient, client),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("backend,repository", list(lease_repository_cases()))
async def test_run_lease_repository_contract(backend, repository) -> None:
    del backend
    run = await repository.create_run(RunRecord())
    now = utc_now()

    first = await repository.claim_run_lease(
        run.id,
        "worker-a",
        now=now,
        ttl_seconds=30,
    )
    blocked = await repository.claim_run_lease(
        run.id,
        "worker-b",
        now=now + timedelta(seconds=1),
        ttl_seconds=30,
    )
    assert first is not None
    assert blocked is None

    renewed = await repository.renew_run_lease(
        run.id,
        first.token,
        now=now + timedelta(seconds=2),
        ttl_seconds=30,
    )
    assert renewed is not None
    assert renewed.expires_at > first.expires_at
    assert not await repository.release_run_lease(run.id, uuid4())
    assert await repository.release_run_lease(run.id, first.token)

    second = await repository.claim_run_lease(
        run.id,
        "worker-b",
        now=now + timedelta(seconds=3),
        ttl_seconds=30,
    )
    assert second is not None and second.owner_id == "worker-b"
