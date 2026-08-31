from __future__ import annotations

from fastapi.testclient import TestClient
from lithops.api.main import create_app
from lithops.application.step_run import RunManager, StaticDecisionEngine
from lithops.benchmark.fake import FakeBenchmarkAdapter
from lithops.infrastructure.persistence.repositories import InMemoryRunRepository


class FailAfterPredictionAppend(InMemoryRunRepository):
    def __init__(self) -> None:
        super().__init__()
        self._fail_prediction_once = True

    async def append_prediction(self, prediction):
        saved = await super().append_prediction(prediction)
        if self._fail_prediction_once:
            self._fail_prediction_once = False
            raise RuntimeError("simulated crash after prediction persistence")
        return saved


def make_client() -> tuple[TestClient, FakeBenchmarkAdapter]:
    adapter = FakeBenchmarkAdapter()
    manager = RunManager(
        repository=InMemoryRunRepository(),
        benchmark=adapter,
        decision_engine=StaticDecisionEngine(),
    )
    return TestClient(create_app(manager)), adapter


def test_health() -> None:
    client, _ = make_client()
    assert client.get("/health").json() == {"status": "ok"}


def test_replay_only_deployment_refuses_to_advance_without_affecting_read_api() -> None:
    adapter = FakeBenchmarkAdapter()
    manager = RunManager(
        repository=InMemoryRunRepository(),
        benchmark=adapter,
        decision_engine=StaticDecisionEngine(),
    )
    client = TestClient(create_app(manager, replay_only=True))
    run_id = client.post("/runs").json()["id"]

    assert client.get(f"/runs/{run_id}").status_code == 200
    response = client.post(
        f"/runs/{run_id}/step",
        headers={"Idempotency-Key": "replay-only-step"},
    )

    assert response.status_code == 403
    assert "read-only replay" in response.json()["detail"]
    assert adapter.advance_week_calls == 0


def test_create_step_replay_and_events() -> None:
    client, adapter = make_client()
    created = client.post("/runs")
    assert created.status_code == 201
    run_id = created.json()["id"]

    missing_key = client.post(f"/runs/{run_id}/step")
    assert missing_key.status_code == 422

    headers = {"Idempotency-Key": "api-week-one"}
    first = client.post(f"/runs/{run_id}/step", headers=headers)
    replay = client.post(f"/runs/{run_id}/step", headers=headers)

    assert first.status_code == 200
    assert first.json()["run"]["current_day"] == 7
    assert first.json()["replayed"] is False
    assert replay.status_code == 200
    assert replay.json()["replayed"] is True
    assert adapter.advance_week_calls == 1

    run = client.get(f"/runs/{run_id}")
    events = client.get(f"/runs/{run_id}/events")
    assert run.status_code == 200
    assert run.json()["current_day"] == 7
    assert events.status_code == 200
    assert events.json()[-1]["type"] == "decision.committed"


def test_learning_read_api_reconstructs_a_decision() -> None:
    client, _ = make_client()
    run_id = client.post("/runs").json()["id"]
    first = client.post(
        f"/runs/{run_id}/step",
        headers={"Idempotency-Key": "read-api-week-0"},
    ).json()
    client.post(
        f"/runs/{run_id}/step",
        headers={"Idempotency-Key": "read-api-week-1"},
    )

    state = client.get(f"/runs/{run_id}/state")
    decisions = client.get(f"/runs/{run_id}/decisions")
    explanation = client.get(
        f"/runs/{run_id}/decisions/{first['decision']['id']}"
    )
    world_model = client.get(f"/runs/{run_id}/world-model")
    predictions = client.get(f"/runs/{run_id}/predictions")
    report = client.get(f"/runs/{run_id}/report")

    assert state.json()["current_day"] == 14
    assert len(decisions.json()) == 2
    assert explanation.status_code == 200
    detail = explanation.json()
    assert len(detail["decision"]["candidate_evaluations"]) >= 3
    assert detail["decision"]["selection_reason"]
    assert detail["prediction"]["outcomes"][0]["actual"]["observed_day"] == 7
    assert detail["world_model"]["relationships"]
    assert world_model.json()["version"] == 2
    assert len(predictions.json()) == 2
    assert report.json()["matured_outcome_count"] == 1
    assert report.json()["latest_model_health"] is not None


def test_unknown_run_returns_not_found() -> None:
    client, _ = make_client()
    response = client.get("/runs/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


def test_run_lifecycle_control_endpoints_enforce_safe_pause() -> None:
    client, _ = make_client()
    run_id = client.post("/runs").json()["id"]

    assert client.post(f"/runs/{run_id}/pause").status_code == 422
    started = client.post(f"/runs/{run_id}/start")
    assert started.status_code == 200
    assert started.json()["status"] == "bootstrapping"
    assert client.post(f"/runs/{run_id}/start").json()["version"] == started.json()[
        "version"
    ]

    stepped = client.post(
        f"/runs/{run_id}/step",
        headers={"Idempotency-Key": "lifecycle-week-0"},
    )
    assert stepped.status_code == 200
    assert stepped.json()["run"]["status"] == "running"

    pausing = client.post(f"/runs/{run_id}/pause")
    assert pausing.status_code == 200
    assert pausing.json()["status"] == "pausing"
    blocked_step = client.post(
        f"/runs/{run_id}/step",
        headers={"Idempotency-Key": "lifecycle-week-1"},
    )
    assert blocked_step.status_code == 422
    assert "pausing" in blocked_step.json()["detail"]
    assert client.post(f"/runs/{run_id}/resume").status_code == 422


def test_api_retry_recovers_prediction_before_any_external_action() -> None:
    repository = FailAfterPredictionAppend()
    adapter = FakeBenchmarkAdapter()
    manager = RunManager(
        repository=repository,
        benchmark=adapter,
        decision_engine=StaticDecisionEngine(),
        planning_rollouts=20,
    )
    client = TestClient(create_app(manager), raise_server_exceptions=False)
    run_id = client.post("/runs").json()["id"]
    headers = {"Idempotency-Key": "prediction-crash"}

    failed = client.post(f"/runs/{run_id}/step", headers=headers)
    assert failed.status_code == 500
    assert adapter.execute_action_calls == 0
    assert adapter.advance_week_calls == 0

    recovered = client.post(f"/runs/{run_id}/step", headers=headers)
    assert recovered.status_code == 200
    assert recovered.json()["run"]["current_day"] == 7
    assert adapter.execute_action_calls == 5
    assert adapter.advance_week_calls == 1

    events = client.get(f"/runs/{run_id}/events").json()
    prediction_events = [
        event for event in events if event["type"] == "prediction.committed"
    ]
    assert len(prediction_events) == 1
