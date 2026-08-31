from pathlib import Path
from uuid import uuid4

import pytest
from lithops.application.experiment_checkpoint import ExperimentCheckpoint
from lithops.domain.experiment_contracts import OBSERVATION_CONTRACT_VERSION
from lithops.domain.models import RunRecord, RunStatus
from pydantic import ValidationError

from scripts.run_ceobench_experiment import _resolve_python_command


def test_checkpoint_round_trip_and_stale_refresh() -> None:
    run = RunRecord(
        horizon_days=84,
        current_day=7,
        status=RunStatus.RUNNING,
        version=3,
    )
    first = ExperimentCheckpoint.from_run(
        run,
        model="qwen/qwen3-32b",
        benchmark_seed=42,
        target_weeks=12,
        world_model_version=1,
    )
    refreshed = ExperimentCheckpoint.from_run(
        run.model_copy(update={"current_day": 21, "version": 7}),
        model="qwen/qwen3-32b",
        benchmark_seed=42,
        target_weeks=12,
        world_model_version=3,
    )

    assert ExperimentCheckpoint.model_validate_json(first.model_dump_json()) == first
    assert refreshed.run_id == first.run_id
    assert refreshed.completed_weeks == 3
    assert refreshed.world_model_version == 3


def test_checkpoint_rejects_incompatible_resume_settings() -> None:
    run = RunRecord(id=uuid4(), horizon_days=84)
    checkpoint = ExperimentCheckpoint.from_run(
        run,
        model="qwen/qwen3-32b",
        benchmark_seed=42,
        target_weeks=12,
        world_model_version=None,
    )

    with pytest.raises(ValueError, match="model"):
        checkpoint.assert_compatible(
            run,
            provider="openrouter",
            model="another/model",
            benchmark_seed=42,
            target_weeks=12,
        )


def test_checkpoint_rejects_provider_drift_on_resume() -> None:
    run = RunRecord(id=uuid4(), horizon_days=84)
    checkpoint = ExperimentCheckpoint.from_run(
        run,
        provider="gemini",
        model="gemini-3.7-flash",
        benchmark_seed=42,
        target_weeks=12,
        world_model_version=None,
    )

    with pytest.raises(ValueError, match="provider"):
        checkpoint.assert_compatible(
            run,
            provider="openrouter",
            model="gemini-3.7-flash",
            benchmark_seed=42,
            target_weeks=12,
        )


def test_checkpoint_rejects_executive_authority_drift_on_resume() -> None:
    run = RunRecord(id=uuid4(), horizon_days=84)
    checkpoint = ExperimentCheckpoint.from_run(
        run,
        provider="gemini",
        model="gemini-3.7-flash",
        benchmark_seed=42,
        target_weeks=12,
        world_model_version=None,
        executive_authority_v2=True,
    )

    with pytest.raises(ValueError, match="executive_authority_v2"):
        checkpoint.assert_compatible(
            run,
            provider="gemini",
            model="gemini-3.7-flash",
            benchmark_seed=42,
            target_weeks=12,
            executive_authority_v2=False,
        )


def test_python_command_anchors_path_without_resolving_virtualenv_shim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpreter = Path("ceobench-venv") / "bin" / "python"
    anchored = Path("anchored") / "ceobench-venv" / "bin" / "python"

    def fail_if_resolved(self: Path, *args, **kwargs) -> Path:
        raise AssertionError("virtualenv interpreter paths must not be resolved")

    monkeypatch.setattr(Path, "resolve", fail_if_resolved)
    monkeypatch.setattr(Path, "absolute", lambda self: anchored)
    monkeypatch.setattr(Path, "is_file", lambda self: True)

    assert _resolve_python_command(str(interpreter)) == str(anchored)


def test_checkpoint_rejects_non_week_boundary() -> None:
    with pytest.raises(ValueError, match="seven-day"):
        ExperimentCheckpoint(
            run_id=uuid4(),
            model="qwen/qwen3-32b",
            benchmark_seed=42,
            target_weeks=12,
            current_day=8,
            completed_weeks=1,
            run_status=RunStatus.RUNNING,
            run_version=1,
        )


def test_checkpoint_rejects_legacy_schema_instead_of_silently_resuming() -> None:
    run = RunRecord(id=uuid4(), horizon_days=84)
    payload = ExperimentCheckpoint.from_run(
        run,
        model="qwen/qwen3-32b",
        benchmark_seed=42,
        target_weeks=12,
        world_model_version=None,
    ).model_dump(mode="json")
    payload["schema_version"] = "lithops-experiment-checkpoint-v1"

    with pytest.raises(ValidationError, match="schema_version"):
        ExperimentCheckpoint.model_validate(payload)


def test_checkpoint_rejects_observation_contract_drift() -> None:
    run = RunRecord(id=uuid4(), horizon_days=84)
    checkpoint = ExperimentCheckpoint.from_run(
        run,
        model="qwen/qwen3-32b",
        benchmark_seed=42,
        target_weeks=12,
        world_model_version=None,
    )

    with pytest.raises(ValueError, match="observation_contract_version"):
        checkpoint.assert_compatible(
            run,
            provider="openrouter",
            model="qwen/qwen3-32b",
            benchmark_seed=42,
            target_weeks=12,
            observation_contract_version=OBSERVATION_CONTRACT_VERSION + "-drift",
        )
