from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from uuid import uuid5

import pytest
from lithops.application.model_challenge import (
    DeterministicExecutiveAuthority,
    ModelChallengeOrchestrator,
    proposals_are_compatible,
)
from lithops.domain.model_challenge import (
    ChallengeParameterSensitivity,
    HypothesisEvidenceKind,
    HypothesisEvidenceReference,
    HypothesisFamily,
    ModelBuilderProposal,
    ModelChallengeResolution,
    ModelChallengeStatus,
    ParameterAdjustmentProposal,
    ParameterDirection,
    ParameterStepSize,
    WorldModelHypothesisDiff,
)
from lithops.domain.world_model import WorldModelParameterName
from lithops.infrastructure.persistence.repositories import InMemoryRunRepository

from backend.tests.unit.test_hypothesis_backtest import challenge_package


@dataclass
class FakeProvider:
    model_id: str = "fake/free-builder"


class FakeBuilder:
    provider_name = "fake"

    def __init__(
        self,
        *,
        name: str,
        family: HypothesisFamily,
        parameter: WorldModelParameterName,
        direction: ParameterDirection,
        failures: tuple[str, ...] = (),
    ) -> None:
        self.spec = SimpleNamespace(
            name=name,
            version="1.0",
            prompt_version=f"{name}-prompt-v1",
        )
        self.provider = FakeProvider()
        self.family = family
        self.parameter = parameter
        self.direction = direction
        self.failures = list(failures)
        self.calls = 0

    async def propose(self, package):
        self.calls += 1
        if self.failures:
            failure = self.failures.pop(0)
            if failure == "timeout":
                await asyncio.sleep(0.03)
            if failure == "invalid":
                raise ValueError("invalid fake output")
        return ModelBuilderProposal(
            id=uuid5(package.challenge_id, f"proposal:{self.spec.name}"),
            challenge_id=package.challenge_id,
            builder_name=self.spec.name,
            builder_version=self.spec.version,
            prompt_version=self.spec.prompt_version,
            provider="fake",
            model_name=self.provider.model_id,
            family=self.family,
            summary=f"{self.spec.name} bounded explanation",
            rationale="The signed residual repeats across the supplied history.",
            diff=WorldModelHypothesisDiff(
                parameter_adjustments=(
                    ParameterAdjustmentProposal(
                        parameter_name=self.parameter,
                        direction=self.direction,
                        step_size=ParameterStepSize.MEDIUM,
                    ),
                ),
            ),
            evidence=(
                HypothesisEvidenceReference(
                    kind=HypothesisEvidenceKind.OBSERVATION,
                    reference=package.observations[0].reference,
                    observed_day=package.observations[0].day,
                ),
                HypothesisEvidenceReference(
                    kind=HypothesisEvidenceKind.PREDICTION_OUTCOME,
                    reference=f"prediction-outcome:{package.residuals[-1].outcome_id}",
                    observed_day=package.residuals[-1].observed_day,
                ),
            ),
            created_at=package.created_at,
        )


def acquisition_builder(name: str, direction: ParameterDirection, **kwargs) -> FakeBuilder:
    return FakeBuilder(
        name=name,
        family=HypothesisFamily.ACQUISITION_EFFICIENCY,
        parameter=WorldModelParameterName.MARKETING_SATURATION,
        direction=direction,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_supported_winner_activates_one_child_and_replay_calls_nothing_twice() -> None:
    package = challenge_package()
    repository = InMemoryRunRepository()
    await repository.append_world_model(
        package.active_model, expected_latest_version=None
    )
    winner = acquisition_builder("acquisition_builder", ParameterDirection.DECREASE)
    loser = acquisition_builder("capacity_builder", ParameterDirection.INCREASE)
    orchestrator = ModelChallengeOrchestrator(
        repository=repository,
        builders=(winner, loser),
    )

    first = await orchestrator.run(package)
    second = await orchestrator.run(package)

    assert first == second
    assert first.challenge.status is ModelChallengeStatus.COMPLETED
    assert first.decision.resolution is ModelChallengeResolution.ACCEPTED
    assert first.decision.selected_proposal_ids == (
        uuid5(package.challenge_id, "proposal:acquisition_builder"),
    )
    assert first.activated_model.based_on_version_id == package.active_model.id
    assert first.activated_model.version == package.active_model.version + 1
    assert first.activated_model.changes[0].update_method == "model_challenge_backtest_v1"
    assert winner.calls == loser.calls == 1
    assert len(await repository.list_model_builder_proposals(package.challenge_id)) == 2
    assert len(await repository.list_hypothesis_backtests(package.challenge_id)) == 2
    event_types = [item.type for item in await repository.list_events(package.run_id)]
    assert event_types.count("model_challenge.started") == 1
    assert event_types.count("model_challenge.completed") == 1


@pytest.mark.asyncio
async def test_timeout_and_invalid_output_retry_once_then_continue_with_partial_fleet() -> None:
    package = challenge_package()
    repository = InMemoryRunRepository()
    await repository.append_world_model(
        package.active_model, expected_latest_version=None
    )
    recovered = acquisition_builder(
        "acquisition_builder",
        ParameterDirection.DECREASE,
        failures=("timeout",),
    )
    failed = acquisition_builder(
        "capacity_builder",
        ParameterDirection.INCREASE,
        failures=("invalid", "invalid"),
    )
    orchestrator = ModelChallengeOrchestrator(
        repository=repository,
        builders=(recovered, failed),
        builder_timeout_seconds=0.01,
    )

    outcome = await orchestrator.run(package)

    assert outcome.challenge.status is ModelChallengeStatus.COMPLETED
    assert outcome.activated_model is not None
    assert recovered.calls == failed.calls == 2
    receipts = await repository.list_model_builder_calls(package.challenge_id)
    assert [item.status.value for item in receipts] == [
        "timed_out",
        "completed",
        "invalid_output",
        "invalid_output",
    ]
    assert all(len(item.input_hash) == 64 for item in receipts)
    assert all(item.error_code != "invalid fake output" for item in receipts)


@pytest.mark.asyncio
async def test_recovery_does_not_repeat_two_persisted_failed_builder_attempts() -> None:
    package = challenge_package()
    repository = InMemoryRunRepository()
    first = acquisition_builder(
        "first_builder",
        ParameterDirection.INCREASE,
        failures=("invalid", "invalid"),
    )
    second = acquisition_builder(
        "second_builder",
        ParameterDirection.INCREASE,
        failures=("invalid", "invalid"),
    )
    orchestrator = ModelChallengeOrchestrator(
        repository=repository,
        builders=(first, second),
    )

    failed = await orchestrator.run(package)
    replay = await orchestrator.run(package)

    assert failed.challenge.status is ModelChallengeStatus.FAILED
    assert replay.challenge.status is ModelChallengeStatus.FAILED
    assert first.calls == second.calls == 2
    assert len(await repository.list_model_builder_calls(package.challenge_id)) == 4


@pytest.mark.asyncio
async def test_no_supported_winner_and_executive_rejection_retain_the_active_model() -> None:
    package = challenge_package()
    no_winner_repository = InMemoryRunRepository()
    await no_winner_repository.append_world_model(
        package.active_model, expected_latest_version=None
    )
    no_winner = ModelChallengeOrchestrator(
        repository=no_winner_repository,
        builders=(
            acquisition_builder("first_builder", ParameterDirection.INCREASE),
            acquisition_builder("second_builder", ParameterDirection.INCREASE),
        ),
    )
    rejected = await no_winner.run(package)
    assert rejected.decision.resolution is ModelChallengeResolution.NO_SUPPORTED_WINNER
    assert rejected.activated_model is None
    assert (
        await no_winner_repository.get_latest_world_model(package.run_id)
    ).id == package.active_model.id

    executive_repository = InMemoryRunRepository()
    await executive_repository.append_world_model(
        package.active_model, expected_latest_version=None
    )
    executive_rejects = ModelChallengeOrchestrator(
        repository=executive_repository,
        builders=(
            acquisition_builder("good_builder", ParameterDirection.DECREASE),
            acquisition_builder("bad_builder", ParameterDirection.INCREASE),
        ),
        authority=DeterministicExecutiveAuthority(approve_supported=False),
    )
    executive_result = await executive_rejects.run(package)
    assert executive_result.decision.resolution is ModelChallengeResolution.EXECUTIVE_REJECTED
    assert executive_result.decision.selected_proposal_ids == ()
    assert executive_result.activated_model is None


@pytest.mark.asyncio
async def test_disjoint_supported_hypotheses_can_merge_but_overlaps_cannot() -> None:
    package = challenge_package()
    residuals = tuple(
        item.model_copy(
            update={
                "parameter_sensitivities": (
                    *item.parameter_sensitivities,
                    ChallengeParameterSensitivity(
                        parameter_name=WorldModelParameterName.PRICE_ELASTICITY,
                        cash_sensitivity_per_unit=1_000_000,
                        evidence_reference=f"finite-difference:{item.outcome_id}:price",
                    ),
                )
            }
        )
        for item in package.residuals
    )
    package = package.model_copy(update={"residuals": residuals})
    acquisition = acquisition_builder(
        "acquisition_builder", ParameterDirection.DECREASE
    )
    pricing = FakeBuilder(
        name="pricing_builder",
        family=HypothesisFamily.PRICING_RESPONSE,
        parameter=WorldModelParameterName.PRICE_ELASTICITY,
        direction=ParameterDirection.DECREASE,
    )
    assert proposals_are_compatible(
        await acquisition.propose(package), await pricing.propose(package)
    )
    overlap = acquisition_builder("overlap_builder", ParameterDirection.DECREASE)
    assert not proposals_are_compatible(
        await acquisition.propose(package), await overlap.propose(package)
    )

    repository = InMemoryRunRepository()
    await repository.append_world_model(
        package.active_model, expected_latest_version=None
    )
    orchestrator = ModelChallengeOrchestrator(
        repository=repository,
        builders=(acquisition, pricing),
    )
    outcome = await orchestrator.run(package)

    assert outcome.decision.resolution in {
        ModelChallengeResolution.ACCEPTED,
        ModelChallengeResolution.MERGED,
    }
    if outcome.decision.resolution is ModelChallengeResolution.MERGED:
        assert len(outcome.decision.selected_proposal_ids) == 2


def test_normal_health_does_not_start_a_fleet() -> None:
    assert ModelChallengeOrchestrator.should_start(None) is False
