"""Bounded dynamic fleet orchestration and deterministic activation."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID, uuid5

from pydantic import ValidationError

from lithops.domain.model_challenge import (
    HypothesisBacktestResult,
    HypothesisEvidenceReference,
    ModelBuilderCallReceipt,
    ModelBuilderCallStatus,
    ModelBuilderProposal,
    ModelChallengeDecision,
    ModelChallengePackage,
    ModelChallengeRecord,
    ModelChallengeResolution,
    ModelChallengeStatus,
    WorldModelHypothesisDiff,
)
from lithops.domain.models import EventRecord
from lithops.domain.ports import LearningRepository
from lithops.domain.world_model import (
    EvidenceKind,
    EvidenceReference,
    WorldModelParameterChange,
    WorldModelVersion,
)
from lithops.world_model import backtest_hypothesis, compile_hypothesis


class ProposalBuilder(Protocol):
    spec: object
    provider_name: str
    provider: object

    async def propose(self, package: ModelChallengePackage) -> ModelBuilderProposal: ...


class ModelChallengeStore(LearningRepository, Protocol):
    async def append_event(self, event: EventRecord) -> EventRecord: ...


class ExecutiveModelAuthority(Protocol):
    name: str
    version: str

    async def approve(
        self,
        *,
        package: ModelChallengePackage,
        proposal_ids: tuple[UUID, ...],
        backtest: HypothesisBacktestResult,
    ) -> bool: ...


@dataclass(frozen=True, slots=True)
class DeterministicExecutiveAuthority:
    """Safe default: approve only the already policy-eligible strongest candidate."""

    name: str = "executive_policy"
    version: str = "1.0"
    approve_supported: bool = True

    async def approve(
        self,
        *,
        package: ModelChallengePackage,
        proposal_ids: tuple[UUID, ...],
        backtest: HypothesisBacktestResult,
    ) -> bool:
        del package, proposal_ids
        return self.approve_supported and backtest.supported


@dataclass(frozen=True, slots=True)
class ChallengeOutcome:
    challenge: ModelChallengeRecord
    decision: ModelChallengeDecision | None
    activated_model: WorldModelVersion | None


def _canonical_hash(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def proposals_are_compatible(
    first: ModelBuilderProposal,
    second: ModelBuilderProposal,
) -> bool:
    if first.challenge_id != second.challenge_id or first.family is second.family:
        return False
    first_parameters = {item.parameter_name for item in first.diff.parameter_adjustments}
    second_parameters = {item.parameter_name for item in second.diff.parameter_adjustments}
    first_relationships = {
        item.relationship_key for item in first.diff.relationship_activations
    }
    second_relationships = {
        item.relationship_key for item in second.diff.relationship_activations
    }
    return not (first_parameters & second_parameters) and not (
        first_relationships & second_relationships
    )


def _merge_proposals(
    package: ModelChallengePackage,
    first: ModelBuilderProposal,
    second: ModelBuilderProposal,
) -> ModelBuilderProposal:
    evidence: dict[tuple[object, str], HypothesisEvidenceReference] = {}
    for item in (*first.evidence, *second.evidence):
        evidence[(item.kind, item.reference)] = item
    return ModelBuilderProposal(
        id=uuid5(
            package.challenge_id,
            f"compatible-merge:{min(str(first.id), str(second.id))}:"
            f"{max(str(first.id), str(second.id))}",
        ),
        challenge_id=package.challenge_id,
        builder_name="executive_merge",
        builder_version="1.0",
        prompt_version="deterministic-compatible-merge-v1",
        provider="deterministic",
        model_name="compatibility-policy-v1",
        family=first.family,
        summary="Compatible supported hypotheses merged by deterministic policy.",
        rationale="The proposals affect disjoint parameters and relationships.",
        diff=WorldModelHypothesisDiff(
            parameter_adjustments=(
                *first.diff.parameter_adjustments,
                *second.diff.parameter_adjustments,
            ),
            relationship_activations=(
                *first.diff.relationship_activations,
                *second.diff.relationship_activations,
            ),
        ),
        evidence=tuple(evidence.values()),
        created_at=package.created_at,
    )


def _activate_model(
    package: ModelChallengePackage,
    activation_proposal: ModelBuilderProposal,
    selected_proposal_ids: tuple[UUID, ...],
    backtest: HypothesisBacktestResult,
) -> WorldModelVersion:
    compiled = compile_hypothesis(package, activation_proposal)
    baseline = {item.name: item for item in package.active_model.parameters}
    candidate = {item.name: item for item in compiled.candidate_model.parameters}
    builder_evidence = tuple(
        EvidenceReference(
            kind=EvidenceKind.MODEL_BUILDER,
            reference=f"model-builder-proposal:{proposal_id}",
            observed_day=package.health_signal.evaluated_day,
            note=f"Supported by deterministic backtest {backtest.id}.",
        )
        for proposal_id in selected_proposal_ids
    )
    residual_evidence = tuple(
        EvidenceReference(
            kind=EvidenceKind.PREDICTION_RESIDUAL,
            reference=f"prediction-outcome:{residual.outcome_id}",
            observed_day=residual.observed_day,
            note=f"Replay fold in deterministic backtest {backtest.id}.",
        )
        for residual in package.residuals
    )
    evidence = (*builder_evidence, *residual_evidence)
    changes = tuple(
        WorldModelParameterChange(
            parameter_name=name,
            previous_estimate=baseline[name].estimate,
            new_estimate=candidate[name].estimate,
            previous_confidence=baseline[name].confidence,
            new_confidence=baseline[name].confidence,
            update_method="model_challenge_backtest_v1",
            evidence=evidence,
        )
        for name in compiled.changed_parameters
    )
    if not changes:
        raise ValueError("P0 activation requires at least one backtested parameter change")
    changed_names = set(compiled.changed_parameters)
    parameters = tuple(
        parameter.model_copy(update={"evidence": (*parameter.evidence, *residual_evidence)})
        if parameter.name in changed_names
        else parameter
        for parameter in compiled.candidate_model.parameters
    )
    return WorldModelVersion(
        id=uuid5(
            package.challenge_id,
            "activated-model:" + ":".join(sorted(str(item) for item in selected_proposal_ids)),
        ),
        run_id=package.run_id,
        version=package.active_model.version + 1,
        source_observation_day=package.health_signal.evaluated_day,
        based_on_version_id=package.active_model.id,
        parameters=parameters,
        relationships=compiled.candidate_model.relationships,
        changes=changes,
        update_method="model_challenge_backtest_v1",
        created_at=package.created_at,
    )


class ModelChallengeOrchestrator:
    def __init__(
        self,
        *,
        repository: ModelChallengeStore,
        builders: tuple[ProposalBuilder, ...],
        authority: ExecutiveModelAuthority | None = None,
        builder_timeout_seconds: float = 30.0,
        complexity_penalty_per_unit: float = 0.01,
        minimum_required_improvement: float = 0.02,
    ) -> None:
        if not 2 <= len(builders) <= 3:
            raise ValueError("a model challenge requires two or three builders")
        if builder_timeout_seconds <= 0:
            raise ValueError("builder timeout must be positive")
        if complexity_penalty_per_unit < 0 or minimum_required_improvement < 0:
            raise ValueError("backtest policy thresholds must be non-negative")
        names = tuple(self._builder_metadata(item)[0] for item in builders)
        if len(names) != len(set(names)):
            raise ValueError("model challenge builder names must be unique")
        self.repository = repository
        self.builders = builders
        self.authority = authority or DeterministicExecutiveAuthority()
        self.builder_timeout_seconds = builder_timeout_seconds
        self.complexity_penalty_per_unit = complexity_penalty_per_unit
        self.minimum_required_improvement = minimum_required_improvement

    @staticmethod
    def should_start(package: ModelChallengePackage | None) -> bool:
        return bool(package and package.health_signal.rebuild_recommended)

    @staticmethod
    def _builder_metadata(builder: ProposalBuilder) -> tuple[str, str, str, str, str]:
        spec = builder.spec
        return (
            spec.name,
            spec.version,
            spec.prompt_version,
            builder.provider_name,
            builder.provider.model_id,
        )

    async def run(self, package: ModelChallengePackage) -> ChallengeOutcome:
        if not self.should_start(package):
            raise ValueError("model challenge requires rebuild_recommended")
        existing = await self.repository.get_model_challenge(package.challenge_id)
        if existing and existing.status is ModelChallengeStatus.COMPLETED:
            decision = await self.repository.get_model_challenge_decision(package.challenge_id)
            activated = (
                await self.repository.get_world_model(decision.activated_model_version_id)
                if decision and decision.activated_model_version_id
                else None
            )
            return ChallengeOutcome(existing, decision, activated)

        names = tuple(self._builder_metadata(item)[0] for item in self.builders)
        challenge = existing or ModelChallengeRecord(
            id=package.challenge_id,
            run_id=package.run_id,
            health_signal_id=package.health_signal.id,
            base_model_version_id=package.active_model.id,
            requested_builders=names,
            created_at=package.created_at,
            updated_at=package.created_at,
        )
        challenge = await self.repository.save_model_challenge(challenge)
        await self.repository.append_model_challenge_package(package)
        await self._event(package, "model_challenge.started", "started")
        challenge = await self.repository.save_model_challenge(
            challenge.model_copy(
                update={
                    "status": ModelChallengeStatus.BUILDING,
                    "updated_at": package.created_at,
                }
            )
        )

        existing_proposals = {
            item.builder_name: item
            for item in await self.repository.list_model_builder_proposals(
                package.challenge_id
            )
        }
        tasks = [
            self._invoke_builder(package, builder)
            for builder in self.builders
            if self._builder_metadata(builder)[0] not in existing_proposals
        ]
        if tasks:
            fresh = await asyncio.gather(*tasks)
            existing_proposals.update(
                {item.builder_name: item for item in fresh if item is not None}
            )
        proposals = tuple(existing_proposals.values())
        if not proposals:
            return await self._fail(package, challenge, "all_builders_failed")

        challenge = await self.repository.save_model_challenge(
            challenge.model_copy(
                update={
                    "status": ModelChallengeStatus.BACKTESTING,
                    "updated_at": package.created_at,
                }
            )
        )
        backtests_by_proposal = {
            item.proposal_id: item
            for item in await self.repository.list_hypothesis_backtests(package.challenge_id)
        }
        eligible: list[tuple[ModelBuilderProposal, HypothesisBacktestResult]] = []
        for proposal in proposals:
            result = backtests_by_proposal.get(proposal.id)
            if result is None:
                try:
                    _, result = backtest_hypothesis(
                        package,
                        proposal,
                        complexity_penalty_per_unit=self.complexity_penalty_per_unit,
                        minimum_required_improvement=self.minimum_required_improvement,
                    )
                except ValueError:
                    continue
                result = await self.repository.append_hypothesis_backtest(result)
                await self._event(
                    package,
                    "hypothesis.backtested",
                    str(result.id),
                    {
                        "proposal_id": str(proposal.id),
                        "supported": result.supported,
                        "penalized_improvement": result.penalized_improvement,
                    },
                )
            eligible.append((proposal, result))
        if not eligible:
            return await self._fail(package, challenge, "all_backtests_failed")

        ranked = sorted(
            eligible,
            key=lambda item: (
                not item[1].supported,
                -item[1].penalized_improvement,
                item[1].candidate_score,
                item[0].builder_name,
            ),
        )
        supported = [item for item in ranked if item[1].supported]
        activation_proposal: ModelBuilderProposal | None = None
        selected_ids: tuple[UUID, ...] = ()
        winning_backtest: HypothesisBacktestResult | None = None
        resolution = ModelChallengeResolution.NO_SUPPORTED_WINNER
        if supported:
            activation_proposal, winning_backtest = supported[0]
            selected_ids = (activation_proposal.id,)
            resolution = ModelChallengeResolution.ACCEPTED
            if len(supported) > 1 and proposals_are_compatible(
                supported[0][0], supported[1][0]
            ):
                merged = _merge_proposals(package, supported[0][0], supported[1][0])
                try:
                    _, merged_result = backtest_hypothesis(
                        package,
                        merged,
                        complexity_penalty_per_unit=self.complexity_penalty_per_unit,
                        minimum_required_improvement=self.minimum_required_improvement,
                    )
                except ValueError:
                    pass
                else:
                    await self.repository.append_model_builder_proposal(merged)
                    await self.repository.append_hypothesis_backtest(merged_result)
                    if (
                        merged_result.supported
                        and merged_result.penalized_improvement
                        >= winning_backtest.penalized_improvement
                    ):
                        activation_proposal = merged
                        winning_backtest = merged_result
                        selected_ids = (supported[0][0].id, supported[1][0].id)
                        resolution = ModelChallengeResolution.MERGED

        challenge = await self.repository.save_model_challenge(
            challenge.model_copy(
                update={
                    "status": ModelChallengeStatus.AWAITING_EXECUTIVE,
                    "updated_at": package.created_at,
                }
            )
        )
        activated_model: WorldModelVersion | None = None
        if activation_proposal is not None and winning_backtest is not None:
            approved = await self.authority.approve(
                package=package,
                proposal_ids=selected_ids,
                backtest=winning_backtest,
            )
            if approved:
                activated_model = _activate_model(
                    package,
                    activation_proposal,
                    selected_ids,
                    winning_backtest,
                )
                activated_model = await self.repository.append_world_model(
                    activated_model,
                    expected_latest_version=package.active_model.version,
                )
            else:
                resolution = ModelChallengeResolution.EXECUTIVE_REJECTED
                selected_ids = ()

        all_backtests = await self.repository.list_hypothesis_backtests(package.challenge_id)
        decision = ModelChallengeDecision(
            id=uuid5(package.challenge_id, "model-challenge-decision-v1"),
            challenge_id=package.challenge_id,
            resolution=resolution,
            selected_proposal_ids=selected_ids,
            supporting_backtest_ids=tuple(item.id for item in all_backtests),
            activated_model_version_id=(activated_model.id if activated_model else None),
            authority_name=self.authority.name,
            authority_version=self.authority.version,
            reason_code=(
                "supported_diff_activated"
                if activated_model
                else "executive_rejected"
                if resolution is ModelChallengeResolution.EXECUTIVE_REJECTED
                else "no_supported_winner"
            ),
            decided_at=package.created_at,
        )
        decision = await self.repository.append_model_challenge_decision(decision)
        completed = await self.repository.save_model_challenge(
            challenge.model_copy(
                update={
                    "status": ModelChallengeStatus.COMPLETED,
                    "decision_id": decision.id,
                    "updated_at": package.created_at,
                    "completed_at": package.created_at,
                }
            )
        )
        await self._event(
            package,
            "model_challenge.completed",
            "completed",
            {
                "decision_id": str(decision.id),
                "resolution": decision.resolution.value,
                "activated_model_version_id": (
                    str(activated_model.id) if activated_model else None
                ),
            },
        )
        return ChallengeOutcome(completed, decision, activated_model)

    async def _invoke_builder(
        self,
        package: ModelChallengePackage,
        builder: ProposalBuilder,
    ) -> ModelBuilderProposal | None:
        name, version, prompt_version, provider, model_name = self._builder_metadata(builder)
        input_hash = _canonical_hash(package)
        existing_calls = {
            receipt.attempt: receipt
            for receipt in await self.repository.list_model_builder_calls(
                package.challenge_id
            )
            if receipt.builder_name == name
        }
        for attempt in (1, 2):
            if attempt in existing_calls:
                continue
            try:
                async with asyncio.timeout(self.builder_timeout_seconds):
                    proposal = await builder.propose(package)
            except TimeoutError:
                status = ModelBuilderCallStatus.TIMED_OUT
                error_code = "builder_timeout"
            except (ValueError, ValidationError):
                status = ModelBuilderCallStatus.INVALID_OUTPUT
                error_code = "invalid_structured_output"
            except Exception:
                status = ModelBuilderCallStatus.FAILED
                error_code = "provider_failure"
            else:
                proposal = await self.repository.append_model_builder_proposal(proposal)
                receipt = ModelBuilderCallReceipt(
                    id=uuid5(package.challenge_id, f"builder-call:{name}:{attempt}"),
                    challenge_id=package.challenge_id,
                    builder_name=name,
                    builder_version=version,
                    prompt_version=prompt_version,
                    provider=provider,
                    model_name=model_name,
                    attempt=attempt,
                    status=ModelBuilderCallStatus.COMPLETED,
                    input_hash=input_hash,
                    output_hash=_canonical_hash(proposal),
                    proposal_id=proposal.id,
                    completed_at=package.created_at,
                )
                await self.repository.append_model_builder_call(receipt)
                await self._event(
                    package,
                    "model_builder.completed",
                    f"{name}:{attempt}",
                    {
                        "builder_name": name,
                        "builder_version": version,
                        "prompt_version": prompt_version,
                        "provider": provider,
                        "model_name": model_name,
                        "proposal_id": str(proposal.id),
                        "input_hash": input_hash,
                        "output_hash": receipt.output_hash,
                    },
                )
                return proposal
            receipt = ModelBuilderCallReceipt(
                id=uuid5(package.challenge_id, f"builder-call:{name}:{attempt}"),
                challenge_id=package.challenge_id,
                builder_name=name,
                builder_version=version,
                prompt_version=prompt_version,
                provider=provider,
                model_name=model_name,
                attempt=attempt,
                status=status,
                input_hash=input_hash,
                error_code=error_code,
                completed_at=package.created_at,
            )
            await self.repository.append_model_builder_call(receipt)
        return None

    async def _fail(
        self,
        package: ModelChallengePackage,
        challenge: ModelChallengeRecord,
        reason: str,
    ) -> ChallengeOutcome:
        failed = await self.repository.save_model_challenge(
            challenge.model_copy(
                update={
                    "status": ModelChallengeStatus.FAILED,
                    "failure_reason": reason,
                    "updated_at": package.created_at,
                    "completed_at": package.created_at,
                }
            )
        )
        await self._event(
            package,
            "model_challenge.failed",
            "failed",
            {"reason_code": reason},
        )
        return ChallengeOutcome(failed, None, None)

    async def _event(
        self,
        package: ModelChallengePackage,
        event_type: str,
        discriminator: str,
        payload: dict[str, object] | None = None,
    ) -> None:
        await self.repository.append_event(
            EventRecord(
                id=uuid5(package.challenge_id, f"event:{event_type}:{discriminator}"),
                run_id=package.run_id,
                type=event_type,
                payload={
                    "challenge_id": str(package.challenge_id),
                    "correlation_id": str(package.challenge_id),
                    **(payload or {}),
                },
                created_at=package.created_at,
            )
        )
