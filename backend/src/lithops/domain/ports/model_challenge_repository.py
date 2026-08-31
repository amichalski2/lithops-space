"""Persistence boundary for the dynamic model-challenge lifecycle."""

from typing import Protocol
from uuid import UUID

from lithops.domain.model_challenge import (
    HypothesisBacktestResult,
    ModelBuilderCallReceipt,
    ModelBuilderProposal,
    ModelChallengeDecision,
    ModelChallengePackage,
    ModelChallengeRecord,
)


class ModelChallengeRepository(Protocol):
    async def get_model_challenge(self, challenge_id: UUID) -> ModelChallengeRecord | None: ...

    async def save_model_challenge(
        self, challenge: ModelChallengeRecord
    ) -> ModelChallengeRecord: ...

    async def append_model_challenge_package(
        self, package: ModelChallengePackage
    ) -> ModelChallengePackage: ...

    async def get_model_challenge_package(
        self, challenge_id: UUID
    ) -> ModelChallengePackage | None: ...

    async def append_model_builder_proposal(
        self, proposal: ModelBuilderProposal
    ) -> ModelBuilderProposal: ...

    async def list_model_builder_proposals(
        self, challenge_id: UUID
    ) -> list[ModelBuilderProposal]: ...

    async def append_hypothesis_backtest(
        self, result: HypothesisBacktestResult
    ) -> HypothesisBacktestResult: ...

    async def list_hypothesis_backtests(
        self, challenge_id: UUID
    ) -> list[HypothesisBacktestResult]: ...

    async def append_model_builder_call(
        self, receipt: ModelBuilderCallReceipt
    ) -> ModelBuilderCallReceipt: ...

    async def list_model_builder_calls(
        self, challenge_id: UUID
    ) -> list[ModelBuilderCallReceipt]: ...

    async def append_model_challenge_decision(
        self, decision: ModelChallengeDecision
    ) -> ModelChallengeDecision: ...

    async def get_model_challenge_decision(
        self, challenge_id: UUID
    ) -> ModelChallengeDecision | None: ...
