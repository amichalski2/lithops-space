"""Append-only persistence contract for strategic-authority records.

Every append is idempotent for identical content and raises ConflictError for a
divergent record under the same identity, so replay reconstructs the exact same
history instead of forking a second source of truth.
"""

from typing import Protocol
from uuid import UUID

from lithops.domain.insights import InsightRecord
from lithops.domain.strategy import (
    CandidateEvaluationSet,
    CommitmentReview,
    ExecutiveChoice,
    ExperimentOutcome,
    StrategicPortfolioRevision,
)


class StrategyRepository(Protocol):
    async def append_portfolio_revision(
        self,
        revision: StrategicPortfolioRevision,
    ) -> StrategicPortfolioRevision: ...

    async def list_portfolio_revisions(
        self,
        run_id: UUID,
    ) -> list[StrategicPortfolioRevision]: ...

    async def get_latest_portfolio_revision(
        self,
        run_id: UUID,
    ) -> StrategicPortfolioRevision | None: ...

    async def append_experiment_outcome(
        self,
        outcome: ExperimentOutcome,
    ) -> ExperimentOutcome: ...

    async def list_experiment_outcomes(self, run_id: UUID) -> list[ExperimentOutcome]: ...

    async def list_commitment_experiment_outcomes(
        self,
        run_id: UUID,
        commitment_id: str,
    ) -> list[ExperimentOutcome]: ...

    async def append_commitment_review(
        self,
        review: CommitmentReview,
    ) -> CommitmentReview: ...

    async def list_commitment_reviews(
        self,
        run_id: UUID,
        commitment_id: str | None = None,
    ) -> list[CommitmentReview]: ...

    async def append_candidate_evaluation_set(
        self,
        evaluation_set: CandidateEvaluationSet,
    ) -> CandidateEvaluationSet: ...

    async def get_candidate_evaluation_set(
        self,
        run_id: UUID,
        week: int,
    ) -> CandidateEvaluationSet | None: ...

    async def append_executive_choice(
        self,
        choice: ExecutiveChoice,
    ) -> ExecutiveChoice: ...

    async def get_executive_choice(
        self,
        run_id: UUID,
        week: int,
    ) -> ExecutiveChoice | None: ...

    async def append_insight_record(self, insight: InsightRecord) -> InsightRecord: ...

    async def list_insight_records(self, run_id: UUID) -> list[InsightRecord]: ...
