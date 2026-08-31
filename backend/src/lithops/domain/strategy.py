"""Durable strategic-authority contracts: hypotheses, portfolios, typed experiment
outcomes, candidate evaluation sets, and Executive choices.

These records make Gemini the strategic decision authority while deterministic
Python remains the evaluator, executor, and safety veto. Every record here is
replay-reconstructible: identities are deterministic, portfolio revisions form a
hash chain, and repositories append idempotently. The tables built on these
models are projections of the decision history, never a second source of truth.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator

SEGMENT_PATTERN = r"^(?:S[1-3]|E[1-3]|D_[SE]\d{2})$"
CHANNEL_PATTERN = (
    r"^(?:social_media|search_ads|linkedin|content_marketing|referral_program)$"
)
HYPOTHESIS_ID_PATTERN = r"^[a-z][a-z0-9_]*$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"


def _canonical_hash(payload: Any) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class BusinessLever(StrEnum):
    """Semantic causal controls available to strategy.

    The enum describes what can be manipulated; it never prescribes which lever
    is correct next.
    """

    PRICE = "price"
    TIER = "tier"
    PROMOTION = "promotion"
    DEVELOPMENT = "development"
    TARGETED_DEVELOPMENT = "targeted_development"
    ACQUISITION = "acquisition"
    CAPACITY_COST = "capacity_cost"
    CONTINUATION = "continuation"


class HypothesisStatus(StrEnum):
    PROPOSED = "proposed"
    RUNNING = "running"
    SUPPORTED = "supported"
    FALSIFIED = "falsified"
    INCONCLUSIVE = "inconclusive"
    SUPERSEDED = "superseded"
    ABANDONED = "abandoned"


ALLOWED_HYPOTHESIS_TRANSITIONS: dict[HypothesisStatus, frozenset[HypothesisStatus]] = {
    HypothesisStatus.PROPOSED: frozenset(
        {
            HypothesisStatus.RUNNING,
            HypothesisStatus.SUPPORTED,
            HypothesisStatus.FALSIFIED,
            HypothesisStatus.INCONCLUSIVE,
            HypothesisStatus.ABANDONED,
            HypothesisStatus.SUPERSEDED,
        }
    ),
    HypothesisStatus.RUNNING: frozenset(
        {
            HypothesisStatus.SUPPORTED,
            HypothesisStatus.FALSIFIED,
            HypothesisStatus.INCONCLUSIVE,
            HypothesisStatus.ABANDONED,
            HypothesisStatus.SUPERSEDED,
        }
    ),
    HypothesisStatus.SUPPORTED: frozenset(
        {HypothesisStatus.RUNNING, HypothesisStatus.SUPERSEDED}
    ),
    HypothesisStatus.FALSIFIED: frozenset({HypothesisStatus.SUPERSEDED}),
    HypothesisStatus.INCONCLUSIVE: frozenset(
        {
            HypothesisStatus.RUNNING,
            HypothesisStatus.ABANDONED,
            HypothesisStatus.SUPERSEDED,
        }
    ),
    HypothesisStatus.SUPERSEDED: frozenset(),
    HypothesisStatus.ABANDONED: frozenset(),
}


class ExperimentOutcomeStatus(StrEnum):
    """Typed classification of what one commitment actually observed.

    NO_EXPOSURE means the intervention produced no exposure or observation at
    all. It is not, by itself, a channel failure: interpreting the cause
    (channel, segment, budget, timing, or lead mechanics) belongs to the
    Executive, never to deterministic code.
    """

    IMMATURE = "immature"
    VALID_EXPOSURE = "valid_exposure"
    NO_EXPOSURE = "no_exposure"
    EXPOSED_ZERO_CONVERSION = "exposed_zero_conversion"
    POSITIVE_CONVERSION = "positive_conversion"
    CENSORED = "censored"
    INVALID_EXECUTION = "invalid_execution"
    STOPPED_FOR_SAFETY = "stopped_for_safety"


class CommitmentReviewVerdict(StrEnum):
    """Weekly execution review of an active commitment.

    The commitment record is immutable; its continued execution is not. Every
    active week produces exactly one of these verdicts, and the verdict follows
    from what the Executive actually chose to run that week — it is recorded,
    not computed on its behalf.
    """

    CONTINUE = "continue"
    STOP_FOR_SAFETY = "stop_for_safety"
    FALSIFIED = "falsified"
    MATURE_AND_PROBE = "mature_and_probe"
    REVERT = "revert"
    # A treatment that earned its place becomes the new operating baseline
    # instead of being rolled back. Without this a won experiment is discarded.
    ADOPTED = "adopted"
    # The Executive moved to something else before the window closed.
    ABANDONED = "abandoned"


class EvidenceEnvelope(BaseModel):
    """The exact offer and reach conditions under which evidence was gathered.

    Evidence is never pooled across incompatible envelopes merely because one
    dimension (for example the quality band) matches.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    segment: str | None = Field(default=None, pattern=SEGMENT_PATTERN)
    channel: str | None = Field(default=None, pattern=CHANNEL_PATTERN)
    quality_band: int | None = Field(default=None, ge=0, le=9)
    catalog_prices: dict[str, float] = Field(default_factory=dict)
    model_tiers: dict[str, str] = Field(default_factory=dict)
    promotion: float = Field(default=0.0, ge=0.0, le=1.0)
    segment_plan_quality_proxies: dict[str, float] = Field(default_factory=dict)
    quality_decision_grade: bool = False
    quality_provenance: str = "unavailable"
    targeted_development_daily: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_prices(self) -> EvidenceEnvelope:
        if any(price < 0.0 for price in self.catalog_prices.values()):
            raise ValueError("catalog prices cannot be negative")
        if any(value < 0.0 for value in self.segment_plan_quality_proxies.values()):
            raise ValueError("quality proxies cannot be negative")
        if any(value < 0.0 for value in self.targeted_development_daily.values()):
            raise ValueError("targeted development cannot be negative")
        return self

    @property
    def canonical_key(self) -> str:
        payload = {
            "segment": self.segment,
            "channel": self.channel,
            "quality_band": self.quality_band,
            "catalog_prices": {
                tier: round(price, 6) for tier, price in self.catalog_prices.items()
            },
            "model_tiers": dict(self.model_tiers),
            "promotion": round(self.promotion, 6),
            "segment_plan_quality_proxies": {
                key: round(value, 6)
                for key, value in self.segment_plan_quality_proxies.items()
            },
            "quality_decision_grade": self.quality_decision_grade,
            "quality_provenance": self.quality_provenance,
            "targeted_development_daily": {
                key: round(value, 6)
                for key, value in self.targeted_development_daily.items()
            },
        }
        return _canonical_hash(payload)

    def matches(self, other: EvidenceEnvelope) -> bool:
        return self.canonical_key == other.canonical_key


class ObjectiveSpec(BaseModel):
    """The benchmark objective the portfolio optimizes toward."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str = Field(default="terminal_cash", min_length=1, max_length=80)
    horizon_day: int = Field(ge=1)
    downside_constraint: str | None = Field(default=None, max_length=500)


class StrategicHypothesis(BaseModel):
    """One causal business claim with its evidence lineage and status history."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hypothesis_id: str = Field(min_length=1, max_length=60, pattern=HYPOTHESIS_ID_PATTERN)
    causal_claim: str = Field(min_length=1, max_length=2_000)
    target_outcome: str = Field(min_length=1, max_length=1_000)
    levers: tuple[BusinessLever, ...] = Field(min_length=1)
    segment: str | None = Field(default=None, pattern=SEGMENT_PATTERN)
    channel: str | None = Field(default=None, pattern=CHANNEL_PATTERN)
    support_envelope: EvidenceEnvelope | None = None
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    evidence_refs: tuple[str, ...] = ()
    falsification_reason: str | None = Field(default=None, max_length=1_000)
    successor_hypothesis_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_lineage(self) -> StrategicHypothesis:
        if len(set(self.levers)) != len(self.levers):
            raise ValueError("hypothesis levers must be unique")
        if (self.status is HypothesisStatus.FALSIFIED) and not self.falsification_reason:
            raise ValueError("falsified hypotheses require a falsification reason")
        if self.status is HypothesisStatus.SUPERSEDED and not self.successor_hypothesis_ids:
            raise ValueError("superseded hypotheses must name at least one successor")
        if self.hypothesis_id in self.successor_hypothesis_ids:
            raise ValueError("a hypothesis cannot succeed itself")
        return self


class StrategicPortfolio(BaseModel):
    """The Executive-owned causal portfolio at one planning week."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    as_of_week: int = Field(ge=0)
    objective: ObjectiveSpec
    binding_constraint: str = Field(min_length=1, max_length=1_000)
    active_hypothesis_ids: tuple[str, ...] = ()
    hypotheses: tuple[StrategicHypothesis, ...] = ()
    remaining_experiment_budget: float = Field(ge=0.0)
    unresolved_questions: tuple[str, ...] = ()
    prior_portfolio_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_portfolio(self) -> StrategicPortfolio:
        ids = [hypothesis.hypothesis_id for hypothesis in self.hypotheses]
        known = set(ids)
        if len(ids) != len(known):
            raise ValueError("portfolio hypothesis IDs must be unique")
        missing_active = set(self.active_hypothesis_ids) - known
        if missing_active:
            raise ValueError(
                f"active hypothesis IDs missing from portfolio: {sorted(missing_active)}"
            )
        for hypothesis in self.hypotheses:
            if hypothesis.hypothesis_id in self.active_hypothesis_ids and (
                hypothesis.status
                not in {HypothesisStatus.PROPOSED, HypothesisStatus.RUNNING}
            ):
                raise ValueError(
                    "active hypotheses must be proposed or running: "
                    f"{hypothesis.hypothesis_id} is {hypothesis.status}"
                )
            missing_successors = set(hypothesis.successor_hypothesis_ids) - known
            if missing_successors:
                raise ValueError(
                    "successor hypotheses must exist in the same portfolio: "
                    f"{sorted(missing_successors)}"
                )
        return self

    @property
    def portfolio_hash(self) -> str:
        return _canonical_hash(self.model_dump(mode="json"))


class StrategicPortfolioRevision(BaseModel):
    """One append-only, hash-chained revision of the strategic portfolio."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    run_id: UUID
    decision_id: UUID | None = None
    week: int = Field(ge=0)
    revision: int = Field(ge=1)
    portfolio: StrategicPortfolio
    created_at: datetime

    @model_validator(mode="after")
    def validate_week_alignment(self) -> StrategicPortfolioRevision:
        if self.portfolio.as_of_week != self.week:
            raise ValueError("portfolio as_of_week must match its revision week")
        if self.revision == 1 and self.portfolio.prior_portfolio_hash is not None:
            raise ValueError("the first portfolio revision cannot have a prior hash")
        if self.revision > 1 and self.portfolio.prior_portfolio_hash is None:
            raise ValueError("later portfolio revisions require the prior hash")
        return self

    @property
    def portfolio_hash(self) -> str:
        return self.portfolio.portfolio_hash


def portfolio_revision_id(run_id: UUID, revision: int) -> UUID:
    return uuid5(NAMESPACE_URL, f"lithops:{run_id}:portfolio-revision:{revision}")


def verify_portfolio_revision_chain(
    revisions: Sequence[StrategicPortfolioRevision],
) -> None:
    """Verify that persisted revisions replay into one consistent history.

    Raises ValueError when the hash chain, revision ordering, or per-hypothesis
    append-only history is violated. An empty sequence is valid.
    """

    previous: StrategicPortfolioRevision | None = None
    for current in revisions:
        if previous is None:
            if current.revision != 1:
                raise ValueError("portfolio history must start at revision 1")
            previous = current
            continue
        if current.revision != previous.revision + 1:
            raise ValueError(
                "portfolio revisions must be contiguous: "
                f"{previous.revision} -> {current.revision}"
            )
        if current.week < previous.week:
            raise ValueError("portfolio revision weeks cannot move backward")
        if current.portfolio.prior_portfolio_hash != previous.portfolio_hash:
            raise ValueError(
                f"portfolio hash chain broken at revision {current.revision}"
            )
        _verify_hypothesis_history(previous.portfolio, current.portfolio)
        previous = current


def _verify_hypothesis_history(
    earlier: StrategicPortfolio,
    later: StrategicPortfolio,
) -> None:
    later_by_id = {
        hypothesis.hypothesis_id: hypothesis for hypothesis in later.hypotheses
    }
    for before in earlier.hypotheses:
        after = later_by_id.get(before.hypothesis_id)
        if after is None:
            raise ValueError(
                f"hypothesis {before.hypothesis_id} was removed from the portfolio"
            )
        if (
            after.causal_claim != before.causal_claim
            or after.target_outcome != before.target_outcome
            or after.levers != before.levers
        ):
            raise ValueError(
                f"hypothesis {before.hypothesis_id} rewrote its causal identity"
            )
        if after.status != before.status and (
            after.status not in ALLOWED_HYPOTHESIS_TRANSITIONS[before.status]
        ):
            raise ValueError(
                f"hypothesis {before.hypothesis_id} made an illegal status "
                f"transition: {before.status} -> {after.status}"
            )
        if after.evidence_refs[: len(before.evidence_refs)] != before.evidence_refs:
            raise ValueError(
                f"hypothesis {before.hypothesis_id} rewrote its evidence history"
            )
        if before.falsification_reason is not None and (
            after.falsification_reason != before.falsification_reason
        ):
            raise ValueError(
                f"hypothesis {before.hypothesis_id} rewrote its falsification reason"
            )
        if (
            after.successor_hypothesis_ids[: len(before.successor_hypothesis_ids)]
            != before.successor_hypothesis_ids
        ):
            raise ValueError(
                f"hypothesis {before.hypothesis_id} rewrote its successor history"
            )


class ExperimentOutcome(BaseModel):
    """A durable, typed result of one commitment under one evidence envelope.

    Zero leads is NO_EXPOSURE: an absence of observation whose cause the
    Executive interprets. Leads with zero matured conversions is
    EXPOSED_ZERO_CONVERSION. Immature or censored cohorts cannot falsify
    conversion, and invalid execution cannot update a business hypothesis.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    run_id: UUID
    commitment_id: str = Field(min_length=3, max_length=80)
    hypothesis_id: str | None = Field(
        default=None, max_length=60, pattern=HYPOTHESIS_ID_PATTERN
    )
    outcome_status: ExperimentOutcomeStatus
    envelope: EvidenceEnvelope
    exposure_spend: float = Field(ge=0.0)
    leads: int = Field(ge=0)
    matured_leads: int = Field(ge=0)
    conversions: int = Field(ge=0)
    started_week: int = Field(ge=0)
    measured_week: int | None = Field(default=None, ge=0)
    evidence_refs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_status_consistency(self) -> ExperimentOutcome:
        if self.matured_leads > self.leads:
            raise ValueError("matured leads cannot exceed observed leads")
        if self.conversions > self.matured_leads:
            raise ValueError("conversions cannot exceed matured leads")
        if self.outcome_status is ExperimentOutcomeStatus.IMMATURE:
            if self.measured_week is not None:
                raise ValueError("immature outcomes cannot have a measured week")
        else:
            if self.measured_week is None:
                raise ValueError(
                    f"{self.outcome_status} outcomes require a measured week"
                )
            if self.measured_week < self.started_week:
                raise ValueError("measured week cannot precede the start week")
        if self.outcome_status is ExperimentOutcomeStatus.NO_EXPOSURE and self.leads != 0:
            raise ValueError("no-exposure outcomes cannot have observed leads")
        if self.outcome_status is ExperimentOutcomeStatus.EXPOSED_ZERO_CONVERSION and (
            self.leads == 0 or self.matured_leads == 0 or self.conversions != 0
        ):
            raise ValueError(
                "exposed-zero-conversion requires matured leads and zero conversions"
            )
        if self.outcome_status is ExperimentOutcomeStatus.POSITIVE_CONVERSION and (
            self.conversions == 0
        ):
            raise ValueError("positive-conversion outcomes require conversions")
        if self.outcome_status is ExperimentOutcomeStatus.VALID_EXPOSURE and (
            self.leads == 0
        ):
            raise ValueError("valid-exposure outcomes require observed leads")
        return self


def experiment_outcome_id(run_id: UUID, commitment_id: str, week: int) -> UUID:
    """Deterministic identity so replay reduces to the exact same record."""

    return uuid5(
        NAMESPACE_URL, f"lithops:{run_id}:experiment-outcome:{commitment_id}:{week}"
    )


class CommitmentReview(BaseModel):
    """The weekly execution verdict for one immutable commitment record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    run_id: UUID
    commitment_id: str = Field(min_length=3, max_length=80)
    week: int = Field(ge=0)
    verdict: CommitmentReviewVerdict
    reason: str = Field(min_length=1, max_length=1_000)
    evidence_refs: tuple[str, ...] = ()
    created_at: datetime


def commitment_review_id(run_id: UUID, commitment_id: str, week: int) -> UUID:
    return uuid5(
        NAMESPACE_URL, f"lithops:{run_id}:commitment-review:{commitment_id}:{week}"
    )


class CandidateEvaluationCard(BaseModel):
    """One comparable, immutable evaluation of a proposed candidate plan.

    Python marks eligibility and veto codes; it does not select strategy.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str = Field(min_length=1, max_length=120)
    plan_hash: str = Field(pattern=SHA256_PATTERN)
    hypothesis_id: str | None = Field(
        default=None, max_length=60, pattern=HYPOTHESIS_ID_PATTERN
    )
    eligible: bool
    veto_codes: tuple[str, ...] = ()
    expected_terminal_cash: float
    downside_terminal_cash: float
    # Upper decile of the same rollouts. A candidate whose payoff is unmeasured
    # must show the width of its unknown on both sides: downside alone turns
    # "never tried" into "never worth trying".
    upside_terminal_cash: float | None = None
    bankruptcy_probability: float = Field(ge=0.0, le=1.0)
    going_concern_failure_probability: float = Field(ge=0.0, le=1.0)
    horizon_expected_cash: dict[int, float] = Field(default_factory=dict)
    # Downside gap versus continuation at the horizon nearest the experiment's
    # own commitment window. A fact for the Executive to weigh, never a veto.
    downside_cost_commitment_window: float | None = Field(default=None, ge=0.0)
    model_disagreement: float | None = Field(default=None, ge=0.0)
    # Which of the world model's parameters this forecast leans on that are
    # still only a generic prior. Without it a six-figure cash figure derived
    # from an uninformative starting value is indistinguishable from one earned
    # by thirty weeks of evidence, and the Executive has no way to ask.
    forecast_rests_on_priors: tuple[str, ...] = ()
    # The subset of those prior-only parameters that THIS candidate's levers
    # actually lean on. Run-level priors name the model's ignorance; this names
    # the candidates that would turn it into a measurement by being executed.
    levers_on_priors: tuple[str, ...] = ()
    # Terminal cash relative to simply carrying on. Candidates often differ by a
    # fraction of a percent on six-figure absolute forecasts, where the eye — and
    # the model — reads them as identical and falls back on whichever risks
    # least. Stated as a difference, a real advantage stops hiding in rounding.
    terminal_cash_versus_continuation: float | None = None
    expected_observation_contrast: str | None = Field(default=None, max_length=1_000)
    support_and_assumption_warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_eligibility(self) -> CandidateEvaluationCard:
        if self.eligible and self.veto_codes:
            raise ValueError("eligible candidates cannot carry veto codes")
        if not self.eligible and not self.veto_codes:
            raise ValueError("ineligible candidates must state at least one veto code")
        if any(horizon <= 0 for horizon in self.horizon_expected_cash):
            raise ValueError("forecast horizons must be positive day counts")
        return self


class CandidateEvaluationSet(BaseModel):
    """The immutable evaluation cards offered to the Executive for one week."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    run_id: UUID
    week: int = Field(ge=0)
    portfolio_hash: str = Field(pattern=SHA256_PATTERN)
    cards: tuple[CandidateEvaluationCard, ...] = Field(min_length=1)
    created_at: datetime

    @model_validator(mode="after")
    def validate_cards(self) -> CandidateEvaluationSet:
        candidate_ids = [card.candidate_id for card in self.cards]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate IDs must be unique within an evaluation set")
        return self

    @property
    def set_hash(self) -> str:
        payload = {
            "run_id": str(self.run_id),
            "week": self.week,
            "portfolio_hash": self.portfolio_hash,
            "cards": [card.model_dump(mode="json") for card in self.cards],
        }
        return _canonical_hash(payload)

    def eligible_candidate_ids(self) -> tuple[str, ...]:
        return tuple(card.candidate_id for card in self.cards if card.eligible)


def candidate_evaluation_set_id(run_id: UUID, week: int) -> UUID:
    return uuid5(NAMESPACE_URL, f"lithops:{run_id}:candidate-evaluation-set:{week}")


class ExecutiveChoice(BaseModel):
    """The Executive's final selection among eligible evaluated candidates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    run_id: UUID
    week: int = Field(ge=0)
    evaluation_set_id: UUID
    evaluation_set_hash: str = Field(pattern=SHA256_PATTERN)
    selected_candidate_id: str = Field(min_length=1, max_length=120)
    decision_thesis: str = Field(min_length=1, max_length=4_000)
    evidence_refs: tuple[str, ...] = ()
    rejected_candidate_ids: tuple[str, ...] = ()
    stop_or_pivot_condition: str = Field(min_length=1, max_length=1_000)
    created_at: datetime

    @model_validator(mode="after")
    def validate_selection(self) -> ExecutiveChoice:
        if self.selected_candidate_id in self.rejected_candidate_ids:
            raise ValueError("the selected candidate cannot also be rejected")
        if len(self.rejected_candidate_ids) != len(set(self.rejected_candidate_ids)):
            raise ValueError("rejected candidate IDs must be unique")
        return self


def executive_choice_id(run_id: UUID, week: int) -> UUID:
    return uuid5(NAMESPACE_URL, f"lithops:{run_id}:executive-choice:{week}")


def validate_choice_against_set(
    choice: ExecutiveChoice,
    evaluation_set: CandidateEvaluationSet,
) -> None:
    """The deterministic gate between selection and execution.

    Raises ValueError unless the choice references this exact evaluation set
    and selects a candidate that is still eligible and unchanged.
    """

    if choice.evaluation_set_id != evaluation_set.id:
        raise ValueError("choice references a different evaluation set")
    if choice.evaluation_set_hash != evaluation_set.set_hash:
        raise ValueError("evaluation set changed between evaluation and choice")
    if choice.selected_candidate_id not in evaluation_set.eligible_candidate_ids():
        raise ValueError(
            f"selected candidate is not eligible: {choice.selected_candidate_id}"
        )
