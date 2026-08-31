from datetime import UTC, datetime
from uuid import UUID

import pytest
from lithops.domain.errors import ConflictError
from lithops.domain.strategy import (
    BusinessLever,
    CandidateEvaluationCard,
    CandidateEvaluationSet,
    CommitmentReview,
    CommitmentReviewVerdict,
    EvidenceEnvelope,
    ExecutiveChoice,
    ExperimentOutcome,
    ExperimentOutcomeStatus,
    HypothesisStatus,
    ObjectiveSpec,
    StrategicHypothesis,
    StrategicPortfolio,
    StrategicPortfolioRevision,
    candidate_evaluation_set_id,
    commitment_review_id,
    executive_choice_id,
    experiment_outcome_id,
    portfolio_revision_id,
    validate_choice_against_set,
    verify_portfolio_revision_chain,
)
from lithops.infrastructure.persistence.repositories import InMemoryRunRepository
from pydantic import ValidationError

RUN_ID = UUID("77777777-7777-7777-7777-777777777777")
CREATED_AT = datetime(2026, 8, 28, tzinfo=UTC)
PLAN_HASH = "a" * 64


def envelope(**overrides) -> EvidenceEnvelope:
    payload = {
        "segment": "S1",
        "channel": "search_ads",
        "quality_band": 4,
        "catalog_prices": {"A": 99.0, "B": 249.0},
        "model_tiers": {"A": "basic", "B": "pro"},
        "promotion": 0.0,
    }
    payload.update(overrides)
    return EvidenceEnvelope(**payload)


def hypothesis(hypothesis_id: str = "h_search_s1", **overrides) -> StrategicHypothesis:
    payload = {
        "hypothesis_id": hypothesis_id,
        "causal_claim": "S1 companies convert search leads at the entry price",
        "target_outcome": "at least one conversion from matured search leads",
        "levers": (BusinessLever.ACQUISITION,),
        "segment": "S1",
        "channel": "search_ads",
        "support_envelope": envelope(),
        "status": HypothesisStatus.PROPOSED,
    }
    payload.update(overrides)
    return StrategicHypothesis(**payload)


def portfolio(
    *,
    as_of_week: int = 0,
    prior_portfolio_hash: str | None = None,
    hypotheses: tuple[StrategicHypothesis, ...] | None = None,
    active: tuple[str, ...] | None = None,
) -> StrategicPortfolio:
    resolved = hypotheses if hypotheses is not None else (hypothesis(),)
    return StrategicPortfolio(
        as_of_week=as_of_week,
        objective=ObjectiveSpec(horizon_day=500),
        binding_constraint="no observed conversions in any envelope",
        active_hypothesis_ids=(
            active
            if active is not None
            else tuple(
                h.hypothesis_id
                for h in resolved
                if h.status in {HypothesisStatus.PROPOSED, HypothesisStatus.RUNNING}
            )
        ),
        hypotheses=resolved,
        remaining_experiment_budget=5_000.0,
        prior_portfolio_hash=prior_portfolio_hash,
    )


def revision(
    *,
    number: int,
    week: int,
    body: StrategicPortfolio,
) -> StrategicPortfolioRevision:
    return StrategicPortfolioRevision(
        id=portfolio_revision_id(RUN_ID, number),
        run_id=RUN_ID,
        week=week,
        revision=number,
        portfolio=body,
        created_at=CREATED_AT,
    )


def card(candidate_id: str = "cand_continuation", **overrides) -> CandidateEvaluationCard:
    payload = {
        "candidate_id": candidate_id,
        "plan_hash": PLAN_HASH,
        "eligible": True,
        "expected_terminal_cash": 120_000.0,
        "downside_terminal_cash": 60_000.0,
        "bankruptcy_probability": 0.02,
        "going_concern_failure_probability": 0.05,
    }
    payload.update(overrides)
    return CandidateEvaluationCard(**payload)


def evaluation_set(
    cards: tuple[CandidateEvaluationCard, ...],
    week: int = 3,
) -> CandidateEvaluationSet:
    return CandidateEvaluationSet(
        id=candidate_evaluation_set_id(RUN_ID, week),
        run_id=RUN_ID,
        week=week,
        portfolio_hash=portfolio().portfolio_hash,
        cards=cards,
        created_at=CREATED_AT,
    )


def choice_for(
    evaluated: CandidateEvaluationSet,
    selected: str,
    rejected: tuple[str, ...] = (),
) -> ExecutiveChoice:
    return ExecutiveChoice(
        id=executive_choice_id(RUN_ID, evaluated.week),
        run_id=RUN_ID,
        week=evaluated.week,
        evaluation_set_id=evaluated.id,
        evaluation_set_hash=evaluated.set_hash,
        selected_candidate_id=selected,
        decision_thesis="continuation preserves runway while the probe matures",
        rejected_candidate_ids=rejected,
        stop_or_pivot_condition="pivot if the matured cohort shows zero conversions",
        created_at=CREATED_AT,
    )


def outcome(
    status: ExperimentOutcomeStatus,
    *,
    leads: int = 0,
    matured_leads: int = 0,
    conversions: int = 0,
    measured_week: int | None = 4,
) -> ExperimentOutcome:
    return ExperimentOutcome(
        id=experiment_outcome_id(RUN_ID, "probe_search_s1", measured_week or 0),
        run_id=RUN_ID,
        commitment_id="probe_search_s1",
        hypothesis_id="h_search_s1",
        outcome_status=status,
        envelope=envelope(),
        exposure_spend=2_500.0,
        leads=leads,
        matured_leads=matured_leads,
        conversions=conversions,
        started_week=2,
        measured_week=measured_week,
    )


class TestEvidenceEnvelope:
    def test_identical_envelopes_match(self) -> None:
        assert envelope().matches(envelope())

    def test_any_dimension_change_creates_a_distinct_envelope(self) -> None:
        base = envelope()
        assert not base.matches(envelope(channel="linkedin"))
        assert not base.matches(envelope(segment="E1"))
        assert not base.matches(envelope(quality_band=5))
        assert not base.matches(envelope(promotion=0.1))
        assert not base.matches(envelope(catalog_prices={"A": 89.0, "B": 249.0}))

    def test_canonical_key_is_stable_across_instances(self) -> None:
        assert envelope().canonical_key == envelope().canonical_key


class TestExperimentOutcome:
    def test_no_exposure_requires_zero_leads(self) -> None:
        assert (
            outcome(ExperimentOutcomeStatus.NO_EXPOSURE).outcome_status
            is ExperimentOutcomeStatus.NO_EXPOSURE
        )
        with pytest.raises(ValidationError):
            outcome(ExperimentOutcomeStatus.NO_EXPOSURE, leads=3)

    def test_exposed_zero_conversion_requires_matured_leads(self) -> None:
        result = outcome(
            ExperimentOutcomeStatus.EXPOSED_ZERO_CONVERSION,
            leads=8,
            matured_leads=8,
        )
        assert result.conversions == 0
        with pytest.raises(ValidationError):
            outcome(ExperimentOutcomeStatus.EXPOSED_ZERO_CONVERSION, leads=8)
        with pytest.raises(ValidationError):
            outcome(
                ExperimentOutcomeStatus.EXPOSED_ZERO_CONVERSION,
                leads=8,
                matured_leads=8,
                conversions=1,
            )

    def test_positive_conversion_requires_conversions(self) -> None:
        outcome(
            ExperimentOutcomeStatus.POSITIVE_CONVERSION,
            leads=8,
            matured_leads=6,
            conversions=2,
        )
        with pytest.raises(ValidationError):
            outcome(ExperimentOutcomeStatus.POSITIVE_CONVERSION, leads=8, matured_leads=6)

    def test_immature_outcomes_have_no_measured_week(self) -> None:
        outcome(ExperimentOutcomeStatus.IMMATURE, measured_week=None)
        with pytest.raises(ValidationError):
            outcome(ExperimentOutcomeStatus.IMMATURE, measured_week=4)
        with pytest.raises(ValidationError):
            outcome(ExperimentOutcomeStatus.VALID_EXPOSURE, leads=4, measured_week=None)

    def test_lead_accounting_is_monotone(self) -> None:
        with pytest.raises(ValidationError):
            outcome(ExperimentOutcomeStatus.VALID_EXPOSURE, leads=2, matured_leads=5)
        with pytest.raises(ValidationError):
            outcome(
                ExperimentOutcomeStatus.VALID_EXPOSURE,
                leads=5,
                matured_leads=2,
                conversions=3,
            )

    def test_identity_is_deterministic(self) -> None:
        first = experiment_outcome_id(RUN_ID, "probe_search_s1", 4)
        second = experiment_outcome_id(RUN_ID, "probe_search_s1", 4)
        assert first == second
        assert first != experiment_outcome_id(RUN_ID, "probe_search_s1", 5)


class TestStrategicPortfolio:
    def test_falsified_requires_reason(self) -> None:
        with pytest.raises(ValidationError):
            hypothesis(status=HypothesisStatus.FALSIFIED)
        hypothesis(
            status=HypothesisStatus.FALSIFIED,
            falsification_reason="matured search cohort converted zero of 30 leads",
        )

    def test_superseded_requires_successor(self) -> None:
        with pytest.raises(ValidationError):
            hypothesis(status=HypothesisStatus.SUPERSEDED)

    def test_active_ids_must_exist_and_be_open(self) -> None:
        with pytest.raises(ValidationError):
            portfolio(active=("h_missing",))
        falsified = hypothesis(
            status=HypothesisStatus.FALSIFIED,
            falsification_reason="zero conversions on matched support",
        )
        with pytest.raises(ValidationError):
            portfolio(hypotheses=(falsified,), active=(falsified.hypothesis_id,))

    def test_portfolio_hash_is_deterministic_and_content_sensitive(self) -> None:
        assert portfolio().portfolio_hash == portfolio().portfolio_hash
        changed = portfolio(hypotheses=(hypothesis(status=HypothesisStatus.RUNNING),))
        assert changed.portfolio_hash != portfolio().portfolio_hash


class TestPortfolioRevisionChain:
    def build_chain(self) -> list[StrategicPortfolioRevision]:
        first = revision(number=1, week=0, body=portfolio())
        second_body = portfolio(
            as_of_week=1,
            prior_portfolio_hash=first.portfolio_hash,
            hypotheses=(
                hypothesis(
                    status=HypothesisStatus.RUNNING,
                    evidence_refs=("decision:week-1",),
                ),
            ),
        )
        second = revision(number=2, week=1, body=second_body)
        return [first, second]

    def test_valid_chain_verifies(self) -> None:
        verify_portfolio_revision_chain(self.build_chain())

    def test_broken_hash_chain_is_rejected(self) -> None:
        first, second = self.build_chain()
        tampered = second.model_copy(
            update={
                "portfolio": second.portfolio.model_copy(
                    update={"prior_portfolio_hash": "b" * 64}
                )
            }
        )
        with pytest.raises(ValueError, match="hash chain broken"):
            verify_portfolio_revision_chain([first, tampered])

    def test_rewriting_evidence_history_is_rejected(self) -> None:
        first, second = self.build_chain()
        third_body = portfolio(
            as_of_week=2,
            prior_portfolio_hash=second.portfolio_hash,
            hypotheses=(
                hypothesis(
                    status=HypothesisStatus.RUNNING,
                    evidence_refs=("decision:rewritten",),
                ),
            ),
        )
        third = revision(number=3, week=2, body=third_body)
        with pytest.raises(ValueError, match="rewrote its evidence history"):
            verify_portfolio_revision_chain([first, second, third])

    def test_illegal_status_transition_is_rejected(self) -> None:
        first, second = self.build_chain()
        reason = "matured cohort converted zero of 30 leads"
        falsified_body = portfolio(
            as_of_week=2,
            prior_portfolio_hash=second.portfolio_hash,
            hypotheses=(
                hypothesis(
                    status=HypothesisStatus.FALSIFIED,
                    falsification_reason=reason,
                    evidence_refs=("decision:week-1", "outcome:week-2"),
                ),
            ),
        )
        third = revision(number=3, week=2, body=falsified_body)
        resurrected_body = portfolio(
            as_of_week=3,
            prior_portfolio_hash=third.portfolio_hash,
            hypotheses=(
                hypothesis(
                    status=HypothesisStatus.RUNNING,
                    falsification_reason=reason,
                    evidence_refs=("decision:week-1", "outcome:week-2"),
                ),
            ),
        )
        fourth = revision(number=4, week=3, body=resurrected_body)
        with pytest.raises(ValueError, match="illegal status transition"):
            verify_portfolio_revision_chain([first, second, third, fourth])

    def test_removed_hypothesis_is_rejected(self) -> None:
        first, second = self.build_chain()
        emptied = portfolio(
            as_of_week=2,
            prior_portfolio_hash=second.portfolio_hash,
            hypotheses=(hypothesis("h_other"),),
        )
        third = revision(number=3, week=2, body=emptied)
        with pytest.raises(ValueError, match="removed from the portfolio"):
            verify_portfolio_revision_chain([first, second, third])


class TestCandidateEvaluation:
    def test_eligibility_and_veto_codes_are_mutually_consistent(self) -> None:
        with pytest.raises(ValidationError):
            card(eligible=True, veto_codes=("over_budget",))
        with pytest.raises(ValidationError):
            card(eligible=False)
        vetoed = card(eligible=False, veto_codes=("bankruptcy_risk",))
        assert not vetoed.eligible

    def test_choice_must_select_an_eligible_unchanged_candidate(self) -> None:
        evaluated = evaluation_set(
            (
                card("cand_continuation"),
                card(
                    "cand_probe",
                    plan_hash="c" * 64,
                    eligible=False,
                    veto_codes=("over_budget",),
                ),
            )
        )
        validate_choice_against_set(
            choice_for(evaluated, "cand_continuation", rejected=("cand_probe",)),
            evaluated,
        )
        with pytest.raises(ValueError, match="not eligible"):
            validate_choice_against_set(choice_for(evaluated, "cand_probe"), evaluated)
        with pytest.raises(ValueError, match="not eligible"):
            validate_choice_against_set(choice_for(evaluated, "cand_unknown"), evaluated)

    def test_choice_detects_a_changed_evaluation_set(self) -> None:
        evaluated = evaluation_set((card("cand_continuation"),))
        chosen = choice_for(evaluated, "cand_continuation")
        mutated = evaluated.model_copy(
            update={"cards": (card("cand_continuation", expected_terminal_cash=1.0),)}
        )
        with pytest.raises(ValueError, match="evaluation set changed"):
            validate_choice_against_set(chosen, mutated)

    def test_selected_candidate_cannot_be_rejected(self) -> None:
        evaluated = evaluation_set((card("cand_continuation"),))
        with pytest.raises(ValidationError):
            choice_for(evaluated, "cand_continuation", rejected=("cand_continuation",))


class TestInMemoryStrategyRepository:
    @pytest.mark.asyncio
    async def test_portfolio_revision_replay_is_idempotent(self) -> None:
        repository = InMemoryRunRepository()
        first = revision(number=1, week=0, body=portfolio())
        await repository.append_portfolio_revision(first)
        again = await repository.append_portfolio_revision(first)
        assert again.portfolio_hash == first.portfolio_hash
        revisions = await repository.list_portfolio_revisions(RUN_ID)
        assert [item.revision for item in revisions] == [1]
        verify_portfolio_revision_chain(revisions)

    @pytest.mark.asyncio
    async def test_portfolio_revision_rejects_a_broken_chain(self) -> None:
        repository = InMemoryRunRepository()
        first = revision(number=1, week=0, body=portfolio())
        await repository.append_portfolio_revision(first)
        divergent = revision(
            number=2,
            week=1,
            body=portfolio(as_of_week=1, prior_portfolio_hash="b" * 64),
        )
        with pytest.raises(ConflictError):
            await repository.append_portfolio_revision(divergent)
        skipped = revision(
            number=3,
            week=1,
            body=portfolio(as_of_week=1, prior_portfolio_hash=first.portfolio_hash),
        )
        with pytest.raises(ConflictError):
            await repository.append_portfolio_revision(skipped)

    @pytest.mark.asyncio
    async def test_portfolio_revision_rejects_divergent_content_for_same_id(self) -> None:
        repository = InMemoryRunRepository()
        first = revision(number=1, week=0, body=portfolio())
        await repository.append_portfolio_revision(first)
        divergent = first.model_copy(
            update={
                "portfolio": portfolio(
                    hypotheses=(hypothesis(status=HypothesisStatus.RUNNING),)
                )
            }
        )
        with pytest.raises(ConflictError):
            await repository.append_portfolio_revision(divergent)

    @pytest.mark.asyncio
    async def test_experiment_outcome_append_is_idempotent_and_conflict_safe(
        self,
    ) -> None:
        repository = InMemoryRunRepository()
        recorded = outcome(
            ExperimentOutcomeStatus.EXPOSED_ZERO_CONVERSION,
            leads=8,
            matured_leads=8,
        )
        await repository.append_experiment_outcome(recorded)
        again = await repository.append_experiment_outcome(recorded)
        assert again == recorded
        divergent = recorded.model_copy(update={"conversions": 0, "leads": 9})
        with pytest.raises(ConflictError):
            await repository.append_experiment_outcome(divergent)
        by_commitment = await repository.list_commitment_experiment_outcomes(
            RUN_ID, "probe_search_s1"
        )
        assert by_commitment == [recorded]

    @pytest.mark.asyncio
    async def test_commitment_review_is_unique_per_week(self) -> None:
        repository = InMemoryRunRepository()
        review = CommitmentReview(
            id=commitment_review_id(RUN_ID, "probe_search_s1", 3),
            run_id=RUN_ID,
            commitment_id="probe_search_s1",
            week=3,
            verdict=CommitmentReviewVerdict.CONTINUE,
            reason="downside budget intact and cohort not yet mature",
            created_at=CREATED_AT,
        )
        await repository.append_commitment_review(review)
        await repository.append_commitment_review(review)
        divergent = review.model_copy(
            update={"verdict": CommitmentReviewVerdict.REVERT}
        )
        with pytest.raises(ConflictError):
            await repository.append_commitment_review(divergent)
        listed = await repository.list_commitment_reviews(RUN_ID, "probe_search_s1")
        assert [item.verdict for item in listed] == [CommitmentReviewVerdict.CONTINUE]

    @pytest.mark.asyncio
    async def test_evaluation_set_and_choice_round_trip(self) -> None:
        repository = InMemoryRunRepository()
        evaluated = evaluation_set((card("cand_continuation"),))
        await repository.append_candidate_evaluation_set(evaluated)
        await repository.append_candidate_evaluation_set(evaluated)
        stored = await repository.get_candidate_evaluation_set(RUN_ID, evaluated.week)
        assert stored is not None and stored.set_hash == evaluated.set_hash

        chosen = choice_for(evaluated, "cand_continuation")
        await repository.append_executive_choice(chosen)
        await repository.append_executive_choice(chosen)
        replayed = await repository.get_executive_choice(RUN_ID, evaluated.week)
        assert replayed is not None
        validate_choice_against_set(replayed, stored)

        divergent = chosen.model_copy(update={"decision_thesis": "different thesis"})
        with pytest.raises(ConflictError):
            await repository.append_executive_choice(divergent)
