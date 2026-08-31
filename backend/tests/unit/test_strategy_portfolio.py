import json
from uuid import UUID

import pytest
from lithops.agents.common import (
    ExperimentInterpretationOutput,
    HypothesisProposalOutput,
    HypothesisStatusUpdateOutput,
    StrategyPortfolioUpdateOutput,
)
from lithops.application.strategy_portfolio import (
    apply_portfolio_update,
    build_strategic_evidence_brief,
    update_strategic_portfolio,
)
from lithops.domain.models import ObservationSnapshot, RunRecord
from lithops.domain.strategy import (
    EvidenceEnvelope,
    ExperimentOutcome,
    ExperimentOutcomeStatus,
    HypothesisStatus,
    ObjectiveSpec,
    StrategicPortfolio,
    experiment_outcome_id,
    verify_portfolio_revision_chain,
)
from lithops.infrastructure.persistence.repositories import InMemoryRunRepository
from pydantic import ValidationError

RUN_ID = UUID("99999999-9999-9999-9999-999999999999")
OBJECTIVE = ObjectiveSpec(horizon_day=500)


def outcome(
    commitment_id: str,
    status: ExperimentOutcomeStatus,
    *,
    hypothesis_id: str = "h_search_s1",
    leads: int = 0,
    matured_leads: int = 0,
    conversions: int = 0,
    measured_week: int | None = 4,
) -> ExperimentOutcome:
    return ExperimentOutcome(
        id=experiment_outcome_id(RUN_ID, commitment_id, measured_week or 0),
        run_id=RUN_ID,
        commitment_id=commitment_id,
        hypothesis_id=hypothesis_id,
        outcome_status=status,
        envelope=EvidenceEnvelope(segment="S1", channel="search_ads", quality_band=4),
        exposure_spend=2_000.0,
        leads=leads,
        matured_leads=matured_leads,
        conversions=conversions,
        started_week=2,
        measured_week=measured_week,
    )


def proposal(hypothesis_id: str = "h_search_s1", **overrides) -> HypothesisProposalOutput:
    payload = {
        "hypothesis_id": hypothesis_id,
        "causal_claim": "S1 search leads convert at the entry price",
        "target_outcome": "one matured conversion",
        "levers": ["acquisition"],
        "segment": "S1",
        "channel": "search_ads",
        "competing_predictions": (
            "reach failure predicts zero leads, offer failure zero conversions"
        ),
        "decisive_observation": "matured conversions from at least 10 leads",
    }
    payload.update(overrides)
    return HypothesisProposalOutput(**payload)


def test_hypothesis_channel_schema_is_gemini_compatible() -> None:
    channel_schema = HypothesisProposalOutput.model_json_schema()["properties"]["channel"]

    assert "enum" not in channel_schema
    assert HypothesisProposalOutput(
        **{
            **proposal().model_dump(mode="python"),
            "channel": "",
        }
    ).channel == ""
    with pytest.raises(ValidationError, match="observed channel code"):
        proposal(channel="unknown_channel")


def update_output(**overrides) -> StrategyPortfolioUpdateOutput:
    payload = {
        "portfolio_thesis": "test the reach-versus-offer split",
        "binding_constraint": "no observed conversions",
        "active_hypothesis_ids": [],
    }
    payload.update(overrides)
    return StrategyPortfolioUpdateOutput(**payload)


def apply(previous, update, outcomes=()):
    week = previous.as_of_week + 1 if previous is not None else 0
    return apply_portfolio_update(
        previous,
        update,
        as_of_week=week,
        objective=OBJECTIVE,
        remaining_experiment_budget=5_000.0,
        outcomes=tuple(outcomes),
    )


def bootstrap_portfolio():
    portfolio, diagnostics = apply(
        None,
        update_output(
            new_hypotheses=[proposal()],
            active_hypothesis_ids=["h_search_s1"],
        ),
    )
    assert diagnostics == ()
    return portfolio


class TestApplyPortfolioUpdate:
    def test_new_hypothesis_becomes_active(self) -> None:
        portfolio = bootstrap_portfolio()
        assert portfolio.active_hypothesis_ids == ("h_search_s1",)
        assert portfolio.hypotheses[0].status is HypothesisStatus.PROPOSED
        assert portfolio.prior_portfolio_hash is None

    def test_no_exposure_cannot_falsify(self) -> None:
        previous = bootstrap_portfolio()
        no_exposure = outcome(
            "probe_e1_linkedin",
            ExperimentOutcomeStatus.NO_EXPOSURE,
        )
        portfolio, diagnostics = apply(
            previous,
            update_output(
                outcome_interpretations=[
                    ExperimentInterpretationOutput(
                        commitment_id="probe_e1_linkedin",
                        hypothesis_id="h_search_s1",
                        interpretation="falsifies",
                        reasoning="no leads arrived",
                    )
                ],
                status_updates=[
                    HypothesisStatusUpdateOutput(
                        hypothesis_id="h_search_s1",
                        new_status="falsified",
                        reason="no exposure at all",
                    )
                ],
            ),
            outcomes=[no_exposure],
        )
        hypothesis = portfolio.hypotheses[0]
        assert hypothesis.status is HypothesisStatus.PROPOSED
        assert hypothesis.falsification_reason is None
        assert any(
            diagnostic.startswith("falsification_downgraded_to_inconclusive")
            for diagnostic in diagnostics
        )
        assert any(
            diagnostic.startswith("falsification_without_valid_exposure")
            for diagnostic in diagnostics
        )
        assert f"outcome:{no_exposure.id}" in hypothesis.evidence_refs

    def _falsified_portfolio(self) -> StrategicPortfolio:
        previous = bootstrap_portfolio()
        exposed = outcome(
            "probe_s1_search",
            ExperimentOutcomeStatus.EXPOSED_ZERO_CONVERSION,
            leads=12,
            matured_leads=12,
        )
        portfolio, diagnostics = apply(
            previous,
            update_output(
                outcome_interpretations=[
                    ExperimentInterpretationOutput(
                        commitment_id="probe_s1_search",
                        hypothesis_id="h_search_s1",
                        interpretation="falsifies",
                        reasoning="twelve matured leads converted zero",
                    )
                ],
                status_updates=[
                    HypothesisStatusUpdateOutput(
                        hypothesis_id="h_search_s1",
                        new_status="falsified",
                        reason="matched-support conversion falsified",
                    )
                ],
            ),
            outcomes=[exposed],
        )
        hypothesis = portfolio.hypotheses[0]
        assert hypothesis.status is HypothesisStatus.FALSIFIED
        assert hypothesis.falsification_reason == "matched-support conversion falsified"
        assert diagnostics == ("inactive_hypothesis_dropped:h_search_s1",)
        assert portfolio.active_hypothesis_ids == ()
        return portfolio

    def test_exposed_zero_conversion_supports_falsification(self) -> None:
        self._falsified_portfolio()

    def test_falsified_hypothesis_stays_falsified(self) -> None:
        falsified = self._falsified_portfolio()
        portfolio, diagnostics = apply(
            falsified,
            update_output(
                status_updates=[
                    HypothesisStatusUpdateOutput(
                        hypothesis_id="h_search_s1",
                        new_status="running",
                        reason="try again",
                    )
                ],
                active_hypothesis_ids=["h_search_s1"],
            ),
        )
        assert portfolio.hypotheses[0].status is HypothesisStatus.FALSIFIED
        assert any(
            diagnostic.startswith("illegal_status_transition") for diagnostic in diagnostics
        )
        assert any(
            diagnostic.startswith("inactive_hypothesis_dropped")
            for diagnostic in diagnostics
        )
        assert portfolio.active_hypothesis_ids == ()

    def test_successor_requires_material_difference(self) -> None:
        with pytest.raises(ValidationError):
            proposal(
                "h_search_e1",
                predecessor_hypothesis_id="h_search_s1",
                material_difference="",
            )

    def test_successor_links_predecessor(self) -> None:
        falsified = self._falsified_portfolio()
        portfolio, diagnostics = apply(
            falsified,
            update_output(
                new_hypotheses=[
                    proposal(
                        "h_search_e1",
                        segment="E1",
                        predecessor_hypothesis_id="h_search_s1",
                        material_difference="different segment with untested demand",
                    )
                ],
                active_hypothesis_ids=["h_search_e1"],
            ),
        )
        assert diagnostics == ()
        by_id = {item.hypothesis_id: item for item in portfolio.hypotheses}
        assert by_id["h_search_s1"].successor_hypothesis_ids == ("h_search_e1",)
        assert portfolio.active_hypothesis_ids == ("h_search_e1",)

    def test_unknown_predecessor_is_rejected(self) -> None:
        previous = bootstrap_portfolio()
        portfolio, diagnostics = apply(
            previous,
            update_output(
                new_hypotheses=[
                    proposal(
                        "h_new",
                        predecessor_hypothesis_id="h_missing",
                        material_difference="different mechanism",
                    )
                ],
            ),
        )
        assert "unknown_predecessor:h_missing" in diagnostics
        assert all(item.hypothesis_id != "h_new" for item in portfolio.hypotheses)


class FakeArchitectEngine:
    def __init__(self, update: StrategyPortfolioUpdateOutput) -> None:
        self.update = update
        self.calls = 0

    async def update_strategy_portfolio(self, *, brief: dict) -> StrategyPortfolioUpdateOutput:
        self.calls += 1
        json.dumps(brief)
        return self.update


class TestUpdateStrategicPortfolio:
    @pytest.mark.asyncio
    async def test_one_revision_per_week_and_replay_reconstruction(self) -> None:
        repository = InMemoryRunRepository()
        run = await repository.create_run(RunRecord(id=RUN_ID))
        engine = FakeArchitectEngine(
            update_output(
                new_hypotheses=[proposal()],
                active_hypothesis_ids=["h_search_s1"],
            )
        )
        observation = ObservationSnapshot(day=0, cash=200_000.0)
        first = await update_strategic_portfolio(
            run=run,
            observation=observation,
            executive=engine,
            strategy_repository=repository,
        )
        assert first is not None and not first.replayed
        assert first.revision.revision == 1
        replay = await update_strategic_portfolio(
            run=run,
            observation=observation,
            executive=engine,
            strategy_repository=repository,
        )
        assert replay is not None and replay.replayed
        assert replay.revision.portfolio_hash == first.revision.portfolio_hash
        assert engine.calls == 1
        verify_portfolio_revision_chain(await repository.list_portfolio_revisions(RUN_ID))

    @pytest.mark.asyncio
    async def test_second_week_extends_the_hash_chain(self) -> None:
        repository = InMemoryRunRepository()
        run = await repository.create_run(RunRecord(id=RUN_ID))
        engine = FakeArchitectEngine(
            update_output(
                new_hypotheses=[proposal()],
                active_hypothesis_ids=["h_search_s1"],
            )
        )
        first = await update_strategic_portfolio(
            run=run,
            observation=ObservationSnapshot(day=0, cash=200_000.0),
            executive=engine,
            strategy_repository=repository,
        )
        assert first is not None
        engine.update = update_output(active_hypothesis_ids=["h_search_s1"])
        second = await update_strategic_portfolio(
            run=run,
            observation=ObservationSnapshot(day=7, cash=190_000.0),
            executive=engine,
            strategy_repository=repository,
        )
        assert second is not None and second.revision.revision == 2
        chain = await repository.list_portfolio_revisions(RUN_ID)
        verify_portfolio_revision_chain(chain)
        assert (
            chain[1].portfolio.prior_portfolio_hash == chain[0].portfolio_hash
        )

    @pytest.mark.asyncio
    async def test_engine_without_architect_stage_is_skipped(self) -> None:
        repository = InMemoryRunRepository()
        run = await repository.create_run(RunRecord(id=RUN_ID))
        result = await update_strategic_portfolio(
            run=run,
            observation=ObservationSnapshot(day=0, cash=200_000.0),
            executive=object(),
            strategy_repository=repository,
        )
        assert result is None
        assert await repository.list_portfolio_revisions(RUN_ID) == []


class TestEvidenceBrief:
    def test_brief_is_compact_and_serializable(self) -> None:
        portfolio = bootstrap_portfolio()
        matured = outcome(
            "probe_s1_search",
            ExperimentOutcomeStatus.EXPOSED_ZERO_CONVERSION,
            leads=12,
            matured_leads=12,
            measured_week=3,
        )
        old = outcome(
            "probe_old",
            ExperimentOutcomeStatus.NO_EXPOSURE,
            measured_week=2,
        )
        brief = build_strategic_evidence_brief(
            run=RunRecord(id=RUN_ID, horizon_days=500),
            observation=ObservationSnapshot(day=28, cash=180_000.0),
            portfolio=portfolio.model_copy(update={"as_of_week": 3}),
            outcomes=(old, matured),
        )
        json.dumps(brief)
        newly = brief["newly_matured_outcomes"]
        assert [item["commitment_id"] for item in newly] == ["probe_s1_search"]
        # The full history is no longer re-sent: it was never referenced by the
        # prompt, duplicated every newly-matured entry, and grew without bound.
        assert "experiment_history" not in brief
        assert brief["portfolio"]["active_hypothesis_ids"] == ["h_search_s1"]
