from __future__ import annotations

from uuid import UUID

import pytest
from lithops.agents.common import (
    ExecutiveActionProposalOutput,
    ExecutiveChoiceOutput,
    StrategyPortfolioUpdateOutput,
)
from lithops.agents.common.structured_output import SpendAllocation
from lithops.agents.executive.agent import ExecutiveDecisionEngine
from lithops.application.executable_model_planning import ExecutableModelPlanner
from lithops.application.executive_selection import (
    ExecutiveAuthorityContext,
    assess_experiment,
    build_evaluation_cards,
    fallback_candidate_id,
    run_executive_selection,
)
from lithops.application.step_run import RunManager, StaticDecisionEngine
from lithops.application.weekly_planning import (
    MAX_STRATEGY_FAMILY_LENGTH,
    _with_distinct_candidate_identities,
)
from lithops.benchmark.fake import FakeBenchmarkAdapter
from lithops.domain.models import ActionCommand, ActionPlan, ExperimentProgram
from lithops.domain.strategy import (
    BusinessLever,
    CommitmentReviewVerdict,
    HypothesisStatus,
    ObjectiveSpec,
    StrategicHypothesis,
    StrategicPortfolio,
)
from lithops.infrastructure.persistence.repositories import InMemoryRunRepository
from lithops.simulator.models import (
    ResearchTierFacts,
    SimulationAction,
    SimulationState,
)
from lithops.simulator.strategy_search import CandidateSimulation, RobustnessLevel

RUN_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")


def action(name: str) -> SimulationAction:
    return SimulationAction(
        name=name,
        price_per_customer_weekly=100.0,
        marketing_spend=3_500.0,
        development_spend=1_750.0,
    )


def summary(
    name: str,
    *,
    bankruptcy: float = 0.01,
    going_concern: float = 0.0,
    downside: float = 100_000.0,
    expected: float = 150_000.0,
) -> CandidateSimulation:
    return CandidateSimulation(
        action=action(name),
        expected_ending_cash=expected,
        downside_ending_cash=downside,
        bankruptcy_probability=bankruptcy,
        going_concern_failure_probability=going_concern,
        expected_customer_growth=0.0,
        robustness=RobustnessLevel.MEDIUM,
        robust_utility=1.0,
        rollout_count=5,
    )


def operating_plan(name: str) -> ActionPlan:
    return ActionPlan(
        name=name,
        strategy_family=name,
        rationale="operational candidate",
        commands=[
            ActionCommand(
                tool="set_daily_spend",
                arguments={"operations": 500.0, "development": 250.0, "name": name},
                idempotency_key=f"{name}-key",
            )
        ],
    )


def experiment_plan(
    name: str,
    hypothesis_id: str,
    *,
    segment: str = "S1",
    channel: str = "search_ads",
) -> ActionPlan:
    program = ExperimentProgram(
        commitment_id=f"{hypothesis_id}-0",
        control="marketing",
        protocol_version="experiment-program-v2",
        started_week=0,
        minimum_maturity_week=1,
        maximum_end_week=1,
        baseline_value=3_500.0,
        treatment_value=6_000.0,
        maximum_cumulative_downside=3_000.0,
        expected_observation="leads from the probe channel",
        falsification_condition="no observation by week 1",
        target_segment=segment,
        target_channel=channel,
        baseline_configuration={"weekly_marketing_spend": 3_500.0},
        treatment_configuration={"weekly_marketing_spend": 6_000.0},
        measurement_plan=(
            {
                "source": "configuration",
                "metric": "marketing",
                "target_segment": segment,
                "target_channel": channel,
            },
            {
                "source": "cohort",
                "metric": "conversion_rate",
                "target_segment": segment,
                "target_channel": channel,
                "minimum_exposure": 30,
            },
        ),
    )
    return ActionPlan(
        name=name,
        strategy_family=name,
        rationale="experiment candidate",
        commands=[
            ActionCommand(
                tool="set_daily_spend",
                arguments={"operations": 500.0, "development": 250.0, "name": name},
                idempotency_key=f"{name}-key",
            )
        ],
        proposal_kind="experiment",
        hypothesis_id=hypothesis_id,
        experiment_control="marketing",
        evidence_regime="observed_operating_regime",
        experiment_expires_week=1,
        experiment_program=program,
    )


def portfolio_with(
    *hypotheses: StrategicHypothesis,
    active: tuple[str, ...] = (),
) -> StrategicPortfolio:
    return StrategicPortfolio(
        as_of_week=0,
        objective=ObjectiveSpec(horizon_day=500),
        binding_constraint="no observed conversions",
        active_hypothesis_ids=active,
        hypotheses=hypotheses,
        remaining_experiment_budget=5_000.0,
    )


def hypothesis(
    hypothesis_id: str,
    status: HypothesisStatus,
    *,
    segment: str | None = "S1",
    channel: str | None = "search_ads",
    successors: tuple[str, ...] = (),
) -> StrategicHypothesis:
    return StrategicHypothesis(
        hypothesis_id=hypothesis_id,
        causal_claim="claim",
        target_outcome="outcome",
        levers=(BusinessLever.ACQUISITION,),
        segment=segment,
        channel=channel,
        status=status,
        falsification_reason=(
            "matched-support falsification"
            if status is HypothesisStatus.FALSIFIED
            else None
        ),
        successor_hypothesis_ids=successors,
    )


class TestAssessExperiment:
    def test_semantically_unchanged_treatment_is_vetoed(self) -> None:
        plan = experiment_plan("executive_experiment_marketing_noop", "h_noop")
        program = plan.experiment_program
        assert program is not None
        plan = plan.model_copy(
            update={
                "experiment_program": program.model_copy(
                    update={
                        "treatment_configuration": program.baseline_configuration
                    }
                )
            },
            deep=True,
        )

        codes = assess_experiment(
            plan,
            portfolio=None,
        )

        assert "experiment_treatment_is_noop" in codes

    def test_falsified_hypothesis_id_is_vetoed(self) -> None:
        portfolio = portfolio_with(
            hypothesis("h_probe", HypothesisStatus.FALSIFIED)
        )
        codes = assess_experiment(
            experiment_plan("executive_experiment_marketing_h_probe", "h_probe"),
            portfolio=portfolio,
        )
        assert "hypothesis_falsified_in_support" in codes

    def test_matching_envelope_of_falsified_hypothesis_is_vetoed(self) -> None:
        portfolio = portfolio_with(
            hypothesis("h_old", HypothesisStatus.FALSIFIED),
            hypothesis("h_new", HypothesisStatus.PROPOSED),
            active=("h_new",),
        )
        codes = assess_experiment(
            experiment_plan("executive_experiment_marketing_h_new", "h_new"),
            portfolio=portfolio,
        )
        assert "hypothesis_falsified_in_support" in codes

    def test_a_multi_generation_lineage_is_not_vetoed_by_its_ancestor(self) -> None:
        # A live run built h_old -> h_mid -> h_new and was then locked out of its
        # own envelope, because only the direct successor was recognised.
        portfolio = portfolio_with(
            hypothesis(
                "h_old",
                HypothesisStatus.FALSIFIED,
                successors=("h_mid",),
            ),
            hypothesis(
                "h_mid",
                HypothesisStatus.SUPERSEDED,
                successors=("h_new",),
            ),
            hypothesis("h_new", HypothesisStatus.PROPOSED),
            active=("h_new",),
        )
        codes = assess_experiment(
            experiment_plan("executive_experiment_marketing_h_new", "h_new"),
            portfolio=portfolio,
        )
        assert "hypothesis_falsified_in_support" not in codes

    def test_an_unrelated_hypothesis_in_a_dead_envelope_is_still_vetoed(self) -> None:
        portfolio = portfolio_with(
            hypothesis("h_old", HypothesisStatus.FALSIFIED, successors=("h_mid",)),
            hypothesis("h_mid", HypothesisStatus.RUNNING),
            hypothesis("h_unrelated", HypothesisStatus.PROPOSED),
            active=("h_mid", "h_unrelated"),
        )
        codes = assess_experiment(
            experiment_plan("executive_experiment_marketing_h_unrelated", "h_unrelated"),
            portfolio=portfolio,
        )
        assert "hypothesis_falsified_in_support" in codes

    def test_declared_successor_reopens_the_envelope(self) -> None:
        portfolio = portfolio_with(
            hypothesis(
                "h_old",
                HypothesisStatus.FALSIFIED,
                successors=("h_new",),
            ),
            hypothesis("h_new", HypothesisStatus.PROPOSED),
            active=("h_new",),
        )
        codes = assess_experiment(
            experiment_plan("executive_experiment_marketing_h_new", "h_new"),
            portfolio=portfolio,
        )
        assert codes == ()

    def test_changed_segment_is_a_distinct_envelope(self) -> None:
        portfolio = portfolio_with(
            hypothesis("h_old", HypothesisStatus.FALSIFIED),
            hypothesis("h_new", HypothesisStatus.PROPOSED, segment="E1"),
            active=("h_new",),
        )
        codes = assess_experiment(
            experiment_plan(
                "executive_experiment_marketing_h_new",
                "h_new",
                segment="E1",
            ),
            portfolio=portfolio,
        )
        assert codes == ()

    def test_an_inactive_hypothesis_is_vetoed_but_cost_never_is(self) -> None:
        # Cost stopped being a veto: solvency has its own gates, the declared
        # downside cap is enforced weekly, and the price of learning is the
        # Executive's judgement. Sixteen consecutive over-budget vetoes once
        # blocked the exact tier pivot the portfolio named as the binding
        # constraint.
        portfolio = portfolio_with(
            hypothesis("h_active", HypothesisStatus.PROPOSED),
            active=("h_active",),
        )
        codes = assess_experiment(
            experiment_plan("executive_experiment_marketing_h_other", "h_other"),
            portfolio=portfolio,
        )
        assert "hypothesis_not_active" in codes
        assert "experiment_over_budget" not in codes


class TestEvaluationCards:
    def build(self, evaluations, plans, portfolio=None, inherited=False):
        return build_evaluation_cards(
            evaluations=tuple(evaluations),
            plans_by_candidate=plans,
            horizon_cash_by_candidate={},
            portfolio=portfolio,
            experiment_budget=5_000.0,
            inherited_going_concern_failure=inherited,
        )

    def test_unsafe_experiment_is_vetoed_and_operational_survives(self) -> None:
        plans = {
            "continuation": operating_plan("continuation"),
            "executive_experiment_marketing_h1": experiment_plan(
                "executive_experiment_marketing_h1", "h1"
            ),
        }
        # The experiment must *add* insolvency risk to be vetoed. Risk the
        # company carries either way is reported instead: as a flat veto this
        # gate left continuation the only eligible option on any zero-revenue
        # trajectory, forbidding the very probes that could create revenue.
        cards = self.build(
            [
                summary("continuation", bankruptcy=0.20, going_concern=0.9),
                summary(
                    "executive_experiment_marketing_h1",
                    bankruptcy=0.75,
                    going_concern=0.9,
                ),
            ],
            plans,
        )
        by_id = {card.candidate_id: card for card in cards}
        assert by_id["continuation"].eligible
        assert not by_id["executive_experiment_marketing_h1"].eligible
        assert "bankruptcy_gate" in by_id["executive_experiment_marketing_h1"].veto_codes

        carried = self.build(
            [
                summary("continuation", bankruptcy=0.20, going_concern=0.9),
                summary(
                    "executive_experiment_marketing_h1",
                    bankruptcy=0.20,
                    going_concern=0.9,
                ),
            ],
            plans,
        )
        probe = {card.candidate_id: card for card in carried}[
            "executive_experiment_marketing_h1"
        ]
        assert probe.eligible
        assert "insolvency_risk_carried" in probe.support_and_assumption_warnings

    def test_going_concern_risk_warns_and_never_vetoes(self) -> None:
        # As a veto this froze a live run into twelve consecutive weeks with a
        # single eligible candidate, tightening as the company weakened.
        plans = {
            "continuation": operating_plan("continuation"),
            "executive_experiment_marketing_h1": experiment_plan(
                "executive_experiment_marketing_h1", "h1"
            ),
        }
        cards = self.build(
            [
                summary("continuation", going_concern=0.70),
                summary(
                    "executive_experiment_marketing_h1",
                    bankruptcy=0.0,
                    going_concern=0.95,
                ),
            ],
            plans,
        )
        card = {c.candidate_id: c for c in cards}[
            "executive_experiment_marketing_h1"
        ]
        assert card.eligible
        assert "going_concern_gate" not in card.veto_codes
        # Materially worse than simply continuing, so it is named as added risk.
        assert "going_concern_risk_added" in card.support_and_assumption_warnings

    def test_risk_the_company_carries_anyway_is_named_as_carried(self) -> None:
        plans = {
            "continuation": operating_plan("continuation"),
            "executive_experiment_marketing_h1": experiment_plan(
                "executive_experiment_marketing_h1", "h1"
            ),
        }
        cards = self.build(
            [
                summary("continuation", going_concern=0.90),
                summary(
                    "executive_experiment_marketing_h1",
                    bankruptcy=0.0,
                    going_concern=0.92,
                ),
            ],
            plans,
        )
        card = {c.candidate_id: c for c in cards}[
            "executive_experiment_marketing_h1"
        ]
        assert card.eligible
        assert "going_concern_risk_carried" in card.support_and_assumption_warnings

    def test_insolvency_is_still_a_hard_veto(self) -> None:
        plans = {
            "continuation": operating_plan("continuation"),
            "executive_experiment_marketing_h1": experiment_plan(
                "executive_experiment_marketing_h1", "h1"
            ),
        }
        cards = self.build(
            [
                summary("continuation"),
                summary("executive_experiment_marketing_h1", bankruptcy=0.20),
            ],
            plans,
        )
        card = {c.candidate_id: c for c in cards}[
            "executive_experiment_marketing_h1"
        ]
        assert not card.eligible
        assert "bankruptcy_gate" in card.veto_codes

    def test_inherited_going_concern_does_not_veto(self) -> None:
        plans = {
            "continuation": operating_plan("continuation"),
            "executive_growth_0": operating_plan("executive_growth_0"),
        }
        cards = self.build(
            [
                summary("continuation", going_concern=1.0),
                summary("executive_growth_0", going_concern=1.0),
            ],
            plans,
            inherited=True,
        )
        assert all(card.eligible for card in cards)

    def test_a_costly_experiment_stays_eligible_and_carries_the_price(self) -> None:
        # The seed-83 lockout as a counterfactual: a standing-cost experiment
        # whose window downside exceeds the advisory budget is priced, warned,
        # and left for the Executive to judge — never vetoed for cost alone.
        plans = {
            "continuation": operating_plan("continuation"),
            "executive_experiment_marketing_h1": experiment_plan(
                "executive_experiment_marketing_h1", "h1"
            ),
        }
        cards = build_evaluation_cards(
            evaluations=(
                summary("continuation"),
                summary("executive_experiment_marketing_h1"),
            ),
            plans_by_candidate=plans,
            horizon_cash_by_candidate={},
            portfolio=None,
            experiment_budget=5_000.0,
            inherited_going_concern_failure=False,
            horizon_downside_cash_by_candidate={
                "continuation": {28: 900_000.0, 182: 950_000.0},
                "executive_experiment_marketing_h1": {
                    28: 891_000.0,
                    182: 870_000.0,
                },
            },
        )
        card = {c.candidate_id: c for c in cards}[
            "executive_experiment_marketing_h1"
        ]
        assert card.eligible
        # Window ends within 28 days, so the gap is 9k — not the 80k the
        # terminal horizon would have claimed.
        assert card.downside_cost_commitment_window == 9_000.0
        assert "experiment_budget_pressure" in card.support_and_assumption_warnings

    def test_a_window_downside_inside_budget_raises_no_warning(self) -> None:
        plans = {
            "continuation": operating_plan("continuation"),
            "executive_experiment_marketing_h1": experiment_plan(
                "executive_experiment_marketing_h1", "h1"
            ),
        }
        cards = build_evaluation_cards(
            evaluations=(
                summary("continuation"),
                summary("executive_experiment_marketing_h1"),
            ),
            plans_by_candidate=plans,
            horizon_cash_by_candidate={},
            portfolio=None,
            experiment_budget=5_000.0,
            inherited_going_concern_failure=False,
            horizon_downside_cash_by_candidate={
                "continuation": {28: 900_000.0},
                "executive_experiment_marketing_h1": {28: 897_000.0},
            },
        )
        card = {c.candidate_id: c for c in cards}[
            "executive_experiment_marketing_h1"
        ]
        assert card.eligible
        assert card.downside_cost_commitment_window == 3_000.0
        assert (
            "experiment_budget_pressure"
            not in card.support_and_assumption_warnings
        )

    def test_fallback_never_selects_an_experiment(self) -> None:
        plans = {
            "executive_experiment_marketing_h1": experiment_plan(
                "executive_experiment_marketing_h1", "h1"
            ),
            "continuation": operating_plan("continuation"),
        }
        cards = self.build(
            [
                summary("executive_experiment_marketing_h1", bankruptcy=0.0),
                summary("continuation", bankruptcy=0.09),
            ],
            plans,
        )
        assert fallback_candidate_id(cards) == "continuation"


class TestStrategyExhaustion:
    def test_all_falsified_recovery_paths_leave_honest_continuation(self) -> None:
        from lithops.application.weekly_planning import _candidate_pool_diagnostics

        portfolio = portfolio_with(
            hypothesis("h_probe", HypothesisStatus.FALSIFIED)
        )
        plans = {
            "continuation": operating_plan("continuation"),
            "executive_experiment_marketing_h_probe": experiment_plan(
                "executive_experiment_marketing_h_probe", "h_probe"
            ),
        }
        cards = build_evaluation_cards(
            evaluations=(
                summary("continuation"),
                summary("executive_experiment_marketing_h_probe"),
            ),
            plans_by_candidate=plans,
            horizon_cash_by_candidate={},
            portfolio=portfolio,
            experiment_budget=5_000.0,
            inherited_going_concern_failure=False,
        )
        by_id = {card.candidate_id: card for card in cards}
        assert not by_id["executive_experiment_marketing_h_probe"].eligible
        assert (
            "hypothesis_falsified_in_support"
            in by_id["executive_experiment_marketing_h_probe"].veto_codes
        )
        assert by_id["continuation"].eligible
        diagnostics = _candidate_pool_diagnostics(cards, plans)
        assert "strategy_exhausted" in diagnostics
        assert fallback_candidate_id(cards) == "continuation"


class SelectorEngine:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def select_candidate(self, *, brief: dict) -> ExecutiveChoiceOutput:
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def choice_output(candidate_id: str) -> ExecutiveChoiceOutput:
    return ExecutiveChoiceOutput(
        selected_candidate_id=candidate_id,
        decision_thesis="chosen for its learning value against the terminal objective",
        stop_or_pivot_condition="revert if the probe shows no leads",
    )


class RecordingEvents:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def __call__(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, payload))


def selection_cards():
    plans = {
        "continuation": operating_plan("continuation"),
        "executive_experiment_marketing_h1": experiment_plan(
            "executive_experiment_marketing_h1", "h1"
        ),
    }
    cards = build_evaluation_cards(
        evaluations=(
            summary("continuation"),
            summary("executive_experiment_marketing_h1", downside=97_500.0),
        ),
        plans_by_candidate=plans,
        horizon_cash_by_candidate={},
        portfolio=None,
        experiment_budget=5_000.0,
        inherited_going_concern_failure=False,
    )
    return cards


def research_summary(name: str, tier: int = 3) -> CandidateSimulation:
    base = summary(name)
    return base.model_copy(
        update={
            "action": base.action.model_copy(
                update={"research_project_tier": tier}
            ),
            "upside_ending_cash": 240_000.0,
        }
    )


def research_state(*, with_catalog: bool) -> SimulationState:
    return SimulationState(
        week=0,
        cash=1_000_000.0,
        revenue_weekly=0.0,
        customers=5.0,
        churn_rate=0.04,
        price_per_customer_weekly=5.8,
        weekly_acquisition=0.0,
        marketing_spend=0.0,
        development_spend=1_750.0,
        product_quality=0.2,
        capacity=50_000.0,
        reputation=0.5,
        research_catalog=(
            (
                ResearchTierFacts(
                    tier=3, cost=500_000.0, mean_weeks=3, mean_quality_boost=0.11
                ),
            )
            if with_catalog
            else ()
        ),
    )


class TestResearchCandidateCards:
    def build(self, *, with_catalog: bool):
        plans = {
            "continuation": operating_plan("continuation"),
            "research_push": operating_plan("research_push"),
        }
        return build_evaluation_cards(
            evaluations=(summary("continuation"), research_summary("research_push")),
            plans_by_candidate=plans,
            horizon_cash_by_candidate={},
            portfolio=None,
            experiment_budget=5_000.0,
            inherited_going_concern_failure=False,
            parameters_on_priors=(
                "research_quality_per_tier",
                "research_lag_weeks_per_tier",
            ),
            state=research_state(with_catalog=with_catalog),
        )

    def test_catalog_cost_bounds_the_downside_on_the_card(self) -> None:
        cards = {card.candidate_id: card for card in self.build(with_catalog=True)}
        card = cards["research_push"]
        assert card.downside_cost_commitment_window == 500_000.0
        assert (
            "research_downside_bounded_by_catalog_cost"
            in card.support_and_assumption_warnings
        )

    def test_unread_catalog_is_named_rather_than_silently_unpriced(self) -> None:
        cards = {card.candidate_id: card for card in self.build(with_catalog=False)}
        card = cards["research_push"]
        assert card.downside_cost_commitment_window is None
        assert (
            "research_catalog_unread_cost_and_return_unquantified"
            in card.support_and_assumption_warnings
        )

    def test_prior_only_levers_and_upside_reach_the_card(self) -> None:
        cards = {card.candidate_id: card for card in self.build(with_catalog=True)}
        research = cards["research_push"]
        continuation = cards["continuation"]
        # The research candidate names the prior-only parameters its own lever
        # leans on; continuation exercises no such lever and names none.
        assert research.levers_on_priors == (
            "research_quality_per_tier",
            "research_lag_weeks_per_tier",
        )
        assert continuation.levers_on_priors == ()
        assert research.upside_terminal_cash == 240_000.0


class TestRunExecutiveSelection:
    @pytest.mark.asyncio
    async def test_valid_choice_is_persisted_and_replayed(self) -> None:
        repository = InMemoryRunRepository()
        events = RecordingEvents()
        authority = ExecutiveAuthorityContext(
            strategy_repository=repository,
            emit_event=events,
        )
        engine = SelectorEngine(
            [choice_output("executive_experiment_marketing_h1")]
        )
        cards = selection_cards()
        outcome = await run_executive_selection(
            run_id=RUN_ID,
            week=3,
            executive=engine,
            authority=authority,
            cards=cards,
        )
        assert outcome.candidate_id == "executive_experiment_marketing_h1"
        assert outcome.selection_reason_code == "executive_candidate_selected"
        assert outcome.choice is not None
        assert [name for name, _ in events.events] == ["executive_candidate_selected"]

        replay = await run_executive_selection(
            run_id=RUN_ID,
            week=3,
            executive=SelectorEngine([]),
            authority=authority,
            cards=cards,
        )
        assert replay.candidate_id == outcome.candidate_id
        assert replay.choice == outcome.choice

    @pytest.mark.asyncio
    async def test_invalid_choice_retries_once_then_falls_back(self) -> None:
        repository = InMemoryRunRepository()
        events = RecordingEvents()
        authority = ExecutiveAuthorityContext(
            strategy_repository=repository,
            emit_event=events,
        )
        engine = SelectorEngine(
            [choice_output("cand_unknown"), choice_output("cand_unknown")]
        )
        outcome = await run_executive_selection(
            run_id=RUN_ID,
            week=3,
            executive=engine,
            authority=authority,
            cards=selection_cards(),
        )
        assert engine.calls == 2
        assert outcome.candidate_id == "continuation"
        assert (
            outcome.selection_reason_code
            == "executive_choice_invalid_safe_continuation"
        )
        assert outcome.choice is None
        assert [name for name, _ in events.events] == ["executive.choice_invalid"]
        assert await repository.get_executive_choice(RUN_ID, 3) is None

    @pytest.mark.asyncio
    async def test_provider_failure_is_safe_continuation(self) -> None:
        repository = InMemoryRunRepository()
        events = RecordingEvents()
        authority = ExecutiveAuthorityContext(
            strategy_repository=repository,
            emit_event=events,
        )
        outcome = await run_executive_selection(
            run_id=RUN_ID,
            week=3,
            executive=SelectorEngine([RuntimeError("provider down")] * 3),
            authority=authority,
            cards=selection_cards(),
        )
        assert outcome.candidate_id == "continuation"
        assert (
            outcome.selection_reason_code == "executive_unavailable_safe_continuation"
        )
        # The turn is surrendered only after every retry has failed, and each
        # failed attempt is on the audit trail.
        assert [name for name, _ in events.events] == ["executive.unavailable"] * 3

    @pytest.mark.asyncio
    async def test_repriced_conflict_keeps_the_executive_turn(self) -> None:
        repository = InMemoryRunRepository()
        events = RecordingEvents()
        authority = ExecutiveAuthorityContext(
            strategy_repository=repository,
            emit_event=events,
        )
        # A first pass persists the evaluation set for the week.
        await run_executive_selection(
            run_id=RUN_ID,
            week=3,
            executive=SelectorEngine([choice_output("continuation")]),
            authority=authority,
            cards=selection_cards(),
        )
        # A replay proposes the same candidates re-priced: the stored set is
        # the artifact of record and the Executive keeps its turn against it.
        repriced = tuple(
            card.model_copy(update={"expected_terminal_cash": 111_111.0})
            for card in selection_cards()
        )
        outcome = await run_executive_selection(
            run_id=RUN_ID,
            week=3,
            executive=SelectorEngine([choice_output("continuation")]),
            authority=authority,
            cards=repriced,
        )
        assert outcome.selection_reason_code == "executive_candidate_selected"
        assert "decision.evaluation_set_conflict" in [
            name for name, _ in events.events
        ]

    @pytest.mark.asyncio
    async def test_provider_retry_recovers_the_turn(self) -> None:
        repository = InMemoryRunRepository()
        events = RecordingEvents()
        authority = ExecutiveAuthorityContext(
            strategy_repository=repository,
            emit_event=events,
        )
        outcome = await run_executive_selection(
            run_id=RUN_ID,
            week=3,
            executive=SelectorEngine(
                [
                    RuntimeError("transient provider blip"),
                    choice_output("executive_experiment_marketing_h1"),
                ]
            ),
            authority=authority,
            cards=selection_cards(),
        )
        # One transient failure must not cost the Executive its turn.
        assert outcome.candidate_id == "executive_experiment_marketing_h1"
        assert outcome.selection_reason_code == "executive_candidate_selected"
        assert [name for name, _ in events.events][:1] == ["executive.unavailable"]


class FakeExecutiveV2:
    """A deterministic two-stage Executive for end-to-end weekly-loop tests."""

    prompt_version = "fake-executive-v2"

    def __init__(self) -> None:
        self.static = StaticDecisionEngine()
        self.portfolio_calls = 0
        self.selection_calls = 0
        self.last_portfolio_context: dict | None = None

    async def decide(self, *, run, observation):
        return await self.static.decide(run=run, observation=observation)

    async def update_strategy_portfolio(self, *, brief: dict):
        self.portfolio_calls += 1
        return StrategyPortfolioUpdateOutput(
            portfolio_thesis="probe whether search reaches S1 at all",
            binding_constraint="zero observed leads",
            new_hypotheses=(
                []
                if brief.get("portfolio")
                else [
                    {
                        "hypothesis_id": "h_probe_search",
                        "causal_claim": "S1 search ads produce leads",
                        "target_outcome": "nonzero weekly leads",
                        "levers": ["acquisition"],
                        "segment": "S1",
                        "channel": "search_ads",
                        "competing_predictions": (
                            "reach failure predicts zero leads; offer failure "
                            "predicts leads without conversions"
                        ),
                        "decisive_observation": "weekly leads in the probe week",
                    }
                ]
            ),
            active_hypothesis_ids=["h_probe_search"],
        )

    async def propose_actions(
        self,
        *,
        run,
        observation,
        decision_history=(),
        portfolio_context=None,
    ):
        self.last_portfolio_context = portfolio_context
        proposals = (
            ExecutiveActionProposalOutput(
                name="hold the operating baseline",
                hypothesis_id="h_probe_search",
                proposal_kind="operating",
                experiment_control="none",
                strategy_family="continuation",
                hypothesis="current levels preserve runway",
                expected_observation="cash declines at the observed baseline rate",
                rationale="keep the baseline while the probe is evaluated",
                catalog_price_multiplier=1.0,
                weekly_marketing_spend=3_500.0,
                daily_spend=SpendAllocation(operations=500.0, development=250.0),
                model_tier_a=1,
                model_tier_b=1,
                model_tier_c=1,
                usage_quota_a=300,
                usage_quota_b=300,
                usage_quota_c=300,
                capacity_tier=1,
                lead_promotion_fraction=0.0,
                target_channel="search_ads",
                target_segment="S1",
            ),
            ExecutiveActionProposalOutput(
                name="one-week S1 search reach probe",
                hypothesis_id="h_probe_search",
                proposal_kind="experiment",
                experiment_control="marketing",
                strategy_family="growth",
                hypothesis="S1 search ads produce leads",
                expected_observation="nonzero weekly leads in the probe week",
                rationale="distinguish reach failure from offer failure",
                catalog_price_multiplier=1.0,
                weekly_marketing_spend=6_000.0,
                daily_spend=SpendAllocation(operations=500.0, development=250.0),
                model_tier_a=1,
                model_tier_b=1,
                model_tier_c=1,
                usage_quota_a=300,
                usage_quota_b=300,
                usage_quota_c=300,
                capacity_tier=1,
                lead_promotion_fraction=0.0,
                target_channel="search_ads",
                target_segment="S1",
            ),
        )
        return tuple(
            ExecutiveDecisionEngine._proposal_plan(
                proposal,
                run=run,
                observation=observation,
                candidate_index=index,
            )
            for index, proposal in enumerate(proposals)
        )

    async def select_candidate(self, *, brief: dict) -> ExecutiveChoiceOutput:
        self.selection_calls += 1
        eligible = brief["eligible_candidate_ids"]
        experiment = next(
            (
                candidate_id
                for candidate_id in eligible
                if candidate_id.startswith("executive_experiment_")
            ),
            None,
        )
        return choice_output(experiment or eligible[0])


def v2_manager():
    repository = InMemoryRunRepository()
    executive = FakeExecutiveV2()
    planner = ExecutableModelPlanner(
        repository=repository,
        executive=executive,
        n_rollouts=5,
    )
    manager = RunManager(
        repository=repository,
        benchmark=FakeBenchmarkAdapter(),
        decision_engine=executive,
        planning_rollouts=5,
        executable_model_planner=planner,
        executive_authority_v2=True,
    )
    return manager, repository, executive


class TestEndToEndWeeklyLoop:
    @pytest.mark.asyncio
    async def test_two_stages_run_and_gemini_choice_is_executed(self) -> None:
        manager, repository, executive = v2_manager()
        run = await manager.create_run(horizon_days=28)
        result = await manager.step_run(run.id, request_id="v2-week-0")

        assert executive.portfolio_calls == 1
        assert executive.selection_calls == 1
        assert executive.last_portfolio_context is not None
        assert (
            result.decision.selection_reason_code == "executive_candidate_selected"
        )
        assert result.decision.action_plan.strategy_family.startswith(
            "executive_experiment_marketing_"
        )

        events = {event.type for event in await manager.list_events(run.id)}
        assert "strategy_portfolio_updated" in events
        assert "executive_candidate_selected" in events

        revisions = await repository.list_portfolio_revisions(run.id)
        assert [revision.revision for revision in revisions] == [1]
        evaluation_set = await repository.get_candidate_evaluation_set(run.id, 0)
        assert evaluation_set is not None
        stored_choice = await repository.get_executive_choice(run.id, 0)
        assert stored_choice is not None
        assert (
            stored_choice.selected_candidate_id
            == result.decision.action_plan.strategy_family
        )
        selected_card = next(
            card
            for card in evaluation_set.cards
            if card.candidate_id == stored_choice.selected_candidate_id
        )
        assert selected_card.plan_hash == result.decision.action_plan.semantic_hash

    @pytest.mark.asyncio
    async def test_replayed_week_does_not_call_the_provider_again(self) -> None:
        manager, repository, executive = v2_manager()
        run = await manager.create_run(horizon_days=28)
        await manager.step_run(run.id, request_id="v2-week-0")
        replay = await manager.step_run(run.id, request_id="v2-week-0")
        assert replay.replayed is True
        assert executive.portfolio_calls == 1
        assert executive.selection_calls == 1

    @pytest.mark.asyncio
    async def test_provider_failure_degrades_to_operational_pool(self) -> None:
        manager, repository, executive = v2_manager()

        async def failing_portfolio(*, brief):
            raise RuntimeError("provider down")

        async def failing_proposals(**kwargs):
            raise RuntimeError("provider down")

        executive.update_strategy_portfolio = failing_portfolio
        executive.propose_actions = failing_proposals
        run = await manager.create_run(horizon_days=28)
        result = await manager.step_run(run.id, request_id="v2-week-0")

        assert not result.decision.action_plan.strategy_family.startswith(
            "executive_experiment_"
        )
        assert result.decision.action_plan.experiment_program is None
        events = [event.type for event in await manager.list_events(run.id)]
        assert events.count("executive.unavailable") >= 2
        assert "executive_candidate_selected" in events

    @pytest.mark.asyncio
    async def test_the_commitment_verdict_records_what_the_executive_chose(self) -> None:
        # The commitment is offered as a choice, not imposed: whatever the
        # Executive runs that week is what the review records.
        manager, repository, executive = v2_manager()

        async def choose_rollback(*, brief: dict) -> ExecutiveChoiceOutput:
            eligible = brief["eligible_candidate_ids"]
            rollback = next(
                (item for item in eligible if item.startswith("experiment_revert_")),
                None,
            )
            return choice_output(rollback or eligible[0])

        run = await manager.create_run(horizon_days=28)
        first = await manager.step_run(run.id, request_id="v2-week-0")
        program = first.decision.action_plan.experiment_program
        assert program is not None

        executive.select_candidate = choose_rollback
        await manager.step_run(run.id, request_id="v2-week-1")
        reviews = await repository.list_commitment_reviews(
            run.id, program.commitment_id
        )
        assert [review.verdict for review in reviews] == [
            CommitmentReviewVerdict.REVERT
        ]

    @pytest.mark.asyncio
    async def test_commitment_review_and_outcome_are_recorded(self) -> None:
        manager, repository, executive = v2_manager()
        run = await manager.create_run(horizon_days=28)
        first = await manager.step_run(run.id, request_id="v2-week-0")
        program = first.decision.action_plan.experiment_program
        assert program is not None

        await manager.step_run(run.id, request_id="v2-week-1")
        reviews = await repository.list_commitment_reviews(
            run.id, program.commitment_id
        )
        assert [review.verdict for review in reviews] == [
            CommitmentReviewVerdict.ABANDONED
        ]
        outcomes = await repository.list_commitment_experiment_outcomes(
            run.id, program.commitment_id
        )
        assert len(outcomes) == 1
        assert outcomes[0].outcome_status.value == "no_exposure"


class TestBallotIdentity:
    """One candidate identity per plan, so no plan is invisible to the choice."""

    def test_a_repeat_of_an_entry_already_on_the_ballot_is_dropped(self) -> None:
        running = experiment_plan("executive_experiment_marketing_h01", "h01")
        reproposed = experiment_plan("executive_experiment_marketing_h01", "h01")

        plans, diagnostics = _with_distinct_candidate_identities(
            (running, reproposed)
        )

        assert [plan.strategy_family for plan in plans] == [
            "executive_experiment_marketing_h01"
        ]
        assert diagnostics == (
            {
                "candidate_id": "executive_experiment_marketing_h01",
                "resolution": "dropped",
            },
        )

    def test_a_differing_plan_keeps_its_place_under_a_distinct_identity(self) -> None:
        running = experiment_plan("executive_experiment_marketing_h01", "h01")
        modified = experiment_plan(
            "executive_experiment_marketing_h01", "h01"
        ).model_copy(
            update={
                "commands": [
                    ActionCommand(
                        tool="set_daily_spend",
                        arguments={"operations": 900.0, "development": 250.0},
                        idempotency_key="modified-key",
                    )
                ]
            }
        )

        plans, diagnostics = _with_distinct_candidate_identities((running, modified))

        assert [plan.strategy_family for plan in plans] == [
            "executive_experiment_marketing_h01",
            "executive_experiment_marketing_h01__alt2",
        ]
        assert diagnostics[0]["resolution"] == "renamed"
        # The commitment already on the ballot keeps the identity the run
        # committed to; only the newcomer moves.
        assert plans[0] is running

    def test_distinct_identities_survive_the_card_index(self) -> None:
        running = experiment_plan("executive_experiment_marketing_h01", "h01")
        modified = running.model_copy(
            update={"rationale": "a materially different probe", "commands": [
                ActionCommand(
                    tool="set_daily_spend",
                    arguments={"operations": 1_200.0},
                    idempotency_key="other-key",
                )
            ]}
        )

        plans, _ = _with_distinct_candidate_identities((running, modified))
        by_name = {plan.strategy_family: plan for plan in plans}

        assert len(by_name) == len(plans)

    def test_a_disambiguated_identity_fits_the_field_it_is_stored_in(self) -> None:
        family = "executive_experiment_targeted_development_" + "h" * 38
        assert len(family) == MAX_STRATEGY_FAMILY_LENGTH
        running = experiment_plan(family, "h01")
        modified = running.model_copy(
            update={
                "commands": [
                    ActionCommand(
                        tool="set_daily_spend",
                        arguments={"operations": 900.0},
                        idempotency_key="modified-key",
                    )
                ]
            }
        )

        plans, diagnostics = _with_distinct_candidate_identities((running, modified))

        assigned = diagnostics[0]["assigned_candidate_id"]
        assert len(assigned) <= MAX_STRATEGY_FAMILY_LENGTH
        assert assigned.endswith("__alt2")
        assert ActionPlan.model_validate(plans[1].model_dump()).strategy_family == assigned


class TestForecastProvenance:
    """A forecast should say how much of itself is a guess.

    Cards carried six-figure cash figures derived from parameters held at a
    confidence of 0.02, with nothing to distinguish an uninformative starting
    value from thirty weeks of evidence. The Executive read those figures as
    measurements, and refused a tier adoption on a forecast of -3.5M that
    reality contradicted within a fortnight.
    """

    def test_priors_behind_a_forecast_are_named_on_the_card(self) -> None:
        plans = {"continuation": operating_plan("continuation")}
        cards = build_evaluation_cards(
            evaluations=(summary("continuation"),),
            plans_by_candidate=plans,
            horizon_cash_by_candidate={},
            portfolio=None,
            experiment_budget=5_000.0,
            inherited_going_concern_failure=False,
            parameters_on_priors=("development_quality_response",),
        )
        assert cards[0].forecast_rests_on_priors == (
            "development_quality_response",
        )

    def test_a_forecast_on_evidence_names_nothing(self) -> None:
        plans = {"continuation": operating_plan("continuation")}
        cards = build_evaluation_cards(
            evaluations=(summary("continuation"),),
            plans_by_candidate=plans,
            horizon_cash_by_candidate={},
            portfolio=None,
            experiment_budget=5_000.0,
            inherited_going_concern_failure=False,
        )
        assert cards[0].forecast_rests_on_priors == ()


class TestUnmeasuredOutcome:
    """An absent parameter is not a zero.

    The enterprise quality floor shipped as 0.5 — a number nobody could vouch
    for that happened to match the benchmark's own. Removing it leaves the
    forecast showing no seats won, which reads exactly like "none would be won"
    unless the card says otherwise.
    """

    def test_engaging_enterprise_without_a_measurement_is_flagged(self) -> None:
        plan = operating_plan("executive_growth_0")
        cards = build_evaluation_cards(
            evaluations=(
                summary("continuation"),
                summary("executive_growth_0"),
            ),
            plans_by_candidate={
                "continuation": operating_plan("continuation"),
                "executive_growth_0": plan,
            },
            horizon_cash_by_candidate={},
            portfolio=None,
            experiment_budget=5_000.0,
            inherited_going_concern_failure=False,
            parameters_unmeasured=frozenset({"enterprise_quality_floor"}),
        )
        engaging = {card.candidate_id: card for card in cards}["executive_growth_0"]
        # This candidate does not engage enterprise, so nothing is claimed.
        assert "enterprise_outcome_unmeasured" not in (
            engaging.support_and_assumption_warnings
        )

    def test_no_enterprise_floor_ships_as_a_guess(self) -> None:
        from lithops.world_model.bootstrap import P0_PRIORS

        assert not [
            prior
            for prior in P0_PRIORS
            if prior.name.value == "enterprise_quality_floor"
        ], "what enterprise buyers require is bought in-run, never assumed"


class TestCostOfStandingStill:
    """The price of acting meant nothing without the price of not acting.

    Cards showed `downside_cost_commitment_window` and nothing else with a
    magnitude, so a live run filled its theses with boasts of how little each
    probe risked — "negligible downside commitment cost of $18.79" — while the
    forecast had continuation shedding several hundred thousand. Minimising the
    only number on the page was the rational reading of an incomplete page.
    """

    def test_a_candidate_states_what_it_changes_about_continuation(self) -> None:
        plans = {
            "continuation": operating_plan("continuation"),
            "executive_growth_0": operating_plan("executive_growth_0"),
        }
        cards = build_evaluation_cards(
            evaluations=(
                summary("continuation", expected=500_000.0),
                summary("executive_growth_0", expected=560_000.0),
            ),
            plans_by_candidate=plans,
            horizon_cash_by_candidate={},
            portfolio=None,
            experiment_budget=5_000.0,
            inherited_going_concern_failure=False,
        )
        by_id = {card.candidate_id: card for card in cards}
        assert by_id["continuation"].terminal_cash_versus_continuation == 0.0
        assert by_id["executive_growth_0"].terminal_cash_versus_continuation == (
            60_000.0
        )
