"""The participation threshold: purchased floors, the forecast cliff, and cards.

Guards defect family #9: delivered quality is what customers judge, a purchased
floor is where judging starts, and a forecast that cannot price the crossing
makes every quality lever look like pure burn.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from lithops.application.executive_selection import build_evaluation_cards
from lithops.domain.insights import (
    InsightParseStatus,
    InsightRecord,
    measured_quality_floor_metrics,
)
from lithops.domain.models import ObservationSnapshot
from lithops.domain.world_model import WorldModelParameterName
from lithops.simulator.models import SimulationAction, SimulationState
from lithops.simulator.state_transition import advance_simulation_week
from lithops.world_model.bootstrap import P0_PRIORS, bootstrap_world_model

RUN_ID = UUID("33333333-3333-3333-3333-333333333333")


def _state(**updates: object) -> SimulationState:
    values: dict[str, object] = {
        "cash": 500_000,
        "revenue_weekly": 2_000,
        "customers": 10,
        "churn_rate": 0.04,
        "price_per_customer_weekly": 12,
        "weekly_acquisition": 0,
        "weekly_leads": 400,
        "total_leads": 400,
        "total_conversions": 0,
        "marketing_spend": 10_000,
        "development_spend": 5_000,
        "product_quality": 0.42,
        "model_tier_a": 2,
        "model_tier_b": 2,
        "model_tier_c": 2,
        "capacity": 5_000,
        "reputation": 0.6,
    }
    values.update(updates)
    return SimulationState.model_validate(values)


def _action(**updates: object) -> SimulationAction:
    values: dict[str, object] = {
        "name": "probe",
        "price_per_customer_weekly": 12.0,
        "marketing_spend": 10_000.0,
        "development_spend": 5_000.0,
        "segment_focus": 1.0,
    }
    values.update(updates)
    return SimulationAction.model_validate(values)


def _estimates() -> dict[WorldModelParameterName, float]:
    return {prior.name: prior.estimate for prior in P0_PRIORS}


def _insight(
    group: str, floor: float | None, *, week: int = 1, usage: float | None = 90.0
) -> InsightRecord:
    return InsightRecord(
        id=uuid4(),
        run_id=RUN_ID,
        week=week,
        tool="get_group_insights",
        target_group=group,
        request_identity=f"get_group_insights:{group}",
        quality_floor=floor,
        usage_units_per_day=usage,
        parse_status=InsightParseStatus.SUCCEEDED,
        parser_version="test",
        created_at=datetime(2026, 8, 30, tzinfo=UTC),
    )


class TestMeasuredFloorMetrics:
    def test_no_purchases_yield_no_metric_at_all(self) -> None:
        assert measured_quality_floor_metrics(()) == {}

    def test_unparsed_floor_is_not_a_zero(self) -> None:
        record = _insight("S2", None)
        assert measured_quality_floor_metrics((record,)) == {}

    def test_lowest_floor_across_groups_binds_and_split_is_by_buyer_kind(self) -> None:
        metrics = measured_quality_floor_metrics(
            (
                _insight("S2", 0.45),
                _insight("S1", 0.20),
                _insight("E1", 0.55),
                _insight("D_E03", 0.70),
                _insight("D_S05", 0.60),
            )
        )
        assert metrics["measured_quality_floor_individual"] == 0.20
        assert metrics["measured_quality_floor_enterprise"] == 0.55

    def test_latest_purchase_per_group_wins(self) -> None:
        metrics = measured_quality_floor_metrics(
            (
                _insight("S1", 0.20, week=1),
                _insight("S1", 0.35, week=9),
            )
        )
        assert metrics["measured_quality_floor_individual"] == 0.35


class TestParticipationGate:
    """Below the floor nothing unlocks; crossing it prices the cliff."""

    def test_unmeasured_floor_forecasts_no_unlock(self) -> None:
        base = advance_simulation_week(_state(), _action(), _estimates())
        # Zero conversions ever: the evidence-anchored funnel converts almost
        # nothing, floor or no floor — but never *because* a floor was assumed.
        assert base.weekly_acquisition < 2.0

    def test_below_a_measured_floor_matches_the_evidence_anchor(self) -> None:
        without = advance_simulation_week(_state(), _action(), _estimates())
        below = advance_simulation_week(
            _state(measured_quality_floor_individual=0.60),
            _action(),
            _estimates(),
        )
        # Delivered ~0.315 sits far under 0.60: the gate adds essentially
        # nothing, and quietly inventing customers below the floor would be the
        # opposite defect.
        assert abs(below.weekly_acquisition - without.weekly_acquisition) < 0.5

    def test_crossing_the_floor_prices_the_cliff(self) -> None:
        floor_state = _state(measured_quality_floor_individual=0.35)
        stand_still = advance_simulation_week(floor_state, _action(), _estimates())
        upgrade = advance_simulation_week(
            floor_state,
            # Tier 5 lifts delivered to 0.42 × 1.10 ≈ 0.46, over the 0.35 floor.
            _action(model_tier_a=5, model_tier_b=5, model_tier_c=5),
            _estimates(),
        )
        # The unlock is a share of the arriving leads, not a ratio of zero.
        assert upgrade.weekly_acquisition > stand_still.weekly_acquisition + 20.0

    def test_the_week_6_pathology_is_priced(self) -> None:
        """With a cleared floor, acting must out-acquire reverting.

        Honest-99 week 6 forecast *reverting* a tier upgrade as 2x better than
        adopting it, because revenue upside was invisible below the zero
        anchor. Once the floor is measured and cleared, the same comparison
        must favour the upgrade on the funnel it unlocks.
        """

        floor_state = _state(measured_quality_floor_individual=0.35)
        adopt = advance_simulation_week(
            floor_state,
            _action(model_tier_a=4, model_tier_b=4, model_tier_c=4),
            _estimates(),
        )
        revert = advance_simulation_week(
            floor_state,
            _action(model_tier_a=1, model_tier_b=1, model_tier_c=1),
            _estimates(),
        )
        assert adopt.weekly_acquisition > revert.weekly_acquisition


class TestQualityPositionWarnings:
    def _cards(self, state: SimulationState, action: SimulationAction):
        from backend.tests.unit.test_executive_authority_v2 import (  # type: ignore[import-not-found]
            operating_plan,
            summary,
        )

        candidate = summary("executive_growth_0")
        candidate = candidate.model_copy(update={"action": action})
        return build_evaluation_cards(
            evaluations=(summary("continuation"), candidate),
            plans_by_candidate={
                "continuation": operating_plan("continuation"),
                "executive_growth_0": operating_plan("executive_growth_0"),
            },
            horizon_cash_by_candidate={},
            portfolio=None,
            experiment_budget=5_000.0,
            inherited_going_concern_failure=False,
            state=state,
        )

    def test_quality_side_candidate_without_a_floor_is_flagged_unquantified(
        self,
    ) -> None:
        cards = self._cards(
            _state(),
            _action(name="executive_growth_0", model_tier_a=4),
        )
        card = {item.candidate_id: item for item in cards}["executive_growth_0"]
        assert "participation_floor_unmeasured" in card.support_and_assumption_warnings

    def test_delivered_below_a_measured_floor_is_named(self) -> None:
        cards = self._cards(
            _state(measured_quality_floor_individual=0.60),
            _action(name="executive_growth_0", model_tier_a=3),
        )
        card = {item.candidate_id: item for item in cards}["executive_growth_0"]
        assert (
            "delivered_quality_below_measured_floor"
            in card.support_and_assumption_warnings
        )


class TestEveryPurchasedInsightFieldHasAConsumer:
    """A bought measurement nobody reads is defect #9's quiet twin.

    `quality_floor` was parsed, stored, displayed — and consumed by nothing:
    the enterprise floor parameter it was meant to feed had no writer. Every
    decision-content field of InsightRecord must be consumed somewhere real:
    the observation injection, the state build, or the simulator.
    """

    def test_every_decision_content_field_is_consumed(self) -> None:
        decision_fields = (
            "willingness_to_pay_monthly",
            "usage_units_per_day",
            "quality_floor",
            "discovered_group",
        )
        root = Path(__file__).resolve().parents[2] / "src" / "lithops"
        consumers = "".join(
            (root / name).read_text()
            for name in (
                "application/step_run.py",
                "application/strategy_portfolio.py",
                "domain/insights.py",
            )
        )
        # A field "consumed" only by the summary shown to the LLM would not
        # count, so demand a reference outside `insight_summaries`.
        outside_summary = re.sub(
            r"def insight_summaries.*?(?=\ndef |\nclass |\Z)",
            "",
            consumers,
            flags=re.DOTALL,
        )
        for field in decision_fields:
            assert field in outside_summary, (
                f"purchased insight field '{field}' has no consumer beyond the "
                "summary shown to the model — a measurement nobody reads"
            )


class TestExecutablePathLearns:
    def test_sensitivities_are_never_suppressed_for_executable_decisions(self) -> None:
        """Recalibration was a structural no-op in the executable path.

        `_ensure_prediction` shipped empty cash sensitivities whenever a
        decision carried a model artifact, so residual attributions were empty
        and `recalibrate_world_model` could never fire — world_model_version
        stayed 1 in every run ever recorded.
        """

        source = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "lithops"
            / "application"
            / "step_run.py"
        ).read_text()
        body = source.split("async def _ensure_prediction", 1)[1].split("async def ", 1)[0]
        assert "estimate_cash_sensitivities" in body
        assert "model_artifact_id is not None" not in body, (
            "sensitivities must not be suppressed for executable-model decisions"
        )


def test_new_participation_parameters_are_bootstrapped_learnable() -> None:
    model = bootstrap_world_model(
        RUN_ID,
        ObservationSnapshot(
            day=0,
            cash=500_000,
            observed_at=datetime(2026, 8, 30, tzinfo=UTC),
        ),
    )
    names = {parameter.name for parameter in model.parameters}
    assert WorldModelParameterName.PARTICIPATION_CONVERSION_RATE in names
    assert WorldModelParameterName.PARTICIPATION_SOFTNESS in names
    by_name = {parameter.name: parameter for parameter in model.parameters}
    # Wide and weakly held: these are for the run to learn, not priors to trust.
    assert by_name[WorldModelParameterName.PARTICIPATION_CONVERSION_RATE].confidence <= 0.25


class TestRevealedDrift:
    """The run's own churn is a drift instrument; zero must never mean unknown."""

    def _decision(self, week, starting, churn, delivered):
        from types import SimpleNamespace

        return SimpleNamespace(
            week=week,
            observation=SimpleNamespace(metrics={"active_customers": starting}),
            actual_outcome=SimpleNamespace(
                metrics={
                    "churn_rate": churn,
                    "delivered_quality_plan_a": delivered,
                    "delivered_quality_plan_b": delivered,
                    "delivered_quality_plan_c": delivered * 0.8,
                }
            ),
        )

    def test_nothing_revealed_is_none_not_zero(self) -> None:
        from lithops.evaluation.trajectory import revealed_quality_bar_lower_bound

        history = (self._decision(1, 10, 0.05, 0.30),)
        assert revealed_quality_bar_lower_bound(history) is None

    def test_mass_churn_reveals_the_bar_above_delivered(self) -> None:
        from lithops.evaluation.trajectory import revealed_quality_bar_lower_bound

        history = (
            self._decision(1, 10, 0.05, 0.30),
            self._decision(2, 9, 0.6, 0.31),
            self._decision(3, 3, 0.9, 0.33),
        )
        assert revealed_quality_bar_lower_bound(history) == 0.33

    def test_tiny_base_churn_reveals_nothing(self) -> None:
        from lithops.evaluation.trajectory import revealed_quality_bar_lower_bound

        history = (self._decision(1, 1, 1.0, 0.30),)
        assert revealed_quality_bar_lower_bound(history) is None

    def test_revealed_bar_binds_the_participation_gate(self) -> None:
        without = advance_simulation_week(
            _state(measured_quality_floor_individual=0.10),
            _action(),
            _estimates(),
        )
        gated = advance_simulation_week(
            _state(
                measured_quality_floor_individual=0.10,
                revealed_quality_bar_lower_bound=0.60,
            ),
            _action(),
            _estimates(),
        )
        # A stale purchased floor of 0.10 would unlock the pool at delivered
        # ~0.31; the revealed bar of 0.60 forecloses it.
        assert gated.weekly_acquisition < without.weekly_acquisition - 10.0


class TestUnquantifiedReleases:
    def test_release_talk_without_numbers_is_counted(self) -> None:
        from lithops.benchmark.ceobench.market_signals import (
            unquantified_release_count,
        )

        rows = [
            {"day": 10, "content": "Big feature launch from CloudPeak. The bar just moved up."},
            {"day": 10, "content": "Interesting update from RivalTech."},
            {"day": 14, "content": "Minor release from ApexSaaS today."},
            {"day": 20, "content": "roughly a 0.30 quality boost from NexGen"},
            {"day": 22, "content": "NovaMind upgrades Plan B intelligence"},
        ]
        # Days 10 (one conversation), 14 → two unquantified releases; the
        # quantified post and our own launch are excluded.
        assert unquantified_release_count(rows, own_brand="NovaMind") == 2


class TestInsolvencyGateDoesNotLockTheExit:
    """Defect #12: a gate that forbids the only escape from a burn trajectory.

    With no revenue every candidate breaches the bankruptcy threshold, so a
    flat veto left continuation as the sole eligible option — which burns on,
    keeps the probability high, and vetoes next week's escape too. One run
    spent six consecutive weeks naming "zero exposure" as its binding
    constraint while its own top-ranked marketing probe sat vetoed. Risk the
    company carries anyway is reported; only risk a candidate *adds* is vetoed.
    """

    def _cards(self, continuation_risk: float, probe_risk: float):
        from backend.tests.unit.test_executive_authority_v2 import (  # type: ignore[import-not-found]
            operating_plan,
            summary,
        )

        continuation = summary("continuation").model_copy(
            update={"bankruptcy_probability": continuation_risk}
        )
        probe = summary("executive_experiment_marketing_expose").model_copy(
            update={"bankruptcy_probability": probe_risk}
        )
        return {
            card.candidate_id: card
            for card in build_evaluation_cards(
                evaluations=(continuation, probe),
                plans_by_candidate={
                    "continuation": operating_plan("continuation"),
                    "executive_experiment_marketing_expose": operating_plan(
                        "executive_experiment_marketing_expose"
                    ),
                },
                horizon_cash_by_candidate={},
                portfolio=None,
                experiment_budget=5_000.0,
                inherited_going_concern_failure=False,
            )
        }

    def test_risk_the_company_already_carries_does_not_veto_the_escape(self) -> None:
        cards = self._cards(continuation_risk=0.85, probe_risk=0.85)
        probe = cards["executive_experiment_marketing_expose"]
        assert probe.eligible, "the escape from a burn trajectory must stay selectable"
        assert "insolvency_risk_carried" in probe.support_and_assumption_warnings

    def test_risk_a_candidate_adds_is_still_vetoed(self) -> None:
        cards = self._cards(continuation_risk=0.12, probe_risk=0.90)
        probe = cards["executive_experiment_marketing_expose"]
        assert not probe.eligible
        assert "bankruptcy_gate" in probe.veto_codes

    def test_a_solvent_company_still_vetoes_a_reckless_probe(self) -> None:
        cards = self._cards(continuation_risk=0.0, probe_risk=0.60)
        assert not cards["executive_experiment_marketing_expose"].eligible
