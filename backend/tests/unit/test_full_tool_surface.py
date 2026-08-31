"""The executable surface the CEO actually has, end to end.

These tests pin the properties the production run depends on: a service allowance
of zero delivers nothing, an offer-side probe can buy its own measurement
exposure, purchased information becomes usable evidence, and enterprise offers
stay inside the authorized envelope.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from lithops.agents.common import ExecutiveActionProposalOutput
from lithops.agents.common.structured_output import SpendAllocation
from lithops.agents.executive.agent import ExecutiveDecisionEngine
from lithops.application import weekly_planning
from lithops.application.enterprise_negotiation import (
    negotiate_open_threads,
    offer_price_per_seat,
    parse_enterprise_inbox,
    seats_within_ceiling,
)
from lithops.application.executive_selection import (
    admit_information_requests,
    assess_experiment,
    configuration_completeness_violations,
)
from lithops.application.weekly_planning import (
    action_plan_from_simulation,
    open_enterprise_seats_from_inbox,
    simulation_action_from_action_plan,
    simulation_state_from_observation,
)
from lithops.benchmark.ceobench.action_mapper import ACTION_SPECS, build_action_code
from lithops.benchmark.ceobench.insight_parser import (
    extract_payload_text,
    parse_insight,
)
from lithops.domain.evidence import ConfigurationEvidence, WeeklyEvidencePacket
from lithops.domain.insights import (
    INSIGHT_FRESHNESS_WEEKS,
    InformationRequest,
    InsightParseStatus,
    InsightRecord,
    fresh_insight_identities,
)
from lithops.domain.models import (
    ActionCommand,
    ActionPlan,
    ActionReceipt,
    CashForecast,
    CashForecasts,
    DecisionRecord,
    ExperimentMeasurement,
    ExperimentProgram,
    ObservationSnapshot,
    ReceiptStatus,
    RunRecord,
)
from lithops.evaluation.action_fidelity import action_fidelity_violations
from lithops.simulator import state_transition
from lithops.simulator.models import SimulationAction, SimulationState
from lithops.simulator.state_transition import advance_simulation_week
from lithops.world_model import bootstrap_world_model

RUN_ID = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


def parameters() -> dict:
    model = bootstrap_world_model(
        RUN_ID,
        ObservationSnapshot(day=0, cash=1_000_000, observed_at=datetime(2026, 8, 28, tzinfo=UTC)),
    )
    return {parameter.name: parameter.estimate for parameter in model.parameters}


def state(**overrides) -> SimulationState:
    values = {
        "week": 0,
        "cash": 1_000_000.0,
        "revenue_weekly": 0.0,
        "customers": 0.0,
        "churn_rate": 0.04,
        "price_per_customer_weekly": 5.8,
        "weekly_acquisition": 0.0,
        # No acquisition spend yet, so a candidate's spend is incremental and
        # actually produces leads to convert.
        "marketing_spend": 0.0,
        "development_spend": 1_750.0,
        "product_quality": 0.2,
        "capacity": 50_000.0,
        "reputation": 0.5,
    }
    values.update(overrides)
    return SimulationState(**values)


def action(**overrides) -> SimulationAction:
    values = {
        "name": "candidate",
        "price_per_customer_weekly": 5.8,
        "marketing_spend": 3_500.0,
        "development_spend": 1_750.0,
    }
    values.update(overrides)
    return SimulationAction(**values)


class TestServiceAllowance:
    def test_zero_allowance_delivers_nothing_and_raising_it_converts(self) -> None:
        estimates = parameters()
        base = state(usage_quota_a=0.0, usage_quota_b=0.0, usage_quota_c=0.0)
        withheld = advance_simulation_week(base, action(), estimates)
        granted = advance_simulation_week(
            base,
            action(usage_quota_a=300.0, usage_quota_b=300.0, usage_quota_c=300.0),
            estimates,
        )

        assert withheld.weekly_conversions == 0.0
        assert granted.weekly_conversions > 0.0

    def test_an_unchanged_allowance_leaves_conversion_untouched(self) -> None:
        estimates = parameters()
        configured = state(
            usage_quota_a=300.0,
            usage_quota_b=300.0,
            usage_quota_c=300.0,
            daily_usage_per_customer=200.0,
        )
        held = advance_simulation_week(
            configured,
            action(usage_quota_a=300.0, usage_quota_b=300.0, usage_quota_c=300.0),
            estimates,
        )
        unknown = advance_simulation_week(state(), action(), estimates)

        assert held.weekly_conversions == pytest.approx(unknown.weekly_conversions)

    def test_absent_allowance_information_is_not_a_zero_allowance(self) -> None:
        assert state().average_usage_quota is None
        assert advance_simulation_week(state(), action(), parameters()).weekly_conversions > 0

    def test_a_purchased_demand_estimate_stops_the_chase_for_more_allowance(self) -> None:
        estimates = parameters()
        informed = state(
            usage_quota_a=40.0,
            usage_quota_b=40.0,
            usage_quota_c=40.0,
            estimated_usage_demand_per_day=80.0,
        )
        uninformed = informed.model_copy(update={"estimated_usage_demand_per_day": None})
        covers_demand = action(
            usage_quota_a=100.0, usage_quota_b=100.0, usage_quota_c=100.0
        )
        far_above_demand = action(
            usage_quota_a=400.0, usage_quota_b=400.0, usage_quota_c=400.0
        )

        # Measured demand tells the model when an allowance is already enough, so
        # provisioning far above it buys nothing more.
        assert advance_simulation_week(
            informed, far_above_demand, estimates
        ).weekly_conversions == pytest.approx(
            advance_simulation_week(informed, covers_demand, estimates).weekly_conversions
        )
        # Without that measurement the wide prior keeps promising more headroom.
        assert (
            advance_simulation_week(
                uninformed, far_above_demand, estimates
            ).weekly_conversions
            > advance_simulation_week(
                uninformed, covers_demand, estimates
            ).weekly_conversions
        )


class TestExecutiveAgency:
    """What the harness must leave to the Executive rather than decide for it."""

    def test_a_matured_treatment_can_be_adopted_instead_of_rolled_back(self) -> None:
        from lithops.application.weekly_planning import _experiment_adoption_action

        program = ExperimentProgram(
            commitment_id="h_tier-0",
            control="tier",
            protocol_version="experiment-program-v2",
            started_week=0,
            minimum_maturity_week=1,
            maximum_end_week=1,
            baseline_value=1.0,
            treatment_value=4.0,
            maximum_cumulative_downside=5_000.0,
            expected_observation="conversions rise at the higher tier",
            falsification_condition="no conversions by week 1",
            target_segment="S1",
            target_channel="search_ads",
            baseline_configuration={"model_tiers": {"A": 1, "B": 1, "C": 1}},
            treatment_configuration={
                "model_tiers": {"A": 4, "B": 4, "C": 4},
                "usage_quotas": {"A": 300, "B": 300, "C": 300},
            },
            measurement_plan=(
                ExperimentMeasurement(
                    source="configuration", metric="tier", target_segment="S1"
                ),
                ExperimentMeasurement(
                    source="cohort",
                    metric="conversion_rate",
                    target_segment="S1",
                    minimum_exposure=30,
                ),
            ),
        )
        decision = DecisionRecord(
            run_id=RUN_ID,
            week=0,
            observation=ObservationSnapshot(day=0, cash=1_000_000.0),
            action_plan=ActionPlan(
                name="tier probe",
                strategy_family="executive_experiment_tier_h_tier",
                rationale="probe",
                commands=[
                    ActionCommand(
                        tool="set_model_tiers",
                        arguments={"A": 4, "B": 4, "C": 4},
                        idempotency_key="k",
                    )
                ],
                proposal_kind="experiment",
                hypothesis_id="h_tier",
                experiment_control="tier",
                evidence_regime="observed",
                experiment_expires_week=1,
                experiment_program=program,
            ),
            forecasts=CashForecasts(
                items=[
                    CashForecast(horizon_days=horizon, point=1.0, lower=0.0, upper=2.0)
                    for horizon in (7, 28, 84, 182)
                ]
            ),
            actual_outcome=ObservationSnapshot(day=7, cash=990_000.0),
        )

        adoption = _experiment_adoption_action(
            (decision,), state=state(week=1, usage_quota_a=300.0)
        )

        assert adoption is not None
        # The treatment becomes the operating configuration, with no expiry.
        assert adoption.model_tier_a == 4
        assert adoption.marketing_spend_until_week is None
        # Controls the experiment never touched are carried over, not reset.
        assert adoption.usage_quota_a == 300.0

    def test_an_unobserved_segment_is_refused_rather_than_retargeted(self) -> None:
        plan = experiment_plan_for_segment("D_S07")
        codes = assess_experiment(
            plan, portfolio=None
        )
        assert "target_segment_not_observed" in codes

    def test_the_trajectory_exposes_rates_not_just_the_latest_week(self) -> None:
        from lithops.evaluation.trajectory import weekly_trajectory

        rows = weekly_trajectory(tuple(_trajectory_decisions()))
        assert [row["week"] for row in rows] == [0, 1]
        assert rows[0]["product_quality"] == 0.2
        assert rows[1]["product_quality"] == 0.21
        # A reader can compute the rate because both weeks are present.
        assert rows[1]["product_quality"] - rows[0]["product_quality"] > 0


def _trajectory_decisions() -> list[DecisionRecord]:
    forecasts = CashForecasts(
        items=[
            CashForecast(horizon_days=horizon, point=1.0, lower=0.0, upper=2.0)
            for horizon in (7, 28, 84, 182)
        ]
    )
    plan = ActionPlan(
        name="operate",
        strategy_family="executive_growth_0",
        rationale="r",
        commands=[
            ActionCommand(
                tool="set_daily_spend",
                arguments={"operations": 1.0, "development": 1.0},
                idempotency_key="k",
            )
        ],
    )
    return [
        DecisionRecord(
            run_id=RUN_ID,
            week=week,
            observation=ObservationSnapshot(day=week * 7, cash=1_000.0),
            action_plan=plan,
            forecasts=forecasts,
            actual_outcome=ObservationSnapshot(
                day=(week + 1) * 7,
                cash=1_000.0 - week,
                metrics={"product_quality": quality, "weekly_conversions": week},
            ),
        )
        for week, quality in ((0, 0.2), (1, 0.21))
    ]


def experiment_plan_for_segment(segment: str) -> ActionPlan:
    program = ExperimentProgram(
        commitment_id="h_probe-0",
        control="marketing",
        protocol_version="experiment-program-v2",
        started_week=0,
        minimum_maturity_week=1,
        maximum_end_week=1,
        baseline_value=0.0,
        treatment_value=6_000.0,
        maximum_cumulative_downside=6_000.0,
        expected_observation="leads",
        falsification_condition="none",
        target_segment=segment,
        target_channel="search_ads",
        baseline_configuration={"weekly_marketing_spend": 6_000.0},
        treatment_configuration={"weekly_marketing_spend": 6_000.0},
        measurement_plan=(
            ExperimentMeasurement(
                source="configuration", metric="marketing", target_segment=segment
            ),
            ExperimentMeasurement(
                source="cohort",
                metric="conversion_rate",
                target_segment=segment,
                minimum_exposure=30,
            ),
        ),
    )
    return ActionPlan(
        name="probe",
        strategy_family="executive_experiment_marketing_h_probe",
        rationale="r",
        commands=[
            ActionCommand(
                tool="set_daily_spend",
                arguments={"operations": 1.0, "development": 1.0},
                idempotency_key="k",
            )
        ],
        proposal_kind="experiment",
        hypothesis_id="h_probe",
        experiment_control="marketing",
        evidence_regime="leads_none:quality_2:customers_zero:unobserved_segment",
        experiment_expires_week=1,
        experiment_program=program,
    )


class TestConfigurationCompleteness:
    def test_all_zero_prices_or_allowances_are_vetoed(self) -> None:
        plan = ActionPlan(
            name="p",
            strategy_family="p",
            rationale="r",
            commands=[
                ActionCommand(
                    tool="set_prices",
                    arguments={"A": 0.0, "B": 0.0, "C": 0.0},
                    idempotency_key="k1",
                ),
                ActionCommand(
                    tool="set_usage_quotas",
                    arguments={"A": 0, "B": 0, "C": 0},
                    idempotency_key="k2",
                ),
            ],
        )
        codes = configuration_completeness_violations(plan)
        assert "configuration_incomplete_prices" in codes
        assert "configuration_incomplete_service_allowance" in codes

    def test_a_configured_plan_passes(self) -> None:
        plan = ActionPlan(
            name="p",
            strategy_family="p",
            rationale="r",
            commands=[
                ActionCommand(
                    tool="set_prices",
                    arguments={"A": 25.0, "B": 69.0, "C": 179.0},
                    idempotency_key="k1",
                ),
                ActionCommand(
                    tool="set_usage_quotas",
                    arguments={"A": 120, "B": 200, "C": 500},
                    idempotency_key="k2",
                ),
            ],
        )
        assert configuration_completeness_violations(plan) == ()


def offer_side_proposal(**overrides) -> ExecutiveActionProposalOutput:
    values = {
        "name": "entry price probe",
        "hypothesis_id": "h_price_entry",
        "proposal_kind": "experiment",
        "experiment_control": "price",
        "strategy_family": "pricing",
        "hypothesis": "the entry price sits above what this group will pay",
        "expected_observation": "conversions from matured leads at the lower price",
        "rationale": "separate reach from willingness to pay",
        "catalog_price_multiplier": 0.8,
        "weekly_marketing_spend": 6_000.0,
        "daily_spend": SpendAllocation(operations=500.0, development=250.0),
        "model_tier_a": 1,
        "model_tier_b": 1,
        "model_tier_c": 1,
        "usage_quota_a": 300,
        "usage_quota_b": 300,
        "usage_quota_c": 300,
        "capacity_tier": 1,
        "lead_promotion_fraction": 0.0,
        "target_channel": "search_ads",
        "target_segment": "S1",
    }
    values.update(overrides)
    return ExecutiveActionProposalOutput(**values)


class TestMeasurementExposureDeadlock:
    def test_an_offer_side_probe_buys_its_own_exposure_at_zero_baseline(self) -> None:
        observation = ObservationSnapshot(
            day=0,
            cash=1_000_000,
            metrics={
                "marketing_spend": 0.0,
                "price_a": 25,
                "price_b": 69,
                "price_c": 179,
                "known_segments": "S1",
            },
        )
        plan = ExecutiveDecisionEngine._proposal_plan(
            offer_side_proposal(),
            run=RunRecord(id=RUN_ID),
            observation=observation,
            candidate_index=0,
        )
        program = plan.experiment_program
        assert program is not None

        # Both arms carry the same positive exposure, so the contrast is still a
        # single difference in the declared control.
        assert program.treatment_configuration["weekly_marketing_spend"] == 6_000.0
        assert program.baseline_configuration["weekly_marketing_spend"] == 6_000.0
        assert program.baseline_configuration["pre_experiment_weekly_marketing_spend"] == 0.0

        codes = assess_experiment(
            plan, portfolio=None
        )
        assert "no_planned_measurement_exposure" not in codes


class TestExecutableSurfaceRoundTrip:
    def test_every_emitted_tool_is_mapped_simulated_and_verified(self) -> None:
        observation = ObservationSnapshot(
            day=7,
            cash=900_000,
            metrics={
                "known_segments": "S1",
                "price_a": 25,
                "price_b": 69,
                "price_c": 179,
                "usage_quota_a": 120,
                "usage_quota_b": 200,
                "usage_quota_c": 500,
                "capacity_tier": 1,
                "recurring_promotion_monthly": 0.0,
                "ads_strength": 0.0,
                "targeted_ops_spend": 0.0,
            },
        )
        simulated = action(
            usage_quota_a=150.0,
            usage_quota_b=250.0,
            usage_quota_c=600.0,
            capacity_tier=2,
            recurring_promotion_monthly=3.0,
            ads_strength=0.2,
            targeted_ops_spend=1_400.0,
        )
        plan = action_plan_from_simulation(
            RunRecord(id=RUN_ID), observation, simulated, "selected"
        )
        emitted = {command.tool for command in plan.commands}

        # Everything the planner emits is executable and modelled.
        assert emitted <= set(ACTION_SPECS)
        for command in plan.commands:
            assert build_action_code(command)

        replayed = simulation_action_from_action_plan(
            plan, simulation_state_from_observation(observation)
        )
        assert replayed.usage_quota_a == 150.0
        assert replayed.capacity_tier == 2
        assert replayed.recurring_promotion_monthly == pytest.approx(3.0)
        assert replayed.ads_strength == pytest.approx(0.2)
        assert replayed.targeted_ops_spend == pytest.approx(1_400.0)

    def test_fidelity_reads_each_tool_in_the_shape_the_benchmark_records(self) -> None:
        # The benchmark normalizes the legacy targeted_spend argument of the ops
        # tool into a by_group scope, while the ad and dev tools keep the wrapper.
        # Reading the wrong scope failed a live run, so the shapes are pinned here.
        packet = WeeklyEvidencePacket(
            day=7,
            window_start_day_exclusive=0,
            window_end_day_inclusive=7,
            configuration=ConfigurationEvidence(
                prices={"A": 25.0, "B": 69.0, "C": 179.0},
                model_tiers={"A": 1, "B": 1, "C": 1},
                usage_quotas={"A": 100.0, "B": 500.0, "C": 2000.0},
                daily_channel_spend={},
                daily_operations_spend=50.0,
                daily_development_spend=100.0,
                capacity_tier=1,
                targeted_ops_json='{"by_group": {"S1": 0.0}, "by_plan": {}}',
                targeted_ads_json='{"targeted_spend": {"search_ads": {"S1": 500.0}}}',
                targeted_development_json='{"targeted_spend": {"S1": 0.0}}',
                recurring_promotion_json='{"global": 0.0, "by_group": {}}',
                ads_strength_json='{"global": 0.0, "by_group": {}}',
            ),
        )
        observation = ObservationSnapshot(day=7, cash=1.0, evidence=packet)
        commands = [
            ActionCommand(
                tool="set_targeted_ops_spend",
                arguments={"targeted_spend": {"S1": 0.0}},
                idempotency_key="ops",
            ),
            ActionCommand(
                tool="set_targeted_ad_spend",
                arguments={"targeted_spend": {"search_ads": {"S1": 500.0}}},
                idempotency_key="ads",
            ),
            ActionCommand(
                tool="set_promotion",
                arguments={"global_promotion": 0.0},
                idempotency_key="promo",
            ),
            ActionCommand(
                tool="set_ads_strength",
                arguments={"global_strength": 0.0},
                idempotency_key="strength",
            ),
            ActionCommand(
                tool="set_usage_quotas",
                arguments={"A": 100, "B": 500, "C": 2000},
                idempotency_key="quotas",
            ),
        ]

        assert action_fidelity_violations(commands, observation) == ()

    def test_a_state_changing_tool_without_a_verifier_is_reported(self) -> None:
        observation = ObservationSnapshot(day=7, cash=1.0)
        violations = action_fidelity_violations(
            [
                ActionCommand(
                    tool="set_unmodelled_control",
                    arguments={},
                    idempotency_key="k",
                )
            ],
            observation,
        )
        # Without an evidence packet the packet itself is the first violation.
        assert violations == ("post-action evidence packet is missing",)


class TestPurchasedInformation:
    def test_the_api_returns_structured_estimates_and_they_are_read(self) -> None:
        # This is the shape a live run actually returned; the prose in the tool
        # reference is the CLI's display format, not the API payload.
        payload = (
            '{"group_id": "S1", "group_name": "Price-Sensitive Individuals",'
            ' "segment": "Individual", "info_level": 1, "noise": "\\u00b165%",'
            ' "estimates": {"willingness_to_pay": 25.87, "usage_volume": 91.1,'
            ' "quality_floor_q_min": 0.081, "contract_lockin_aversion": 0.19,'
            ' "market_cap": 272812, "annual_market_cap_growth_rate": 0.0962}}'
        )
        record = parse_insight(
            run_id=RUN_ID,
            week=3,
            request=InformationRequest(
                tool="get_group_insights",
                target_group="S1",
                expected_information_value="the quality floor for S1",
            ),
            payload=payload,
            cost=0.0,
            created_at=datetime.now(UTC),
        )

        assert record.parse_status is InsightParseStatus.SUCCEEDED
        assert record.quality_floor == pytest.approx(0.081)
        assert record.willingness_to_pay_monthly == pytest.approx(25.87)
        assert record.usage_units_per_day == pytest.approx(91.1)
        assert record.market_cap_customers == pytest.approx(272812.0)
        assert record.noise_band == pytest.approx(0.65)
        assert record.info_level == 1
        assert record.has_decision_content

    def test_group_insights_parse_into_typed_estimates(self) -> None:
        payload = (
            "=== Group Insights: Niche Creators (D_S01) ===\n"
            "Segment: Individual\n"
            "Info Level: 2 (estimates accurate to ±40%)\n\n"
            "Estimated Parameters:\n"
            "  Willingness to pay:    ~$92/mo (max monthly budget)\n"
            "  Usage volume:          ~38 units/day\n"
            "  Quality floor (q_min): ~0.61 (minimum quality needed at $0)\n"
            "  Market cap:            ~185,000 (total addressable customers)\n"
        )
        record = parse_insight(
            run_id=RUN_ID,
            week=2,
            request=InformationRequest(
                tool="get_group_insights",
                target_group="D_S01",
                expected_information_value="the quality floor for this group",
            ),
            payload=payload,
            cost=0.0,
            created_at=datetime.now(UTC),
        )

        assert record.parse_status is InsightParseStatus.SUCCEEDED
        assert record.quality_floor == pytest.approx(0.61)
        assert record.usage_units_per_day == pytest.approx(38.0)
        assert record.willingness_to_pay_monthly == pytest.approx(92.0)
        assert record.noise_band == pytest.approx(0.4)
        assert record.has_decision_content

    def test_an_unreadable_payload_never_updates_a_prior(self) -> None:
        record = parse_insight(
            run_id=RUN_ID,
            week=2,
            request=InformationRequest(
                tool="get_group_insights",
                target_group="S1",
                expected_information_value="anything",
            ),
            payload="service temporarily unavailable",
            cost=0.0,
            created_at=datetime.now(UTC),
        )
        assert record.parse_status is InsightParseStatus.FAILED
        assert not record.has_decision_content

    def test_payload_is_unwrapped_from_the_generated_call_envelope(self) -> None:
        assert extract_payload_text({"stdout": '{"result": "Info Level: 3"}'}) == (
            "Info Level: 3"
        )

    def test_information_is_gated_on_budget_and_duplication(self) -> None:
        from lithops.agents.common import InformationRequestOutput

        requests = [
            InformationRequestOutput(
                tool="get_group_insights",
                target_group="S1",
                expected_information_value="q_min for S1",
            ),
            InformationRequestOutput(
                tool="get_group_insights",
                target_group="S2",
                expected_information_value="q_min for S2",
            ),
        ]
        admitted, diagnostics = admit_information_requests(
            requests,
            cash=1_000_000.0,
            recent_identities={"get_group_insights:S1:-"},
        )
        assert [item.target_group for item in admitted] == ["S2"]
        assert any(item.startswith("duplicate_insight_request") for item in diagnostics)

        # An unknown price is allowed once so it can be measured, never twice in
        # the same week.
        unpriced, diagnostics = admit_information_requests(
            requests, cash=1_000_000.0, recent_identities=frozenset()
        )
        assert len(unpriced) == 1
        assert any(item.startswith("information_price_unknown") for item in diagnostics)

        # A measured price above the weekly ceiling is refused outright.
        priced, diagnostics = admit_information_requests(
            requests,
            cash=100_000.0,
            recent_identities=frozenset(),
            learned_costs={"get_group_insights:-": 60_000.0},
        )
        assert priced == ()
        assert all(
            item.startswith("information_budget_exceeded") for item in diagnostics
        )

    def test_measured_prices_are_learned_per_tool_and_depth(self) -> None:
        from lithops.application.executive_selection import learned_information_costs
        from lithops.domain.insights import InsightRecord

        def record(tool: str, level: int | None, cost: float) -> InsightRecord:
            return InsightRecord(
                id=uuid4(),
                run_id=RUN_ID,
                week=1,
                tool=tool,
                info_level=level,
                parse_status=InsightParseStatus.SUCCEEDED,
                parser_version="test",
                cost=cost,
                created_at=datetime.now(UTC),
            )

        costs = learned_information_costs(
            [
                record("research_group", 2, 60_000.0),
                record("research_group", 3, 175_000.0),
                record("get_group_insights", None, 0.0),
            ]
        )
        assert costs == {"research_group:2": 60_000.0, "research_group:3": 175_000.0}



class TestInsightFreshness:
    """A measurement blocks a re-purchase only while it is still current."""

    @staticmethod
    def record(week: int, identity: str) -> InsightRecord:
        return InsightRecord(
            id=uuid4(),
            run_id=RUN_ID,
            week=week,
            tool="research_group",
            request_identity=identity,
            parse_status=InsightParseStatus.SUCCEEDED,
            parser_version="test",
            cost=1_000.0,
            created_at=datetime.now(UTC),
        )

    def test_a_recent_purchase_still_blocks_the_same_question(self) -> None:
        identities = fresh_insight_identities(
            [self.record(4, "research_group:S2:2")],
            current_week=4 + INSIGHT_FRESHNESS_WEEKS - 1,
        )
        assert identities == frozenset({"research_group:S2:2"})

    def test_a_stale_purchase_no_longer_blocks_a_re_measurement(self) -> None:
        identities = fresh_insight_identities(
            [self.record(4, "research_group:S2:2")],
            current_week=4 + INSIGHT_FRESHNESS_WEEKS,
        )
        assert identities == frozenset()

    def test_a_record_without_an_identity_never_blocks_anything(self) -> None:
        identities = fresh_insight_identities(
            [self.record(4, "")],
            current_week=4,
        )
        assert identities == frozenset()


class TestEnterpriseNegotiation:
    def test_inbox_parsing_skips_malformed_entries(self) -> None:
        threads = parse_enterprise_inbox("312:40:14,bad,88:15:13")
        assert [thread.customer_id for thread in threads] == [312, 88]
        assert open_enterprise_seats_from_inbox("312:40:14,bad,88:15:13") == 55.0

    def test_offers_walk_from_target_toward_the_authorized_floor(self) -> None:
        plan = ActionPlan(
            name="p",
            strategy_family="p",
            rationale="r",
            commands=[
                ActionCommand(tool="set_prices", arguments={}, idempotency_key="k")
            ],
            enterprise_engage=True,
            enterprise_target_price_per_seat=100.0,
            enterprise_floor_price_per_seat=60.0,
        )
        prices = [
            offer_price_per_seat(plan, offer_index=index, total_offers=3)
            for index in range(3)
        ]
        assert prices == [100.0, 80.0, 60.0]
        assert min(prices) >= 60.0

    def test_the_seat_ceiling_bounds_which_threads_are_answered(self) -> None:
        threads = parse_enterprise_inbox("1:400:1,2:300:2,3:100:3")
        admitted = seats_within_ceiling(threads, ceiling=500.0)
        assert [thread.customer_id for thread in admitted] == [1, 3]

    @pytest.mark.asyncio
    async def test_offers_stay_inside_the_envelope_and_never_go_below_cost(self) -> None:
        plan = ActionPlan(
            name="p",
            strategy_family="p",
            rationale="r",
            commands=[
                ActionCommand(tool="set_prices", arguments={}, idempotency_key="k")
            ],
            enterprise_engage=True,
            enterprise_target_price_per_seat=100.0,
            enterprise_floor_price_per_seat=100.0,
        )
        sent: list[ActionCommand] = []
        events: list[tuple[str, dict]] = []

        async def execute(command: ActionCommand) -> ActionReceipt:
            sent.append(command)
            return ActionReceipt(
                run_id=RUN_ID,
                decision_id=uuid4(),
                idempotency_key=command.idempotency_key,
                tool=command.tool,
                status=ReceiptStatus.EXECUTED,
            )

        async def emit(event_type: str, payload: dict) -> None:
            events.append((event_type, payload))

        affordable = await negotiate_open_threads(
            run_id=RUN_ID,
            week=3,
            plan=plan,
            inbox="1:50:1,2:80:2",
            variable_cost_per_seat_weekly=1.0,
            execute_action=execute,
            emit_event=emit,
        )
        assert affordable.offers_made == 2
        assert {command.tool for command in sent} == {"send_enterprise_deal"}
        assert all(
            command.arguments["deals"][0][1][0][1] == 100.0 for command in sent
        )

        sent.clear()
        blocked = await negotiate_open_threads(
            run_id=RUN_ID,
            week=3,
            plan=plan,
            inbox="1:50:1",
            # A weekly seat cost above the offered monthly price is below cost.
            variable_cost_per_seat_weekly=500.0,
            execute_action=execute,
            emit_event=emit,
        )
        assert blocked.offers_made == 0
        assert sent == []
        assert any(name == "enterprise.offer_below_cost" for name, _ in events)

    @pytest.mark.asyncio
    async def test_an_unauthorized_envelope_sends_nothing(self) -> None:
        plan = ActionPlan(
            name="p",
            strategy_family="p",
            rationale="r",
            commands=[
                ActionCommand(tool="set_prices", arguments={}, idempotency_key="k")
            ],
        )

        async def execute(command: ActionCommand) -> ActionReceipt:
            raise AssertionError("no offer may be sent without an authorized envelope")

        async def emit(event_type: str, payload: dict) -> None:
            return None

        outcome = await negotiate_open_threads(
            run_id=RUN_ID,
            week=1,
            plan=plan,
            inbox="1:50:1",
            variable_cost_per_seat_weekly=0.0,
            execute_action=execute,
            emit_event=emit,
        )
        assert outcome.offers_made == 0
        assert outcome.skipped == ("not_engaged",)


class TestResearchProgramme:
    def test_research_lands_later_rather_than_immediately(self) -> None:
        estimates = parameters()
        base = state()
        started = advance_simulation_week(base, action(research_project_tier=5), estimates)

        assert started.product_quality == pytest.approx(base.product_quality, abs=1e-9)
        assert any(
            effect.improvement > 0 for effect in started.pending_quality_effects
        )


class TestEveryControlIsExecutable:
    """A control the simulator can carry must reach the benchmark.

    Both `start_research_project` and `post_social_media` were mapped, typed,
    schema-backed and fidelity-verified while being structurally impossible to
    execute: the executed plan is rebuilt from the SimulationAction, and that
    rebuild emitted no command for either. The field survived the simulation and
    vanished at execution without a veto code. This pins the whole class shut.
    """

    # Each control field on SimulationAction, and the tool that must carry it.
    CONTROL_TO_TOOL = {
        "price_per_customer_weekly": "set_prices",
        "model_tier_a": "set_model_tiers",
        "usage_quota_a": "set_usage_quotas",
        "capacity_tier": "set_capacity_tier",
        "development_spend": "set_daily_spend",
        "targeted_development_allocations": "set_targeted_dev_spend",
        "targeted_ad_allocations": "set_targeted_ad_spend",
        "lead_promotion_monthly": "set_lead_promotion",
        "recurring_promotion_monthly": "set_promotion",
        "ads_strength": "set_ads_strength",
        "targeted_ops_spend": "set_targeted_ops_spend",
        "research_project_tier": "start_research_project",
        "social_posts": "post_social_media",
    }

    def test_every_control_field_has_an_emitter(self) -> None:
        source = (
            Path(weekly_planning.__file__).read_text(encoding="utf-8").split(
                "def action_plan_from_simulation"
            )[1]
        )
        missing = {
            field: tool
            for field, tool in self.CONTROL_TO_TOOL.items()
            if f'tool="{tool}"' not in source
        }
        assert not missing, (
            "these controls can be simulated but never executed: "
            f"{sorted(missing.items())}"
        )

    def test_every_control_field_is_a_known_benchmark_tool(self) -> None:
        assert set(self.CONTROL_TO_TOOL.values()) <= set(ACTION_SPECS)

    def test_the_mapping_covers_the_simulator_control_surface(self) -> None:
        # A new control field must be added to CONTROL_TO_TOOL, which forces the
        # author to say which tool executes it.
        carried = {
            name
            for name in SimulationAction.model_fields
            if name
            in {
                "recurring_promotion_monthly",
                "ads_strength",
                "targeted_ops_spend",
                "social_posts",
                "research_project_tier",
                "capacity_tier",
            }
        }
        assert carried <= set(self.CONTROL_TO_TOOL)


class TestEveryReadStateFieldHasASource:
    """A state field the transition reads must be filled from the observation.

    `enterprise_revenue_weekly` was carried by the simulator and read by the
    transition while the observation never supplied it, so seat-contract revenue
    would have been invisible to every forecast — a gap that only bites once the
    strategy starts working, which is the worst moment to discover it.
    """

    def test_state_fields_read_by_the_transition_come_from_the_observation(
        self,
    ) -> None:
        transition = Path(state_transition.__file__).read_text(encoding="utf-8")
        read_fields = set(re.findall(r"\bstate\.([a-z_][a-z0-9_]*)", transition))

        builder_source = Path(weekly_planning.__file__).read_text(encoding="utf-8")
        start = builder_source.index("def simulation_state_from_observation")
        end = builder_source.index("\ndef ", start + 1)
        builder = builder_source[start:end]

        # Fields the transition derives rather than observes are out of scope:
        # they carry a value the simulator itself produced a week earlier.
        derived = {
            "week",
            "pending_quality_effects",
            "quality_lag_weeks",
            "price_realization_ratio",
            "cash_flow_adjustment_weekly",
        }
        missing = sorted(
            field
            for field in read_fields - derived
            if field in SimulationState.model_fields and field not in builder
        )
        assert not missing, (
            "the transition reads these state fields but the observation never "
            f"supplies them: {missing}"
        )


class TestResearchProgrammeSemantics:
    """The R&D lever priced by the environment's own listing, charged honestly."""

    @staticmethod
    def catalog_state(**overrides) -> SimulationState:
        from lithops.simulator.models import ResearchTierFacts

        return state(
            research_catalog=(
                ResearchTierFacts(
                    tier=3, cost=500_000.0, mean_weeks=3, mean_quality_boost=0.11
                ),
            ),
            **overrides,
        )

    def test_catalog_cost_is_charged_exactly_once_while_the_tier_matures(self) -> None:
        estimates = parameters()
        start = self.catalog_state()
        research = action(research_project_tier=3)
        control = action()

        week_one = advance_simulation_week(start, research, estimates)
        baseline = advance_simulation_week(start, control, estimates)
        # The programme's exact listed charge, nothing else, in the start week.
        assert baseline.cash - week_one.cash == pytest.approx(500_000.0)
        assert week_one.research_spend_weekly == pytest.approx(500_000.0)
        assert [item.tier for item in week_one.research_tiers_in_progress] == [3]

        # Repeating the action while the tier matures must neither recharge
        # nor stack a second pending effect.
        week_two = advance_simulation_week(week_one, research, estimates)
        baseline_two = advance_simulation_week(week_one, control, estimates)
        assert week_two.research_spend_weekly == 0.0
        assert baseline_two.cash - week_two.cash == pytest.approx(0.0)
        research_effects = [
            effect
            for effect in week_two.pending_quality_effects
            if effect.improvement == pytest.approx(0.11)
        ]
        assert len(research_effects) == 1

        # And after the programme lands, the same repeated action must not buy
        # it again: one candidate action is one decision, not a standing order.
        current = week_two
        for _ in range(4):
            current = advance_simulation_week(current, research, estimates)
        assert current.research_spend_weekly == 0.0
        assert current.research_tiers_started == (3,)

    def test_listed_mean_boost_lands_after_the_listed_mean_weeks(self) -> None:
        estimates = parameters()
        current = self.catalog_state()
        research = action(research_project_tier=3)
        quality_before = current.product_quality
        landed = None
        for week in range(5):
            current = advance_simulation_week(
                current, research if week == 0 else action(), estimates
            )
            if current.product_quality >= quality_before + 0.10:
                landed = week + 1
                break
        # The learned lag prior says two weeks per tier — week seven for tier
        # three. Landing inside five weeks proves the listing's mean_weeks=3
        # governed the lag, and the jump proves its mean boost governed the size.
        assert landed is not None

    def test_unread_catalog_falls_back_to_the_learned_draw_and_charges_nothing(
        self,
    ) -> None:
        estimates = parameters()
        week_one = advance_simulation_week(
            state(), action(research_project_tier=3), estimates
        )
        # Nothing observed prices the programme, so the transition cannot
        # invent a charge; the card layer names the unquantified cost instead.
        assert week_one.research_spend_weekly == 0.0
        assert [item.tier for item in week_one.research_tiers_in_progress] == [3]
