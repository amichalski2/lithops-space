from uuid import uuid4

import pytest
from lithops.application.weekly_planning import (
    INITIAL_WEEKLY_AVERAGE_PRICE,
    _active_experiment_plan,
    _executable_candidates,
    _experiment_reversion_action,
    _remaining_horizon_days,
    _stage_experiment_plan,
    _targeted_development_scale_variants,
    action_plan_from_simulation,
    exploration_memory_from_decisions,
    sandbox_action_payload,
    simulation_action_from_action_plan,
    simulation_state_from_observation,
)
from lithops.domain.models import (
    ActionCommand,
    ActionPlan,
    CandidateEvaluationRecord,
    DecisionRecord,
    ExperimentProgram,
    ObservationSnapshot,
    RunRecord,
)
from lithops.simulator import SimulationAction, TargetedDevelopmentAllocation
from lithops.simulator.invariants import evaluate_simulation_action
from lithops.simulator.strategy_search import (
    ExplorationMemory,
    FunnelRegimeEvidence,
    generate_candidate_actions,
)


def test_terminal_selection_horizon_tracks_remaining_benchmark_days() -> None:
    run = RunRecord(horizon_days=504)

    assert _remaining_horizon_days(run, ObservationSnapshot(day=0, cash=1)) == 504
    assert _remaining_horizon_days(run, ObservationSnapshot(day=266, cash=1)) == 238
    assert _remaining_horizon_days(run, ObservationSnapshot(day=497, cash=1)) == 7


def test_weekly_price_is_converted_to_ceobench_monthly_price_and_ads_are_targeted() -> None:
    run = RunRecord(id=uuid4())
    observation = ObservationSnapshot(
        day=14,
        cash=1_000_000,
        metrics={
            "known_segments": "S1,E1",
            "operations_spend": 4200,
            "price_a": 60,
            "price_b": 60,
            "price_c": 60,
        },
    )
    selected = SimulationAction(
        name="balanced_growth",
        price_per_customer_weekly=14,
        marketing_spend=700,
        development_spend=350,
    )

    plan = action_plan_from_simulation(run, observation, selected, "robust")

    assert plan.commands[0].arguments == {"A": 60, "B": 60, "C": 60}
    assert plan.commands[1].arguments == {"A": 1, "B": 1, "C": 1}
    assert plan.commands[2].arguments == {"operations": 600, "development": 50}
    assert plan.commands[3].tool == "set_targeted_ad_spend"
    allocation = plan.commands[3].arguments["targeted_spend"]
    assert allocation == {"search_ads": {"S1": 50}, "linkedin": {"E1": 50}}
    assert sum(value for groups in allocation.values() for value in groups.values()) == 100


def test_executive_targeting_survives_simulation_and_execution_exactly() -> None:
    run = RunRecord(id=uuid4())
    observation = ObservationSnapshot(
        day=7,
        cash=1_000_000,
        metrics={
            "known_segments": "S1,S2,E1",
            "price_a": 25,
            "price_b": 69,
            "price_c": 179,
        },
    )
    proposed = ActionPlan(
        name="linkedin_s2_probe",
        strategy_family="executive_experiment_marketing_linkedin_s2_probe",
        rationale="Isolate one channel and one segment.",
        commands=[
            ActionCommand(
                tool="set_targeted_ad_spend",
                arguments={"targeted_spend": {"linkedin": {"S2": 125.0}}},
                idempotency_key="proposal:targeting",
            )
        ],
    )
    state = simulation_state_from_observation(observation)

    simulated = simulation_action_from_action_plan(proposed, state)
    executed = action_plan_from_simulation(run, observation, simulated, "selected")

    assert simulated.marketing_spend == 875
    assert [item.model_dump() for item in simulated.targeted_ad_allocations] == [
        {"channel": "linkedin", "segment": "S2", "daily_spend": 125.0}
    ]
    assert executed.commands[3].arguments["targeted_spend"] == {
        "linkedin": {"S2": 125.0}
    }


def test_invalid_observed_segments_fall_back_to_public_initial_segment() -> None:
    plan = action_plan_from_simulation(
        RunRecord(),
        ObservationSnapshot(day=0, cash=1, metrics={"known_segments": "bad"}),
        SimulationAction(
            name="test",
            price_per_customer_weekly=7,
            marketing_spend=70,
            development_spend=0,
        ),
        "test",
    )

    assert plan.commands[3].arguments["targeted_spend"] == {
        "search_ads": {"S1": pytest.approx(10)}
    }


def test_zero_initial_table_price_uses_explicit_business_prior() -> None:
    state = simulation_state_from_observation(
        ObservationSnapshot(
            day=0,
            cash=1_000_000,
            metrics={
                "active_customers": 0,
                "weekly_revenue": 0,
                "price_per_customer_weekly": 0,
            },
        )
    )

    assert state.price_per_customer_weekly == pytest.approx(INITIAL_WEEKLY_AVERAGE_PRICE)
    assert state.price_per_customer_weekly > 20


def test_observed_compute_cost_drives_unit_margin_instead_of_arbitrary_default() -> None:
    observation = ObservationSnapshot(
        day=7,
        cash=993_906.41,
        metrics={
            "active_customers": 2,
            "weekly_revenue": 11.6666666667,
            "price_per_customer_weekly": 5.8333333333,
            "operating_cost_per_customer_weekly": 0.04425,
            "capacity_spend_weekly": 595,
            "marketing_spend": 1_000,
            "development_spend": 1_000,
            "operations_spend": 3_500,
        },
    )

    state = simulation_state_from_observation(observation)
    candidates, _ = _executable_candidates(
        state,
        action_plan_from_simulation(
            RunRecord(),
            observation,
            SimulationAction(
                name="executive",
                price_per_customer_weekly=state.price_per_customer_weekly,
                marketing_spend=state.marketing_spend,
                development_spend=state.development_spend,
                operations_spend=state.operations_spend,
            ),
            "test",
        ),
    )

    assert state.operating_cost_per_customer_weekly == pytest.approx(0.04425)
    assert state.capacity_spend_weekly == pytest.approx(595)
    assert len([item for item in candidates if evaluate_simulation_action(state, item).valid]) >= 3


def test_executable_pool_has_real_continuation_and_multiple_executive_proposals() -> None:
    observation = ObservationSnapshot(
        day=14,
        cash=900_000,
        metrics={
            "active_customers": 4,
            "weekly_revenue": 40,
            "price_a": 25,
            "price_b": 69,
            "price_c": 179,
            "marketing_spend": 700,
            "development_spend": 350,
            "operations_spend": 2_800,
            "model_tier_a": 1,
            "model_tier_b": 1,
            "model_tier_c": 1,
        },
    )
    state = simulation_state_from_observation(observation)
    first = action_plan_from_simulation(
        RunRecord(),
        observation,
        SimulationAction(
            name="first",
            price_per_customer_weekly=(
                state.effective_catalog_price_per_customer_weekly * 1.05
            ),
            marketing_spend=1_400,
            development_spend=350,
            operations_spend=2_800,
        ),
        "first hypothesis",
    ).model_copy(update={"strategy_family": "executive_growth_0"})
    second = action_plan_from_simulation(
        RunRecord(),
        observation,
        SimulationAction(
            name="second",
            price_per_customer_weekly=state.effective_catalog_price_per_customer_weekly,
            marketing_spend=350,
            development_spend=700,
            operations_spend=2_800,
        ),
        "second hypothesis",
    ).model_copy(update={"strategy_family": "executive_product_quality_1"})

    candidates, _ = _executable_candidates(state, (first, second))
    by_name = {candidate.name: candidate for candidate in candidates}

    assert "continuation" in by_name
    continuation = by_name["continuation"]
    assert continuation.marketing_spend == state.marketing_spend
    assert continuation.development_spend == state.development_spend
    assert continuation.price_per_customer_weekly == pytest.approx(
        state.effective_catalog_price_per_customer_weekly
    )
    assert "executive_growth_0" in by_name
    assert "executive_product_quality_1" in by_name
    assert "aggressive_growth" not in by_name
    assert "cash_preservation" not in by_name
    assert "pricing_efficiency" not in by_name
    signatures = [
        candidate.model_dump_json(exclude={"name"}) for candidate in candidates
    ]
    assert len(signatures) == len(set(signatures))


def test_executive_v2_keeps_the_epistemically_admitted_quality_candidate() -> None:
    observation = ObservationSnapshot(
        day=28,
        cash=983_000,
        metrics={
            "active_customers": 0,
            "weekly_revenue": 0,
            "weekly_leads": 70,
            "weekly_conversions": 0,
            "product_quality": 0.208,
            "marketing_spend": 0,
            "development_spend": 0,
            "operations_spend": 594,
            "price_a": 25,
            "price_b": 69,
            "price_c": 179,
        },
    )
    state = simulation_state_from_observation(observation)

    def executive(name: str) -> ActionPlan:
        return action_plan_from_simulation(
            RunRecord(),
            observation,
            SimulationAction(
                name=name,
                price_per_customer_weekly=(
                    state.effective_catalog_price_per_customer_weekly
                ),
                marketing_spend=0,
                development_spend=0,
                operations_spend=state.operations_spend,
            ),
            name,
        ).model_copy(update={"strategy_family": f"executive_{name}"})

    candidates, _ = _executable_candidates(
        state,
        (executive("continuation_a"), executive("continuation_b")),
        ExplorationMemory(
            attempted_strategies=(
                "controlled_exploration_marketing_q2",
                "controlled_exploration_lead_promotion_q2",
            ),
            funnel_regime_evidence=(
                FunnelRegimeEvidence(
                    quality_band=2,
                    leads=283,
                    conversions=0,
                    weeks=1,
                ),
            )
        ),
    )

    names = {candidate.name for candidate in candidates}
    assert "controlled_exploration_development_q2" in names
    assert "aggressive_growth" not in names


def test_realized_arpu_does_not_reanchor_catalog_price_candidates() -> None:
    """Regression for the live 25 -> 7.21 -> 2.38 -> 1.21 catalog collapse."""

    observation = ObservationSnapshot(
        day=35,
        cash=960_000,
        metrics={
            "active_customers": 15,
            "weekly_revenue": 19.8,
            "price_per_customer_weekly": 1.32,
            "price_a": 25,
            "price_b": 69,
            "price_c": 179,
            "operating_cost_per_customer_weekly": 20.92,
        },
    )

    state = simulation_state_from_observation(observation)
    candidates = generate_candidate_actions(state)
    continuation = candidates[0]
    plan = action_plan_from_simulation(RunRecord(), observation, continuation, "test")

    assert state.price_per_customer_weekly == pytest.approx(1.32)
    assert state.catalog_price_per_customer_weekly == pytest.approx(91 * 7 / 30)
    assert continuation.price_per_customer_weekly == pytest.approx(91 * 7 / 30)
    assert plan.commands[0].arguments == {"A": 25, "B": 69, "C": 179}


def test_inherited_arpu_breach_blocks_catalog_cut_but_allows_recovery() -> None:
    state = simulation_state_from_observation(
        ObservationSnapshot(
            day=35,
            cash=960_000,
            metrics={
                "active_customers": 15,
                "weekly_revenue": 19.8,
                "price_per_customer_weekly": 1.32,
                "price_a": 25,
                "price_b": 69,
                "price_c": 179,
                "operating_cost_per_customer_weekly": 20.92,
            },
        )
    )
    catalog = state.effective_catalog_price_per_customer_weekly
    cut = SimulationAction(
        name="cut",
        price_per_customer_weekly=catalog * 0.95,
        marketing_spend=0,
        development_spend=0,
    )
    recovery = cut.model_copy(
        update={"name": "recovery", "price_per_customer_weekly": catalog * 1.10}
    )

    cut_report = evaluate_simulation_action(state, cut)
    recovery_report = evaluate_simulation_action(state, recovery)

    assert not cut_report.valid
    assert recovery_report.valid


def test_ceobench_plan_executes_only_controls_present_in_the_simulated_action() -> None:
    plan = action_plan_from_simulation(
        RunRecord(),
        ObservationSnapshot(
            day=14,
            cash=1_000_000,
            metrics={
                "source": "ceobench_public_cli",
                "enterprise_inbox": "312:40:14,bad,88:15:13,312:40:14",
                "price_a": 25,
                "price_b": 69,
                "price_c": 179,
            },
        ),
        SimulationAction(
            name="test",
            price_per_customer_weekly=21,
            marketing_spend=70,
            development_spend=0,
        ),
        "test",
    )

    assert [command.tool for command in plan.commands] == [
        "set_prices",
        "set_model_tiers",
        "set_daily_spend",
        "set_targeted_ad_spend",
        "set_targeted_dev_spend",
    ]


def test_zero_customer_history_adds_one_reversible_exploration_candidate() -> None:
    observation = ObservationSnapshot(
        day=14,
        cash=974_236.5,
        metrics={
            "source": "ceobench_public_cli",
            "active_customers": 0,
            "weekly_conversions": 0,
            "marketing_spend": 0,
            "development_spend": 0,
            "product_quality": 0.2,
        },
    )
    state = simulation_state_from_observation(observation)
    executive_plan = action_plan_from_simulation(
        RunRecord(),
        observation,
        SimulationAction(
            name="executive_proposal",
            price_per_customer_weekly=21,
            marketing_spend=0,
            development_spend=0,
        ),
        "test",
    )

    candidates, _ = _executable_candidates(state, executive_plan)

    experiments = [
        candidate
        for candidate in candidates
        if candidate.name == "controlled_exploration_marketing_q2"
    ]
    assert len(experiments) == 1
    selected = experiments[0]
    assert selected.marketing_spend > state.marketing_spend
    assert selected.development_spend == state.development_spend
    assert selected.model_tier_a == state.model_tier_a
    assert selected.model_tier_b == state.model_tier_b
    assert selected.model_tier_c == state.model_tier_c
    assert selected.marketing_spend_until_week == state.week + 1
    assert selected.marketing_spend_after_experiment == state.marketing_spend
    assert any(candidate.marketing_spend == 0 for candidate in candidates)
    plan = action_plan_from_simulation(RunRecord(), observation, selected, "test")
    by_tool = {command.tool: command for command in plan.commands}
    assert by_tool["set_daily_spend"].arguments["development"] == pytest.approx(
        selected.development_spend / 7
    )
    assert by_tool["set_targeted_ad_spend"].arguments == {
        "targeted_spend": {"search_ads": {"S1": selected.marketing_spend / 7}}
    }
    assert by_tool["set_model_tiers"].arguments == {"A": 1, "B": 1, "C": 1}
    # Unchanged controls are not restated: the benchmark carries them forward.
    assert "set_usage_quotas" not in by_tool
    assert "set_capacity_tier" not in by_tool
    assert "set_lead_promotion" not in by_tool
    assert "send_enterprise_deal" not in by_tool


def test_committed_experiment_memory_reverts_only_the_following_week() -> None:
    observation = ObservationSnapshot(day=21, cash=900_000)
    selected = SimulationAction(
        name="controlled_exploration_marketing_q5",
        price_per_customer_weekly=21,
        marketing_spend=5_000,
        development_spend=2_000,
        marketing_spend_until_week=4,
        development_spend_until_week=4,
        marketing_spend_after_experiment=4_000,
        development_spend_after_experiment=2_000,
        lead_promotion_monthly=5,
        lead_promotion_until_week=4,
        lead_promotion_after_experiment=0,
    )
    plan = action_plan_from_simulation(
        RunRecord(), observation, selected, "bounded experiment"
    )
    evaluation = CandidateEvaluationRecord(
        strategy=selected.name,
        action_parameters=selected.model_dump(mode="json"),
        expected_ending_cash=850_000,
        downside_ending_cash=800_000,
        bankruptcy_probability=0,
        expected_customer_growth=1,
        robustness="high",
        robust_utility=-0.1,
        rollout_count=10,
    )
    decision = DecisionRecord.model_construct(
        id=uuid4(),
        week=3,
        action_plan=plan,
        candidate_evaluations=[evaluation],
        actual_outcome=ObservationSnapshot(day=28, cash=895_000),
    )

    immediate = exploration_memory_from_decisions((decision,), current_week=4)
    later = exploration_memory_from_decisions((decision,), current_week=5)

    assert immediate.attempted_strategies == (selected.name,)
    assert immediate.revert_marketing_spend == 4_000
    assert immediate.revert_development_spend == 2_000
    assert immediate.revert_lead_promotion_monthly == 0
    assert later.attempted_strategies == (selected.name,)
    assert later.revert_marketing_spend is None
    assert later.revert_development_spend is None
    assert later.revert_lead_promotion_monthly is None


@pytest.mark.parametrize("control", ["price", "tier"])
def test_price_and_tier_experiments_revert_to_persisted_baseline(control: str) -> None:
    program = ExperimentProgram(
        commitment_id=f"{control}-revert-0",
        control=control,
        protocol_version="experiment-program-v2",
        started_week=0,
        minimum_maturity_week=1,
        maximum_end_week=1,
        baseline_value=1.0,
        treatment_value=0.8 if control == "price" else 5.0,
        maximum_cumulative_downside=5_000,
        expected_observation="matched cohort response",
        falsification_condition="no matched response",
        target_segment="S1",
        target_channel="search_ads",
        baseline_configuration={
            "prices": {"A": 25.0, "B": 69.0, "C": 179.0},
            "model_tiers": {"A": 1, "B": 2, "C": 3},
            "weekly_marketing_spend": 3_500.0,
        },
        treatment_configuration={
            "prices": {"A": 20.0, "B": 55.2, "C": 143.2},
            "model_tiers": {"A": 5, "B": 5, "C": 5},
            "weekly_marketing_spend": 3_500.0,
        },
        measurement_plan=(
            {
                "source": "configuration",
                "metric": control,
                "target_segment": "S1",
                "target_channel": "search_ads",
            },
            {
                "source": "cohort",
                "metric": "conversion_rate",
                "target_segment": "S1",
                "target_channel": "search_ads",
                "minimum_exposure": 30,
            },
        ),
    )
    plan = ActionPlan(
        name=f"{control} experiment",
        strategy_family=f"executive_experiment_{control}_h1",
        rationale="bounded treatment",
        commands=[
            ActionCommand(
                tool="set_prices",
                arguments={"A": 20.0, "B": 55.2, "C": 143.2},
                idempotency_key=f"{control}-prices",
            )
        ],
        proposal_kind="experiment",
        hypothesis_id="h_revert",
        experiment_control=control,
        evidence_regime="test",
        experiment_expires_week=1,
        experiment_program=program,
    )
    committed = DecisionRecord.model_construct(
        id=uuid4(),
        week=0,
        action_plan=plan,
        actual_outcome=ObservationSnapshot(day=7, cash=990_000),
    )
    state = simulation_state_from_observation(
        ObservationSnapshot(
            day=7,
            cash=990_000,
            metrics={
                "price_a": 20.0,
                "price_b": 55.2,
                "price_c": 143.2,
                "model_tier_a": 5,
                "model_tier_b": 5,
                "model_tier_c": 5,
            },
        )
    )

    reverted = _experiment_reversion_action((committed,), state=state)

    assert reverted is not None
    if control == "price":
        assert reverted.price_per_customer_weekly == pytest.approx(
            (25.0 + 69.0 + 179.0) / 3.0 * 7.0 / 30.0
        )
    else:
        assert (reverted.model_tier_a, reverted.model_tier_b, reverted.model_tier_c) == (
            1,
            2,
            3,
        )


def test_legacy_executive_experiment_defaults_to_one_week_and_is_not_repeated() -> None:
    observation = ObservationSnapshot(
        day=21,
        cash=900_000,
        metrics={
            "price_a": 25,
            "price_b": 69,
            "price_c": 179,
            "marketing_spend": 700,
            "development_spend": 350,
            "operations_spend": 2_800,
        },
    )
    state = simulation_state_from_observation(observation)
    strategy = "executive_experiment_development_quality_lag_probe"
    plan = action_plan_from_simulation(
        RunRecord(),
        observation,
        SimulationAction(
            name=strategy,
            price_per_customer_weekly=state.effective_catalog_price_per_customer_weekly,
            marketing_spend=state.marketing_spend,
            development_spend=2_100,
            operations_spend=state.operations_spend,
        ),
        "test a development lag",
    ).model_copy(update={"strategy_family": strategy})

    candidate = next(
        item
        for item in _executable_candidates(state, (plan,))[0]
        if item.name == strategy
    )
    assert candidate.marketing_spend_until_week is None
    assert candidate.development_spend_until_week == state.week + 1
    assert candidate.marketing_spend_after_experiment is None
    assert candidate.development_spend_after_experiment == state.development_spend

    memory = exploration_memory_from_decisions(
        (
            DecisionRecord.model_construct(
                id=uuid4(),
                week=3,
                action_plan=plan,
                candidate_evaluations=[],
                actual_outcome=ObservationSnapshot(day=28, cash=899_000),
            ),
        ),
        current_week=4,
    )
    assert memory.attempted_strategies == (strategy,)
    repeated, _ = _executable_candidates(state, (plan,), memory)
    assert all(item.name != strategy for item in repeated)


def test_delayed_experiment_remains_binding_then_reverts_to_original_control() -> None:
    observation = ObservationSnapshot(
        day=21,
        cash=900_000,
        metrics={"development_spend": 350, "operations_spend": 2_800},
    )
    state = simulation_state_from_observation(observation)
    program = ExperimentProgram(
        commitment_id="quality-support-q2-3",
        control="development",
        started_week=3,
        minimum_maturity_week=8,
        maximum_end_week=8,
        baseline_value=350,
        treatment_value=5_350,
        maximum_cumulative_downside=30_000,
        expected_observation="Quality enters a previously unsupported band.",
        falsification_condition="Quality support is unchanged at maturity.",
    )
    strategy = "executive_experiment_development_quality_support"
    plan = action_plan_from_simulation(
        RunRecord(),
        observation,
        SimulationAction(
            name=strategy,
            price_per_customer_weekly=state.effective_catalog_price_per_customer_weekly,
            marketing_spend=state.marketing_spend,
            development_spend=5_350,
            operations_spend=state.operations_spend,
        ),
        "bounded delayed intervention",
    ).model_copy(
        update={
            "strategy_family": strategy,
            "proposal_kind": "experiment",
            "hypothesis_id": "quality_support",
            "experiment_control": "development",
            "evidence_regime": "quality_band_2",
            "experiment_expires_week": 8,
            "experiment_program": program,
        }
    )
    decision = DecisionRecord.model_construct(
        id=uuid4(),
        week=3,
        action_plan=plan,
        candidate_evaluations=[],
        actual_outcome=ObservationSnapshot(day=28, cash=895_000),
    )

    assert _active_experiment_plan((decision,), current_week=4) == plan
    continued, _ = _executable_candidates(
        state.model_copy(update={"week": 4}),
        (plan,),
        active_commitment=True,
    )
    assert len(continued) == 1
    assert continued[0].development_spend_until_week == 8
    assert continued[0].development_spend_after_experiment == 350

    reverted = _experiment_reversion_action(
        (decision,),
        state=state.model_copy(update={"week": 8, "development_spend": 5_350}),
    )
    assert reverted is not None
    assert reverted.development_spend == 350


def test_targeted_development_program_builds_then_runs_one_acquisition_probe() -> None:
    observation = ObservationSnapshot(
        day=14,
        cash=1_000_000,
        metrics={"marketing_spend": 0, "development_spend": 0},
    )
    state = simulation_state_from_observation(observation)
    program = ExperimentProgram(
        commitment_id="quality-frontier-s2-2",
        control="targeted_development",
        started_week=2,
        minimum_maturity_week=5,
        maximum_end_week=6,
        baseline_value=0,
        treatment_value=14_000,
        maximum_cumulative_downside=150_000,
        expected_observation="Quality changes before a fresh acquisition cohort is exposed.",
        falsification_condition="The changed support produces no quality movement.",
        target_segment="S2",
        target_channel="social_media",
        acquisition_probe_weekly_spend=7_000,
        treatment_targeted_development={"S2": 2_000},
    )
    plan = ActionPlan(
        name="Build S2 quality",
        strategy_family="executive_experiment_targeted_development_quality_frontier",
        rationale="Create new quality support before measuring acquisition.",
        proposal_kind="experiment",
        hypothesis_id="quality_frontier",
        experiment_control="targeted_development",
        evidence_regime="leads_gte100:quality_1:customers_zero",
        experiment_expires_week=6,
        experiment_program=program,
        commands=[
            ActionCommand(
                tool="set_targeted_dev_spend",
                arguments={"targeted_spend": {"S2": 2_000}},
                idempotency_key="quality-dev",
            ),
            ActionCommand(
                tool="set_targeted_ad_spend",
                arguments={"targeted_spend": {}},
                idempotency_key="quality-ads",
            ),
        ],
    )
    decision = DecisionRecord.model_construct(
        id=uuid4(),
        week=2,
        action_plan=plan,
        candidate_evaluations=[],
        actual_outcome=ObservationSnapshot(day=21, cash=986_000),
    )

    build = _active_experiment_plan((decision,), current_week=4)
    assert build is not None
    build_tools = {item.tool: item for item in build.commands}
    assert build_tools["set_targeted_dev_spend"].arguments["targeted_spend"] == {
        "S2": 2_000
    }
    assert build_tools["set_targeted_ad_spend"].arguments["targeted_spend"] == {}

    probe = _active_experiment_plan((decision,), current_week=5)
    assert probe is not None
    probe_tools = {item.tool: item for item in probe.commands}
    assert probe_tools["set_targeted_dev_spend"].arguments["targeted_spend"] == {}
    assert probe_tools["set_targeted_ad_spend"].arguments["targeted_spend"] == {
        "social_media": {"S2": 1_000.0}
    }

    reverted = _experiment_reversion_action(
        (decision,), state=state.model_copy(update={"week": 6})
    )
    assert reverted is not None
    assert reverted.targeted_development_allocations == ()
    assert reverted.targeted_ad_allocations == ()

    # The same staging must be applied to a newly selected Executive plan, not
    # only when the commitment is loaded from next week's decision history.
    initially_misstaged = plan.model_copy(
        update={
            "commands": [
                plan.commands[0],
                plan.commands[1].model_copy(
                    update={
                        "arguments": {
                            "targeted_spend": {"social_media": {"S2": 1_000.0}}
                        }
                    }
                ),
            ]
        },
        deep=True,
    )
    staged = _stage_experiment_plan(initially_misstaged, current_week=2)
    staged_tools = {item.tool: item for item in staged.commands}
    assert staged_tools["set_targeted_ad_spend"].arguments["targeted_spend"] == {}

    variants = _targeted_development_scale_variants((plan,))
    assert len(variants) == 2
    assert {
        next(iter(item.experiment_program.treatment_targeted_development.values()))
        for item in variants
        if item.experiment_program is not None
    } == {2_000.0, 150_000.0 / 21.0}

    payload = sandbox_action_payload(
        SimulationAction(
            name="staged_support",
            price_per_customer_weekly=10,
            marketing_spend=7_000,
            development_spend=0,
            targeted_development_allocations=(
                TargetedDevelopmentAllocation(segment="S2", daily_spend=2_000),
            ),
            targeted_development_spend_until_week=5,
            targeted_development_spend_after_experiment=0,
            marketing_spend_start_week=5,
            marketing_spend_until_week=6,
            marketing_spend_after_experiment=0,
        ),
        state,
        horizon_weeks=8,
    )
    assert payload["targeted_development_spend_weekly"] == 14_000
    assert payload["targeted_development_duration_weeks"] == 3
    assert payload["targeted_development_spend_after_experiment"] == 0
    assert payload["marketing_spend_start_after_weeks"] == 3


def test_controlled_development_builds_then_probes_the_mature_quality_regime() -> None:
    observation = ObservationSnapshot(
        day=63,
        cash=800_000,
        metrics={
            "known_segments": "S1,E1",
            "marketing_spend": 0,
            "development_spend": 0,
        },
    )
    state = simulation_state_from_observation(observation)
    selected = SimulationAction(
        name="controlled_exploration_development_q3",
        price_per_customer_weekly=state.effective_catalog_price_per_customer_weekly,
        marketing_spend=0,
        development_spend=3_500,
        operations_spend=state.operations_spend,
        development_spend_until_week=14,
        development_spend_after_experiment=0,
    )

    plan = action_plan_from_simulation(
        RunRecord(), observation, selected, "quality support remains decision relevant"
    )
    program = plan.experiment_program
    assert program is not None
    assert program.control == "development"
    assert program.minimum_maturity_week == 14
    assert program.maximum_end_week == 15
    assert program.target_segment == "S1"
    assert program.target_channel == "search_ads"
    assert program.acquisition_probe_weekly_spend == 2_000

    build = _stage_experiment_plan(plan, current_week=12)
    build_tools = {item.tool: item for item in build.commands}
    assert build_tools["set_daily_spend"].arguments["development"] == 500
    assert build_tools["set_targeted_ad_spend"].arguments["targeted_spend"] == {}

    probe = _stage_experiment_plan(plan, current_week=14)
    probe_tools = {item.tool: item for item in probe.commands}
    assert probe_tools["set_daily_spend"].arguments["development"] == 0
    assert probe_tools["set_targeted_ad_spend"].arguments["targeted_spend"] == {
        "search_ads": {"S1": pytest.approx(2_000 / 7)}
    }

    candidates, _ = _executable_candidates(state, (plan,), active_commitment=True)
    candidate = next(item for item in candidates if item.name == selected.name)
    assert candidate.development_spend_until_week == 14
    assert candidate.marketing_spend_start_week == 14
    assert candidate.marketing_spend_until_week == 15
    payload = sandbox_action_payload(candidate, state, horizon_weeks=12)
    assert payload["development_spend_duration_weeks"] == 5
    assert payload["experiment_duration_weeks"] == 6
    assert payload["marketing_spend_start_after_weeks"] == 5

    decision = DecisionRecord.model_construct(
        id=uuid4(),
        week=9,
        action_plan=plan,
        candidate_evaluations=[],
        actual_outcome=ObservationSnapshot(day=70, cash=796_500),
    )
    active_probe = _active_experiment_plan((decision,), current_week=14)
    assert active_probe is not None
    probe_observation = observation.model_copy(update={"day": 98, "cash": 779_000})
    probe_state = simulation_state_from_observation(probe_observation)
    probe_action = simulation_action_from_action_plan(
        active_probe,
        probe_state,
        name=active_probe.strategy_family,
    )
    reconstructed = action_plan_from_simulation(
        RunRecord(),
        probe_observation,
        probe_action,
        "continue the already committed probe",
        proposal_lineage=active_probe,
    )
    assert reconstructed.experiment_program is not None
    assert reconstructed.experiment_program.started_week == 9
    assert reconstructed.experiment_program.minimum_maturity_week == 14
    assert reconstructed.experiment_program.maximum_end_week == 15

    reverted = _experiment_reversion_action(
        (decision,), state=state.model_copy(update={"week": 15})
    )
    assert reverted is not None
    assert reverted.development_spend == 0
    assert reverted.targeted_ad_allocations == ()


def test_structured_experiment_memory_allows_same_hypothesis_only_in_new_evidence_regime() -> None:
    observation = ObservationSnapshot(
        day=21,
        cash=900_000,
        metrics={
            "active_customers": 0,
            "price_a": 25,
            "price_b": 69,
            "price_c": 179,
            "marketing_spend": 0,
            "development_spend": 350,
            "operations_spend": 2_800,
        },
    )
    state = simulation_state_from_observation(observation)
    strategy = "executive_experiment_development_quality_threshold"
    base = action_plan_from_simulation(
        RunRecord(),
        observation,
        SimulationAction(
            name=strategy,
            price_per_customer_weekly=state.effective_catalog_price_per_customer_weekly,
            marketing_spend=0,
            development_spend=2_100,
            operations_spend=state.operations_spend,
        ),
        "test quality threshold",
    ).model_copy(
        update={
            "strategy_family": strategy,
            "proposal_kind": "experiment",
            "hypothesis_id": "quality_threshold",
            "experiment_control": "development",
            "evidence_regime": "leads_lt100:quality_1:customers_zero",
            "experiment_expires_week": 4,
        }
    )
    memory = exploration_memory_from_decisions(
        (
            DecisionRecord.model_construct(
                id=uuid4(),
                week=3,
                action_plan=base,
                candidate_evaluations=[],
                actual_outcome=ObservationSnapshot(day=28, cash=899_000),
            ),
        ),
        current_week=4,
    )

    assert all(
        item.name != strategy
        for item in _executable_candidates(state, (base,), memory)[0]
    )
    changed_regime = base.model_copy(
        update={
            "evidence_regime": "leads_gte100:quality_3:customers_zero",
            "experiment_expires_week": 5,
        }
    )
    assert any(
        item.name == strategy
        for item in _executable_candidates(state, (changed_regime,), memory)[0]
    )


def test_lead_promotion_is_an_explicit_reversible_action_command() -> None:
    observation = ObservationSnapshot(
        day=7,
        cash=990_000,
        metrics={
            "active_customers": 0,
            "weekly_acquisition": 0,
            "weekly_leads": 20,
            "total_leads": 20,
            "total_conversions": 0,
            "price_a": 25,
            "price_b": 69,
            "price_c": 179,
            "lead_promotion_monthly": 0,
        },
    )
    state = simulation_state_from_observation(observation)
    promotion = next(
        action
        for action in generate_candidate_actions(state)
        if action.name == "controlled_exploration_lead_promotion_q5"
    )

    plan = action_plan_from_simulation(
        RunRecord(), observation, promotion, "decision-sensitive experiment"
    )
    command = next(
        command for command in plan.commands if command.tool == "set_lead_promotion"
    )
    restored = simulation_action_from_action_plan(plan, state)

    assert command.arguments == {"global_promotion": 5.0}
    assert restored.lead_promotion_monthly == 5.0
    assert promotion.lead_promotion_until_week == state.week + 1
    assert promotion.lead_promotion_after_experiment == 0
