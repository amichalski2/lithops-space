from datetime import UTC, datetime
from uuid import UUID

import pytest
from lithops.domain.models import ObservationSnapshot
from lithops.simulator.models import (
    RolloutOutcome,
    SimulationAction,
    SimulationState,
    TargetedDevelopmentAllocation,
)
from lithops.simulator.strategy_search import (
    MAX_ACCEPTABLE_BANKRUPTCY_PROBABILITY,
    CandidateSimulation,
    ExplorationMemory,
    FunnelRegimeEvidence,
    NoViableStrategyError,
    RobustnessLevel,
    assess_controlled_exploration,
    generate_candidate_actions,
    search_strategies,
    select_robust_strategy,
    summarize_candidate,
)
from lithops.world_model import bootstrap_world_model

RUN_ID = UUID("33333333-3333-3333-3333-333333333333")


def state(*, cash: float = 500_000) -> SimulationState:
    return SimulationState(
        cash=cash,
        revenue_weekly=50_000,
        customers=1_000,
        churn_rate=0.04,
        price_per_customer_weekly=50,
        weekly_acquisition=80,
        marketing_spend=10_000,
        development_spend=5_000,
        product_quality=0.6,
        capacity=5_000,
        reputation=0.6,
    )


def model():
    return bootstrap_world_model(
        RUN_ID,
        ObservationSnapshot(
            day=0,
            cash=500_000,
            observed_at=datetime(2026, 8, 25, tzinfo=UTC),
        ),
    )


def outcome(
    initial: SimulationState,
    *,
    index: int,
    ending_cash: float,
    ending_customers: float = 1_100,
) -> RolloutOutcome:
    final = initial.model_copy(
        update={
            "week": 12,
            "cash": ending_cash,
            "customers": ending_customers,
        }
    )
    return RolloutOutcome(
        rollout_index=index,
        states=(initial, final),
        ending_cash=ending_cash,
        ending_customers=ending_customers,
        bankrupt=ending_cash < 0,
    )


def test_candidate_generator_returns_four_distinct_bounded_strategies() -> None:
    candidates = generate_candidate_actions(state())

    assert len(candidates) == 4
    assert len({candidate.name for candidate in candidates}) == 4
    assert len({candidate.model_dump_json() for candidate in candidates}) == 4


def test_fragile_high_mean_strategy_loses_to_robust_strategy() -> None:
    initial = state(cash=1_000)
    fragile_action = SimulationAction(
        name="fragile",
        price_per_customer_weekly=50,
        marketing_spend=100,
        development_spend=100,
    )
    robust_action = fragile_action.model_copy(update={"name": "robust"})
    fragile = summarize_candidate(
        initial,
        fragile_action,
        tuple(
            outcome(initial, index=index, ending_cash=value)
            for index, value in enumerate((-1_000, -500, 10_000, 12_000))
        ),
    )
    robust = summarize_candidate(
        initial,
        robust_action,
        tuple(
            outcome(initial, index=index, ending_cash=value)
            for index, value in enumerate((1_300, 1_400, 1_500, 1_600))
        ),
    )

    result = select_robust_strategy((fragile, robust, robust.model_copy(
        update={"action": robust_action.model_copy(update={"name": "robust_alt"}),
                "robust_utility": robust.robust_utility - 0.01}
    )))

    assert fragile.expected_ending_cash > robust.expected_ending_cash
    assert result.selected.action.name == "robust"
    assert result.selection_reason_code == "survival_gated_robust_utility"


def test_customer_growth_does_not_hide_a_cash_loss_in_utility() -> None:
    initial = state(cash=100_000)
    action = SimulationAction(
        name="same_cash",
        price_per_customer_weekly=50,
        marketing_spend=100,
        development_spend=100,
    )
    low_growth = summarize_candidate(
        initial,
        action,
        tuple(
            outcome(
                initial,
                index=index,
                ending_cash=90_000,
                ending_customers=1_000,
            )
            for index in range(4)
        ),
    )
    loss_making_growth = summarize_candidate(
        initial,
        action.model_copy(update={"name": "loss_making_growth"}),
        tuple(
            outcome(
                initial,
                index=index,
                ending_cash=90_000,
                ending_customers=100_000,
            )
            for index in range(4)
        ),
    )

    assert loss_making_growth.expected_customer_growth > low_growth.expected_customer_growth
    assert loss_making_growth.robust_utility == low_growth.robust_utility


def candidate(
    name: str,
    *,
    risk: float,
    utility: float,
    downside: float,
    expected_cash: float = 500_000,
    going_concern_failure: float = 0.0,
) -> CandidateSimulation:
    return CandidateSimulation(
        action=SimulationAction(
            name=name,
            price_per_customer_weekly=50,
            marketing_spend=1_000,
            development_spend=1_000,
        ),
        expected_ending_cash=expected_cash,
        downside_ending_cash=downside,
        bankruptcy_probability=risk,
        going_concern_failure_probability=going_concern_failure,
        expected_customer_growth=10_000,
        robustness=(
            RobustnessLevel.HIGH
            if risk <= 0.05
            else RobustnessLevel.MEDIUM
            if risk <= 0.20
            else RobustnessLevel.LOW
        ),
        robust_utility=utility,
        rollout_count=200,
    )


def test_survival_gate_rejects_live_trace_high_growth_risk() -> None:
    preservation = candidate(
        "cash_preservation",
        risk=0.0,
        utility=0.259437,
        downside=206_985.97,
        expected_cash=218_435.35,
    )
    executive = candidate(
        "executive_proposal",
        risk=0.35,
        utility=23.526286,
        downside=-274_498.49,
        expected_cash=1_063_003.03,
    )
    aggressive = candidate(
        "aggressive_growth",
        risk=0.725,
        utility=9.893874,
        downside=-409_096.33,
    )

    result = select_robust_strategy((preservation, executive, aggressive))

    assert result.selected == preservation
    assert result.selection_reason_code == "survival_gated_robust_utility"
    assert "1/3 survival-and-going-concern-eligible" in result.selection_reason
    assert MAX_ACCEPTABLE_BANKRUPTCY_PROBABILITY == 0.10


def test_all_infeasible_fallback_chooses_minimum_risk_then_downside() -> None:
    minimum_risk_weak_downside = candidate(
        "minimum_risk_weak_downside",
        risk=0.25,
        utility=100.0,
        downside=-90_000,
    )
    minimum_risk_strong_downside = candidate(
        "minimum_risk_strong_downside",
        risk=0.25,
        utility=-100.0,
        downside=-20_000,
    )
    higher_risk = candidate(
        "higher_risk",
        risk=0.30,
        utility=1_000.0,
        downside=50_000,
    )

    result = select_robust_strategy(
        (minimum_risk_weak_downside, higher_risk, minimum_risk_strong_downside)
    )

    assert result.selected == minimum_risk_strong_downside
    assert result.selection_reason_code == "survival_gate_emergency_minimum_risk"
    assert "all 3 strategies exceed" in result.selection_reason


def test_survival_eligible_utility_tie_prefers_stronger_downside() -> None:
    weaker_downside = candidate(
        "weaker_downside",
        risk=0.05,
        utility=1.0,
        downside=100_000,
    )
    stronger_downside = candidate(
        "stronger_downside",
        risk=0.10,
        utility=1.0,
        downside=200_000,
    )
    ineligible = candidate(
        "ineligible",
        risk=0.11,
        utility=999.0,
        downside=300_000,
    )

    result = select_robust_strategy((weaker_downside, ineligible, stronger_downside))

    assert result.selected == stronger_downside


def test_decision_sensitive_experiment_may_spend_a_bounded_downside_budget() -> None:
    standard = candidate(
        "cash_preservation",
        risk=0.0,
        utility=1.0,
        downside=900_000,
    )
    experiment = candidate(
        "controlled_exploration_lead_promotion",
        risk=0.0,
        utility=0.9,
        downside=897_000,
    )

    result = select_robust_strategy(
        (standard, experiment),
        prefer_bounded_exploration=True,
    )

    assert result.selected == experiment
    assert result.selection_reason_code == "decision_sensitive_bounded_exploration"
    assert "cost 3000.00" in result.selection_reason
    assert "budget 4500.00" in result.selection_reason


def test_decision_sensitive_experiment_is_not_forced_past_downside_budget() -> None:
    standard = candidate(
        "cash_preservation",
        risk=0.0,
        utility=1.0,
        downside=900_000,
    )
    experiment = candidate(
        "controlled_exploration_lead_promotion",
        risk=0.0,
        utility=0.9,
        downside=890_000,
    )

    result = select_robust_strategy(
        (standard, experiment),
        prefer_bounded_exploration=True,
    )

    assert result.selected == standard
    assert result.selection_reason_code == "survival_gated_robust_utility"


def test_exploration_preference_does_not_override_cash_survival_gate() -> None:
    standard = candidate(
        "cash_preservation",
        risk=0.0,
        utility=1.0,
        downside=900_000,
    )
    experiment = candidate(
        "controlled_exploration_lead_promotion",
        risk=0.2,
        utility=2.0,
        downside=899_000,
    )

    result = select_robust_strategy(
        (standard, experiment),
        prefer_bounded_exploration=True,
    )

    assert result.selected == standard


def test_bounded_experiment_may_measure_the_going_concern_uncertainty_it_targets() -> None:
    standard = candidate(
        "aggressive_growth",
        risk=0.0,
        utility=1.0,
        downside=900_000,
    )
    experiment = candidate(
        "controlled_exploration_lead_promotion",
        risk=0.0,
        utility=0.9,
        downside=899_000,
        going_concern_failure=0.95,
    )

    result = select_robust_strategy(
        (standard, experiment),
        prefer_bounded_exploration=True,
    )

    assert result.selected == experiment
    assert result.selection_reason_code == "decision_sensitive_bounded_exploration"
    assert "0.950 is recorded" in result.selection_reason


def test_going_concern_gate_rejects_cash_optimization_by_customer_extinction() -> None:
    shutdown = candidate(
        "shutdown_by_price",
        risk=0.0,
        utility=10.0,
        downside=600_000,
        expected_cash=650_000,
        going_concern_failure=1.0,
    )
    viable = candidate(
        "viable_business",
        risk=0.0,
        utility=0.5,
        downside=450_000,
        expected_cash=500_000,
        going_concern_failure=0.0,
    )

    result = select_robust_strategy((shutdown, viable))

    assert result.selected == viable
    assert result.selection_reason_code == "survival_gated_robust_utility"
    assert "going-concern failure probability" in result.selection_reason


def test_selection_fails_closed_when_every_candidate_certainly_ends_the_business() -> None:
    candidates = (
        candidate(
            "shutdown",
            risk=0.0,
            utility=1.0,
            downside=100_000,
            going_concern_failure=1.0,
        ),
        candidate(
            "slow_shutdown",
            risk=0.0,
            utility=2.0,
            downside=200_000,
            going_concern_failure=1.0,
        ),
    )

    with pytest.raises(NoViableStrategyError, match="no viable strategy") as caught:
        select_robust_strategy(candidates)

    assert caught.value.candidate_risks == (
        ("shutdown", 1.0),
        ("slow_shutdown", 1.0),
    )


def test_inherited_extinction_selects_least_damaging_plan_without_claiming_recovery() -> None:
    weaker_downside = candidate(
        "aggressive_growth",
        risk=0.0,
        utility=10.0,
        downside=700_000,
        going_concern_failure=1.0,
    )
    stronger_downside = candidate(
        "cash_preservation",
        risk=0.0,
        utility=0.5,
        downside=800_000,
        going_concern_failure=1.0,
    )

    result = select_robust_strategy(
        (weaker_downside, stronger_downside),
        inherited_going_concern_failure=True,
    )

    assert result.selected == stronger_downside
    assert result.selection_reason_code == "inherited_going_concern_minimum_failure"
    assert "not reported as recovery" in result.selection_reason


def test_inherited_extinction_can_fund_a_bounded_new_quality_support_regime() -> None:
    continuation = candidate(
        "continuation",
        risk=0.0,
        utility=0.5,
        downside=800_000,
        going_concern_failure=1.0,
    )
    quality_frontier = candidate(
        "executive_experiment_targeted_development_quality_frontier",
        risk=0.0,
        utility=-0.5,
        downside=680_000,
        going_concern_failure=1.0,
    )

    result = select_robust_strategy(
        (continuation, quality_frontier),
        prefer_bounded_exploration=True,
        inherited_going_concern_failure=True,
    )

    assert result.selected == quality_frontier
    assert (
        result.selection_reason_code
        == "inherited_continuity_support_frontier_experiment"
    )


def test_inherited_extinction_honors_admitted_controlled_quality_frontier() -> None:
    continuation = candidate(
        "continuation",
        risk=0.0,
        utility=0.5,
        downside=800_000,
        going_concern_failure=1.0,
    )
    controlled = candidate(
        "controlled_exploration_development_q2",
        risk=0.0,
        utility=-0.5,
        downside=775_000,
        going_concern_failure=1.0,
    ).model_copy(
        update={
            "action": candidate(
                "controlled_exploration_development_q2",
                risk=0.0,
                utility=-0.5,
                downside=775_000,
            ).action.model_copy(update={"development_spend": 6_000})
        }
    )

    result = select_robust_strategy(
        (continuation, controlled),
        prefer_bounded_exploration=True,
        inherited_going_concern_failure=True,
    )

    assert result.selected == controlled
    assert result.selection_reason_code == (
        "inherited_continuity_support_frontier_experiment"
    )


def test_support_frontier_uses_strongest_affordable_not_cheapest_scale() -> None:
    continuation = candidate(
        "continuation",
        risk=0.0,
        utility=0.5,
        downside=800_000,
        going_concern_failure=1.0,
    )

    def targeted(name: str, daily: float, downside: float) -> CandidateSimulation:
        base = candidate(
            name,
            risk=0.0,
            utility=-0.5,
            downside=downside,
            going_concern_failure=1.0,
        )
        return base.model_copy(
            update={
                "action": base.action.model_copy(
                    update={
                        "targeted_development_allocations": (
                            TargetedDevelopmentAllocation(
                                segment="S1",
                                daily_spend=daily,
                            ),
                        )
                    }
                )
            }
        )

    cheap = targeted(
        "executive_experiment_targeted_development_quality_m1",
        500,
        790_000,
    )
    strong = targeted(
        "executive_experiment_targeted_development_quality_m2",
        5_000,
        700_000,
    )
    over_budget = targeted(
        "executive_experiment_targeted_development_quality_m3",
        10_000,
        500_000,
    )

    result = select_robust_strategy(
        (continuation, cheap, strong, over_budget),
        prefer_bounded_exploration=True,
        inherited_going_concern_failure=True,
    )

    assert result.selected == strong
    assert "strongest product-support displacement" in result.selection_reason


def test_inherited_extinction_funds_new_regime_measurement_after_support_moves() -> None:
    continuation = candidate(
        "continuation",
        risk=0.0,
        utility=0.5,
        downside=800_000,
        going_concern_failure=1.0,
    )
    probe = candidate(
        "controlled_exploration_marketing_q3",
        risk=0.0,
        utility=-0.5,
        downside=797_000,
        going_concern_failure=1.0,
    )

    result = select_robust_strategy(
        (continuation, probe),
        prefer_bounded_exploration=True,
        inherited_going_concern_failure=True,
    )

    assert result.selected == probe
    assert result.selection_reason_code == "inherited_continuity_bounded_exploration"
    assert "does not assume that the probe converts" in result.selection_reason


def test_inherited_extinction_does_not_fund_probe_past_measurement_budget() -> None:
    continuation = candidate(
        "continuation",
        risk=0.0,
        utility=0.5,
        downside=800_000,
        going_concern_failure=1.0,
    )
    expensive_probe = candidate(
        "controlled_exploration_marketing_q3",
        risk=0.0,
        utility=10.0,
        downside=790_000,
        going_concern_failure=1.0,
    )

    result = select_robust_strategy(
        (continuation, expensive_probe),
        prefer_bounded_exploration=True,
        inherited_going_concern_failure=True,
    )

    assert result.selected == continuation
    assert result.selection_reason_code == "inherited_going_concern_minimum_failure"


def test_empty_funnel_adds_one_week_single_lever_exploration() -> None:
    shrinking = state().model_copy(
        update={
            "week": 9,
            "customers": 20,
            "churn_rate": 0.2,
            "weekly_acquisition": 1,
            "marketing_spend": 4_000,
        }
    )

    actions = generate_candidate_actions(shrinking)
    experiment = next(
        action
        for action in actions
        if action.name == "controlled_exploration_marketing_q6"
    )

    assert experiment.marketing_spend == 5_000
    assert experiment.development_spend == shrinking.development_spend
    assert (experiment.model_tier_a, experiment.model_tier_b, experiment.model_tier_c) == (
        1,
        1,
        1,
    )
    assert experiment.marketing_spend_until_week == 10
    assert experiment.development_spend_until_week == 10
    assert experiment.marketing_spend_after_experiment == 4_000
    assert experiment.development_spend_after_experiment == shrinking.development_spend


def test_failed_funnel_evidence_changes_experiment_and_eventually_stops_it() -> None:
    early = state().model_copy(
        update={
            "week": 1,
            "customers": 0,
            "weekly_acquisition": 0,
            "weekly_leads": 22,
            "weekly_conversions": 0,
            "total_leads": 22,
            "total_conversions": 0,
            "entry_price_monthly": 50,
            "product_quality": 0.2,
        }
    )
    early_actions = generate_candidate_actions(early)
    experiment = next(
        action
        for action in early_actions
        if action.name == "controlled_exploration_lead_promotion_q2"
    )
    assert experiment.marketing_spend == early.marketing_spend
    assert experiment.development_spend == early.development_spend
    assert experiment.lead_promotion_monthly == pytest.approx(10.0)
    assert experiment.lead_promotion_until_week == early.week + 1
    assert experiment.lead_promotion_after_experiment == 0

    repeated_failure = early.model_copy(update={"week": 8, "weekly_leads": 16})
    next_probe = next(
        action
        for action in generate_candidate_actions(
            repeated_failure,
            ExplorationMemory(
                funnel_regime_evidence=(
                    FunnelRegimeEvidence(
                        quality_band=2,
                        leads=127,
                        conversions=0,
                        weeks=8,
                    ),
                )
            ),
        )
        if action.name == "controlled_exploration_development_q2"
    )
    assert next_probe.development_spend_until_week == repeated_failure.week + 5


def test_failed_low_quality_regime_does_not_close_a_new_quality_regime() -> None:
    improved_quality = state().model_copy(
        update={
            "week": 9,
            "customers": 0,
            "weekly_acquisition": 0,
            "weekly_leads": 0,
            "weekly_conversions": 0,
            "product_quality": 0.45,
        }
    )
    memory = ExplorationMemory(
        funnel_regime_evidence=(
            FunnelRegimeEvidence(
                quality_band=2,
                leads=1_098,
                conversions=0,
                weeks=60,
            ),
        )
    )

    actions = generate_candidate_actions(improved_quality, memory)

    assert any(
        action.name == "controlled_exploration_marketing_q4" for action in actions
    )


def test_three_resolved_quality_nulls_switch_to_a_price_probe_before_more_development() -> None:
    observed = state(cash=682_000).model_copy(
        update={
            "week": 15,
            "customers": 0,
            "revenue_weekly": 0,
            "weekly_acquisition": 0,
            "weekly_leads": 0,
            "weekly_conversions": 0,
            "product_quality": 0.409,
            "entry_price_monthly": 25,
        }
    )
    memory = ExplorationMemory(
        funnel_regime_evidence=tuple(
            FunnelRegimeEvidence(
                quality_band=band,
                leads=leads,
                conversions=0,
                weeks=weeks,
            )
            for band, leads, weeks in ((2, 174, 2), (3, 104, 3), (4, 117, 4))
        )
    )

    admission = assess_controlled_exploration(observed, memory)
    actions = generate_candidate_actions(observed, memory)

    assert admission.admitted is True
    assert admission.reason_code == "cross_regime_quality_null_requires_price_probe"
    assert admission.candidate_strategy == "controlled_exploration_lead_promotion_q4"
    assert any(action.name == admission.candidate_strategy for action in actions)
    assert all(
        action.name != "controlled_exploration_development_q4" for action in actions
    )


def test_zero_conversion_funnel_buys_one_promotion_test_without_assuming_a_win() -> None:
    observed = state(cash=980_000).model_copy(
        update={
            "week": 1,
            "revenue_weekly": 0,
            "customers": 0,
            "weekly_acquisition": 0,
            "weekly_leads": 22,
            "weekly_conversions": 0,
            "total_leads": 22,
            "total_conversions": 0,
            "marketing_spend": 0,
            "development_spend": 3_000,
            "product_quality": 0.2,
            "price_per_customer_weekly": 5.833,
            "catalog_price_per_customer_weekly": 5.833,
            "entry_price_monthly": 25,
        }
    )
    actions = generate_candidate_actions(
        observed,
        ExplorationMemory(
            attempted_strategies=("controlled_exploration_marketing_q2",)
        ),
    )

    result = search_strategies(
        state=observed,
        world_model=model(),
        actions=actions,
        horizon_weeks=12,
        n_rollouts=80,
        seed=53,
        prefer_bounded_exploration=True,
    )
    exploitation_only = select_robust_strategy(result.candidates)

    assert result.selected.action.name == "controlled_exploration_lead_promotion_q2"
    assert result.selection_reason_code == "decision_sensitive_bounded_exploration"
    assert exploitation_only.selected.action.name == "aggressive_growth"


def test_experiment_memory_prevents_repetition_and_restores_baseline_spend() -> None:
    after_experiment = state().model_copy(
        update={
            "week": 3,
            "customers": 0,
            "weekly_acquisition": 0,
            "marketing_spend": 5_000,
        }
    )
    actions = generate_candidate_actions(
        after_experiment,
        ExplorationMemory(
            attempted_strategies=("controlled_exploration_marketing_q6",),
            revert_marketing_spend=4_000,
            revert_development_spend=after_experiment.development_spend,
        ),
    )

    assert all(
        action.name != "controlled_exploration_marketing_q6" for action in actions
    )
    balanced = next(action for action in actions if action.name == "balanced_growth")
    assert balanced.marketing_spend == 4_000


def test_unprofitable_unit_economics_adds_attributable_tier_recovery() -> None:
    unprofitable = state().model_copy(
        update={
            "customers": 20,
            "price_per_customer_weekly": 8.0,
            "operating_cost_per_customer_weekly": 20.0,
            "model_tier_a": 4,
            "model_tier_b": 3,
            "model_tier_c": 2,
        }
    )

    actions = generate_candidate_actions(unprofitable)
    recovery = next(action for action in actions if action.name == "unit_cost_recovery")

    assert (recovery.model_tier_a, recovery.model_tier_b, recovery.model_tier_c) == (
        3,
        2,
        1,
    )
    assert recovery.marketing_spend == unprofitable.marketing_spend
    assert recovery.development_spend == unprofitable.development_spend


def test_seeded_strategy_search_is_reproducible_and_machine_readable() -> None:
    result = search_strategies(
        state=state(),
        world_model=model(),
        horizon_weeks=6,
        n_rollouts=40,
        seed=99,
    )
    replay = search_strategies(
        state=state(),
        world_model=model(),
        horizon_weeks=6,
        n_rollouts=40,
        seed=99,
    )

    assert result == replay
    assert len(result.candidates) == 4
    assert result.selected in result.candidates
    assert result.selection_reason
