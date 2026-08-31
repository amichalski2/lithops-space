"""A direction the company holds, as distinct from a question it asks.

Multi-week protection existed only for experiments, so any strategy worth more
than a week had to be dressed as a probe — and probes roll back by default. The
Executive kept diagnosing a constraint correctly, committing for one week, and
reverting; not from inconsistency, but because the harness gave scaffolding to
learning and none to acting.
"""

from __future__ import annotations

import pytest
from lithops.agents.common.structured_output import ExecutiveActionProposalOutput
from lithops.application.executive_selection import assess_experiment
from lithops.domain.models import ActionCommand, ActionPlan, ExperimentProgram
from pydantic import ValidationError


def commitment(**overrides) -> ExperimentProgram:
    base = dict(
        commitment_id="hold-quality-4",
        control="strategy",
        protocol_version="operating-commitment-v1",
        started_week=4,
        minimum_maturity_week=12,
        maximum_end_week=12,
        baseline_value=0.0,
        treatment_value=0.0,
        maximum_cumulative_downside=250_000.0,
        expected_observation="delivered quality clears the S2 floor",
        falsification_condition="quality gains stall for three consecutive weeks",
    )
    base.update(overrides)
    return ExperimentProgram(**base)


def plan_with(program: ExperimentProgram) -> ActionPlan:
    return ActionPlan(
        name="hold-quality",
        strategy_family="executive_product_quality_0",
        rationale="hold the quality investment",
        commands=[
            ActionCommand(
                tool="set_daily_spend",
                arguments={"development": 10_000.0},
                idempotency_key="hold-quality-key",
            )
        ],
        proposal_kind="operating",
        hypothesis_id="hyp_quality_bet",
        experiment_program=program,
    )


class TestStandingCommitment:
    def test_it_owes_no_control_arm_or_measurement_plan(self) -> None:
        # Demanding these is what forced every strategy to pose as an experiment.
        assert assess_experiment(plan_with(commitment()), portfolio=None) == ()

    def test_it_must_name_what_would_stop_it(self) -> None:
        with pytest.raises(ValidationError, match="stop condition"):
            commitment(falsification_condition="   ")

    def test_it_declares_a_spending_limit_like_any_commitment(self) -> None:
        with pytest.raises(ValidationError):
            commitment(maximum_cumulative_downside=0.0)

    def test_holding_a_direction_for_weeks_requires_a_stop_condition(self) -> None:
        with pytest.raises(ValidationError, match="standing_commitment_needs_a_stop"):
            ExecutiveActionProposalOutput.model_validate(
                proposal_payload(experiment_duration_weeks=6, stop_condition="")
            )

    def test_a_single_week_operating_proposal_needs_none(self) -> None:
        proposal = ExecutiveActionProposalOutput.model_validate(
            proposal_payload(experiment_duration_weeks=1, stop_condition="")
        )
        assert proposal.experiment_duration_weeks == 1


def proposal_payload(**overrides) -> dict:
    payload = {
        "name": "hold quality",
        "proposal_kind": "operating",
        "experiment_control": "none",
        "experiment_duration_weeks": 1,
        "hypothesis_id": "hyp_quality_bet",
        "strategy_family": "product_quality",
        "hypothesis": "sustained development clears the S2 quality floor",
        "expected_observation": "quality rises past the floor within the window",
        "rationale": "the floor is the binding constraint on revenue",
        "catalog_price_multiplier": 1.0,
        "weekly_marketing_spend": 5_000.0,
        "daily_spend": {"development": 10_000.0, "operations": 500.0},
        "model_tier_a": 2,
        "model_tier_b": 2,
        "model_tier_c": 2,
        "usage_quota_a": 500,
        "usage_quota_b": 1_000,
        "usage_quota_c": 2_000,
        "capacity_tier": 2,
        "lead_promotion_fraction": 0.0,
        "target_channel": "search_ads",
        "target_segment": "S1",
    }
    payload.update(overrides)
    return payload
