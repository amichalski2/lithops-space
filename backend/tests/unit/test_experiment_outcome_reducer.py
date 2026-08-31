from typing import Any
from uuid import UUID

import pytest
from lithops.application.step_run import RunManager
from lithops.domain.evidence import (
    AcquisitionEvidence,
    CohortEvidence,
    ConfigurationEvidence,
    WeeklyEvidencePacket,
)
from lithops.domain.models import (
    ActionCommand,
    ActionPlan,
    ActionReceipt,
    CashForecast,
    CashForecasts,
    DecisionRecord,
    DecisionStatus,
    ExperimentProgram,
    ObservationSnapshot,
    ReceiptStatus,
    RunRecord,
)
from lithops.domain.strategy import ExperimentOutcomeStatus
from lithops.evaluation.experiment_outcomes import (
    evidence_envelope_from_observation,
    matched_experiment_evidence,
    probe_weeks,
    reduce_experiment_outcome,
)
from lithops.infrastructure.persistence.repositories import InMemoryRunRepository

RUN_ID = UUID("88888888-8888-8888-8888-888888888888")


def base_metrics(**overrides: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "weekly_leads": 0.0,
        "weekly_conversions": 0.0,
        "weekly_lost_leads": 0.0,
        "product_quality": 0.62,
        "price_a": 99.0,
        "price_b": 299.0,
        "price_c": 999.0,
        "model_tier_a": 1.0,
        "model_tier_b": 2.0,
        "model_tier_c": 3.0,
        "lead_promotion_monthly": 0.0,
        "marketing_spend": 3_500.0,
        "marketing_spend_search_ads_weekly": 0.0,
        "marketing_spend_linkedin_weekly": 0.0,
        "development_spend": 1_750.0,
        "targeted_development_spend": 0.0,
    }
    metrics.update(overrides)
    return metrics


def linkedin_program(commitment_id: str = "probe_e1_linkedin") -> ExperimentProgram:
    return ExperimentProgram(
        commitment_id=commitment_id,
        control="targeted_development",
        started_week=2,
        minimum_maturity_week=4,
        maximum_end_week=5,
        baseline_value=0.0,
        treatment_value=500.0,
        maximum_cumulative_downside=6_000.0,
        expected_observation="E1 LinkedIn leads within the probe week",
        falsification_condition="zero matured conversions on matched support",
        target_segment="E1",
        target_channel="linkedin",
        acquisition_probe_weekly_spend=2_500.0,
    )


def search_program(commitment_id: str = "probe_s1_search") -> ExperimentProgram:
    return ExperimentProgram(
        commitment_id=commitment_id,
        control="marketing",
        started_week=2,
        minimum_maturity_week=3,
        maximum_end_week=3,
        baseline_value=3_500.0,
        treatment_value=6_000.0,
        maximum_cumulative_downside=3_000.0,
        expected_observation="S1 search leads convert at the entry price",
        falsification_condition="zero matured conversions on matched support",
        target_segment="S1",
        target_channel="search_ads",
    )


def forecasts() -> CashForecasts:
    return CashForecasts(
        items=[
            CashForecast(horizon_days=horizon, point=100.0, lower=90.0, upper=110.0)
            for horizon in (7, 28, 84, 182)
        ]
    )


def decision(
    week: int,
    program: ExperimentProgram,
    metrics: dict[str, Any] | None,
) -> DecisionRecord:
    plan = ActionPlan(
        name=f"experiment week {week}",
        strategy_family=f"executive_experiment_{program.control}_h1",
        rationale="committed experiment program week",
        commands=[
            ActionCommand(
                tool="set_daily_spend",
                arguments={"operations": 500.0, "development": 250.0},
                idempotency_key=f"{program.commitment_id}-w{week}",
            )
        ],
        proposal_kind="experiment",
        hypothesis_id="h_probe",
        experiment_control=program.control,
        evidence_regime="observed_operating_regime",
        experiment_expires_week=program.maximum_end_week,
        experiment_program=program,
    )
    return DecisionRecord(
        run_id=RUN_ID,
        week=week,
        status=DecisionStatus.COMMITTED,
        observation=ObservationSnapshot(day=week * 7, cash=250_000.0),
        action_plan=plan,
        forecasts=forecasts(),
        actual_outcome=(
            ObservationSnapshot(day=(week + 1) * 7, cash=245_000.0, metrics=metrics)
            if metrics is not None
            else None
        ),
    )


def reduce(
    program: ExperimentProgram,
    decisions: list[DecisionRecord],
    *,
    current_week: int,
    receipts: dict[UUID, list[ActionReceipt]] | None = None,
    stopped_week: int | None = None,
):
    return reduce_experiment_outcome(
        run_id=RUN_ID,
        program=program,
        hypothesis_id="h_probe",
        decisions=decisions,
        receipts_by_decision=receipts,
        current_week=current_week,
        stopped_week=stopped_week,
    )


def linkedin_probe_decisions(probe_metrics: dict[str, Any]) -> list[DecisionRecord]:
    program = linkedin_program()
    return [
        decision(2, program, base_metrics()),
        decision(3, program, base_metrics()),
        decision(4, program, probe_metrics),
    ]


class TestOutcomeClassification:
    def test_e1_linkedin_zero_leads_is_no_exposure(self) -> None:
        outcome = reduce(
            linkedin_program(),
            linkedin_probe_decisions(
                base_metrics(marketing_spend_linkedin_weekly=2_500.0)
            ),
            current_week=5,
        )
        assert outcome.outcome_status is ExperimentOutcomeStatus.NO_EXPOSURE
        assert outcome.envelope.segment == "E1"
        assert outcome.envelope.channel == "linkedin"
        assert outcome.exposure_spend == pytest.approx(2_500.0)
        assert outcome.measured_week == 4

    def test_no_exposure_does_not_match_earlier_s1_search_history(self) -> None:
        s1_outcome = reduce(
            search_program(),
            [
                decision(
                    2,
                    search_program(),
                    base_metrics(
                        weekly_leads=12.0,
                        weekly_lost_leads=12.0,
                        marketing_spend_search_ads_weekly=6_000.0,
                    ),
                )
            ],
            current_week=3,
        )
        assert (
            s1_outcome.outcome_status is ExperimentOutcomeStatus.EXPOSED_ZERO_CONVERSION
        )
        e1_outcome = reduce(
            linkedin_program(),
            linkedin_probe_decisions(
                base_metrics(marketing_spend_linkedin_weekly=2_500.0)
            ),
            current_week=5,
        )
        assert matched_experiment_evidence([s1_outcome], e1_outcome.envelope) == ()
        assert matched_experiment_evidence([s1_outcome], s1_outcome.envelope) == (
            s1_outcome,
        )

    def test_s1_search_leads_without_conversions_is_exposed_zero_conversion(
        self,
    ) -> None:
        outcome = reduce(
            search_program(),
            [
                decision(
                    2,
                    search_program(),
                    base_metrics(weekly_leads=9.0, weekly_lost_leads=9.0),
                )
            ],
            current_week=3,
        )
        assert outcome.outcome_status is ExperimentOutcomeStatus.EXPOSED_ZERO_CONVERSION
        assert outcome.leads == 9
        assert outcome.matured_leads == 9
        assert outcome.conversions == 0

    def test_unmatured_leads_are_censored_not_falsifying(self) -> None:
        outcome = reduce(
            search_program(),
            [decision(2, search_program(), base_metrics(weekly_leads=9.0))],
            current_week=3,
        )
        assert outcome.outcome_status is ExperimentOutcomeStatus.CENSORED
        assert outcome.matured_leads == 0

    def test_conversions_are_positive_conversion(self) -> None:
        outcome = reduce(
            search_program(),
            [
                decision(
                    2,
                    search_program(),
                    base_metrics(
                        weekly_leads=9.0,
                        weekly_conversions=2.0,
                        weekly_lost_leads=4.0,
                    ),
                )
            ],
            current_week=3,
        )
        assert outcome.outcome_status is ExperimentOutcomeStatus.POSITIVE_CONVERSION
        assert outcome.conversions == 2

    def test_v2_outcome_uses_matched_cohort_instead_of_global_weekly_spike(
        self,
    ) -> None:
        program = search_program().model_copy(
            update={
                "protocol_version": "experiment-program-v2",
                "baseline_configuration": {"weekly_marketing_spend": 3_500.0},
                "treatment_configuration": {"weekly_marketing_spend": 6_000.0},
                "measurement_plan": [
                    {
                        "source": "cohort",
                        "metric": "conversion_rate",
                        "target_segment": "S1",
                        "target_channel": "search_ads",
                    }
                ],
            }
        )
        packet = WeeklyEvidencePacket(
            day=21,
            window_start_day_exclusive=14,
            window_end_day_inclusive=21,
            acquisition=(
                AcquisitionEvidence(
                    segment="S1", channel="search_ads", leads=51, spend=1_000
                ),
            ),
            cohorts=(
                CohortEvidence(
                    segment="S1",
                    channel="search_ads",
                    leads=51,
                    conversions=0,
                    losses=51,
                    pending=0,
                ),
                CohortEvidence(
                    segment="S2",
                    channel="social_media",
                    leads=304,
                    conversions=20,
                    losses=284,
                    pending=0,
                ),
            ),
            configuration=ConfigurationEvidence(
                prices={"A": 99.0, "B": 299.0, "C": 999.0},
                model_tiers={"A": 1, "B": 2, "C": 3},
                daily_channel_spend={"search_ads": 1_000 / 7},
                daily_operations_spend=500,
                daily_development_spend=250,
                capacity_tier=0,
            ),
        )
        probe = decision(
            2,
            program,
            base_metrics(
                weekly_leads=355,
                weekly_conversions=20,
                weekly_lost_leads=335,
            ),
        )
        probe = probe.model_copy(
            update={
                "actual_outcome": probe.actual_outcome.model_copy(
                    update={"evidence": packet}, deep=True
                )
            },
            deep=True,
        )

        outcome = reduce(program, [probe], current_week=3)

        assert outcome.outcome_status is ExperimentOutcomeStatus.EXPOSED_ZERO_CONVERSION
        assert outcome.leads == 51
        assert outcome.conversions == 0
        assert outcome.exposure_spend == 1_000

    def test_missing_probe_observation_is_immature(self) -> None:
        program = linkedin_program()
        outcome = reduce(
            program,
            [decision(2, program, base_metrics()), decision(3, program, base_metrics())],
            current_week=4,
        )
        assert outcome.outcome_status is ExperimentOutcomeStatus.IMMATURE
        assert outcome.measured_week is None

    def test_rejected_receipt_is_invalid_execution(self) -> None:
        program = search_program()
        probe = decision(2, program, base_metrics(weekly_leads=9.0))
        receipt = ActionReceipt(
            run_id=RUN_ID,
            decision_id=probe.id,
            idempotency_key="probe_s1_search-w2",
            tool="set_daily_spend",
            status=ReceiptStatus.REJECTED,
        )
        outcome = reduce(
            program,
            [probe],
            current_week=3,
            receipts={probe.id: [receipt]},
        )
        assert outcome.outcome_status is ExperimentOutcomeStatus.INVALID_EXECUTION
        assert outcome.leads == 0

    def test_stopped_for_safety(self) -> None:
        program = linkedin_program()
        outcome = reduce(
            program,
            [decision(2, program, base_metrics())],
            current_week=3,
            stopped_week=3,
        )
        assert outcome.outcome_status is ExperimentOutcomeStatus.STOPPED_FOR_SAFETY
        assert outcome.measured_week == 3


class TestEnvelopeAttribution:
    def test_offer_configuration_changes_create_distinct_envelopes(self) -> None:
        base_observation = ObservationSnapshot(day=21, cash=1.0, metrics=base_metrics())
        base = evidence_envelope_from_observation(
            base_observation, segment="S1", channel="search_ads"
        )
        for changed_metrics in (
            base_metrics(price_a=89.0),
            base_metrics(model_tier_b=3.0),
            base_metrics(lead_promotion_monthly=49.0),
            base_metrics(product_quality=0.75),
        ):
            changed = evidence_envelope_from_observation(
                ObservationSnapshot(day=21, cash=1.0, metrics=changed_metrics),
                segment="S1",
                channel="search_ads",
            )
            assert not base.matches(changed)
        assert not base.matches(
            evidence_envelope_from_observation(
                base_observation, segment="E1", channel="search_ads"
            )
        )
        assert not base.matches(
            evidence_envelope_from_observation(
                base_observation, segment="S1", channel="linkedin"
            )
        )

    def test_probe_weeks_cover_treatment_windows(self) -> None:
        assert probe_weeks(search_program()) == (2,)
        assert probe_weeks(linkedin_program()) == (4,)


class TestReplayDeterminism:
    def test_reducing_twice_yields_the_identical_record(self) -> None:
        decisions = linkedin_probe_decisions(
            base_metrics(marketing_spend_linkedin_weekly=2_500.0)
        )
        first = reduce(linkedin_program(), decisions, current_week=5)
        second = reduce(linkedin_program(), decisions, current_week=5)
        assert first == second

    def test_later_program_weeks_do_not_mutate_the_final_record(self) -> None:
        decisions = linkedin_probe_decisions(
            base_metrics(marketing_spend_linkedin_weekly=2_500.0)
        )
        final = reduce(linkedin_program(), decisions, current_week=5)
        with_revert_week = [*decisions, decision(5, linkedin_program(), base_metrics())]
        assert reduce(linkedin_program(), with_revert_week, current_week=6) == final


class TestRunManagerRecording:
    @pytest.mark.asyncio
    async def test_final_outcome_is_persisted_and_immature_is_not(self) -> None:
        repository = InMemoryRunRepository()
        manager = RunManager(
            repository=repository,
            benchmark=None,  # type: ignore[arg-type]
            decision_engine=None,  # type: ignore[arg-type]
            executive_authority_v2=True,
        )
        run = await repository.create_run(RunRecord(id=RUN_ID))
        program = search_program()
        probe = decision(2, program, base_metrics(weekly_leads=9.0, weekly_lost_leads=9.0))
        await repository.save_decision(probe)

        immature_probe = decision(3, linkedin_program("probe_immature"), base_metrics())
        await repository.save_decision(immature_probe)
        recorded = await manager._record_experiment_outcome(
            run, immature_probe, immature_probe.actual_outcome
        )
        assert recorded is None
        assert await repository.list_experiment_outcomes(RUN_ID) == []

        recorded = await manager._record_experiment_outcome(
            run, probe, probe.actual_outcome
        )
        assert recorded is not None
        assert (
            recorded.outcome_status is ExperimentOutcomeStatus.EXPOSED_ZERO_CONVERSION
        )
        replayed = await manager._record_experiment_outcome(
            run, probe, probe.actual_outcome
        )
        assert replayed == recorded
        assert await repository.list_experiment_outcomes(RUN_ID) == [recorded]
