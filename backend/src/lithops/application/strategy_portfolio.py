"""Strategic portfolio reconstruction and the first Executive stage.

Gemini owns diagnosis and hypothesis revision; this module owns everything that
must stay deterministic: the compact evidence brief, append-only governance over
the Executive's update, revision hashing, and replay idempotency. The persisted
revisions are projections of this deterministic application — replay reuses the
stored revision for a week instead of calling the provider again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from lithops.agents.common import (
    HypothesisProposalOutput,
    StrategyPortfolioUpdateOutput,
)
from lithops.domain.insights import InsightRecord, measured_quality_floor_metrics
from lithops.domain.models import (
    DecisionRecord,
    ObservationSnapshot,
    RunRecord,
    utc_now,
)
from lithops.domain.ports import StrategyRepository
from lithops.domain.strategy import (
    ALLOWED_HYPOTHESIS_TRANSITIONS,
    BusinessLever,
    ExperimentOutcome,
    ExperimentOutcomeStatus,
    HypothesisStatus,
    ObjectiveSpec,
    StrategicHypothesis,
    StrategicPortfolio,
    StrategicPortfolioRevision,
    portfolio_revision_id,
)
from lithops.evaluation.trajectory import weekly_trajectory

# These statuses observed nothing about the offer itself, so they can support a
# reach reinterpretation but can never falsify a conversion claim.
NON_FALSIFYING_OUTCOME_STATUSES = frozenset(
    {
        ExperimentOutcomeStatus.IMMATURE,
        ExperimentOutcomeStatus.NO_EXPOSURE,
        ExperimentOutcomeStatus.CENSORED,
        ExperimentOutcomeStatus.INVALID_EXECUTION,
        ExperimentOutcomeStatus.STOPPED_FOR_SAFETY,
    }
)


class StrategyArchitectEngine(Protocol):
    async def update_strategy_portfolio(
        self,
        *,
        brief: dict,
    ) -> StrategyPortfolioUpdateOutput: ...


@dataclass(frozen=True, slots=True)
class PortfolioUpdateResult:
    revision: StrategicPortfolioRevision
    diagnostics: tuple[str, ...]
    replayed: bool


def experiment_budget_ceiling(cash: float) -> float:
    """A deterministic research-spend ceiling, never a spend target."""

    return max(0.0, min(30_000.0, cash * 0.03))


def portfolio_context_for_proposals(portfolio: StrategicPortfolio) -> dict[str, Any]:
    """The compact portfolio state sent with the candidate-proposal request."""

    return {
        "as_of_week": portfolio.as_of_week,
        "binding_constraint": portfolio.binding_constraint,
        "remaining_experiment_budget": portfolio.remaining_experiment_budget,
        "active_hypothesis_ids": list(portfolio.active_hypothesis_ids),
        "unresolved_questions": list(portfolio.unresolved_questions),
        "hypotheses": [
            {
                "hypothesis_id": hypothesis.hypothesis_id,
                "causal_claim": hypothesis.causal_claim,
                "target_outcome": hypothesis.target_outcome,
                "levers": [lever.value for lever in hypothesis.levers],
                "segment": hypothesis.segment,
                "channel": hypothesis.channel,
                "status": hypothesis.status.value,
                "falsification_reason": hypothesis.falsification_reason,
                "successor_hypothesis_ids": list(hypothesis.successor_hypothesis_ids),
            }
            for hypothesis in portfolio.hypotheses
        ],
    }


# Levers whose payoff is the quality customers receive: below a measured
# participation floor their conversion effect is untestable, not absent.
_QUALITY_SIDE_LEVERS = frozenset(
    {
        BusinessLever.TIER,
        BusinessLever.DEVELOPMENT,
        BusinessLever.TARGETED_DEVELOPMENT,
    }
)


def _operative_regime_reached(
    outcome: ExperimentOutcome, individual_quality_floor: float | None
) -> bool | None:
    """Whether delivered quality during the window reached the measured floor.

    ``None`` when nothing can be said: no floor has been bought, or the window
    carried no delivered-quality proxies. A quality-side experiment run entirely
    below the floor observed the regime where nothing converts by construction —
    its zero is a fact about the regime, not about the lever.
    """

    if individual_quality_floor is None:
        return None
    proxies = outcome.envelope.segment_plan_quality_proxies
    if not proxies:
        return None
    return max(proxies.values()) >= individual_quality_floor


def _outcome_summary(
    outcome: ExperimentOutcome,
    *,
    individual_quality_floor: float | None = None,
) -> dict[str, Any]:
    return {
        "commitment_id": outcome.commitment_id,
        "hypothesis_id": outcome.hypothesis_id,
        "outcome_status": outcome.outcome_status.value,
        "leads": outcome.leads,
        "matured_leads": outcome.matured_leads,
        "conversions": outcome.conversions,
        "exposure_spend": outcome.exposure_spend,
        "started_week": outcome.started_week,
        "measured_week": outcome.measured_week,
        "operative_regime_reached": _operative_regime_reached(
            outcome, individual_quality_floor
        ),
        "envelope": {
            "segment": outcome.envelope.segment,
            "channel": outcome.envelope.channel,
            "quality_band": outcome.envelope.quality_band,
            "catalog_prices": outcome.envelope.catalog_prices,
            "model_tiers": outcome.envelope.model_tiers,
            "promotion": outcome.envelope.promotion,
        },
    }


def insight_summaries(insights: tuple[InsightRecord, ...]) -> list[dict[str, Any]]:
    """Purchased estimates with the accuracy band their source declared."""

    return [
        {
            "week": record.week,
            "tool": record.tool,
            "target_group": record.target_group,
            "info_level": record.info_level,
            "accuracy_band": record.noise_band,
            "willingness_to_pay_monthly": record.willingness_to_pay_monthly,
            "usage_units_per_day": record.usage_units_per_day,
            "quality_floor": record.quality_floor,
            "market_cap_customers": record.market_cap_customers,
            "discovered_group": record.discovered_group,
            "parse_status": record.parse_status.value,
        }
        for record in insights
        if record.has_decision_content
    ]


def build_strategic_evidence_brief(
    *,
    run: RunRecord,
    observation: ObservationSnapshot,
    portfolio: StrategicPortfolio | None,
    outcomes: tuple[ExperimentOutcome, ...],
    decision_history: tuple[DecisionRecord, ...] = (),
    model_health_status: str | None = None,
    insights: tuple[InsightRecord, ...] = (),
) -> dict[str, Any]:
    """A compact deterministic evidence reduction — not a transcript."""

    week = observation.day // 7
    individual_floor = measured_quality_floor_metrics(insights).get(
        "measured_quality_floor_individual"
    )
    matured = [outcome for outcome in outcomes if outcome.measured_week is not None]
    newly_matured = [
        outcome
        for outcome in matured
        if portfolio is None
        or (outcome.measured_week or 0) >= portfolio.as_of_week
    ]
    active_program = None
    if decision_history:
        program = decision_history[-1].action_plan.experiment_program
        if program is not None and week < program.maximum_end_week:
            active_program = {
                "commitment_id": program.commitment_id,
                "control": program.control,
                "started_week": program.started_week,
                "minimum_maturity_week": program.minimum_maturity_week,
                "maximum_end_week": program.maximum_end_week,
                "target_segment": program.target_segment,
                "target_channel": program.target_channel,
                "maximum_cumulative_downside": program.maximum_cumulative_downside,
            }
    return {
        "run_id": str(run.id),
        "as_of_week": week,
        "remaining_days": max(0, run.horizon_days - observation.day),
        "cash": observation.cash,
        "objective": (
            "risk-adjusted terminal cash subject to solvency and going concern"
        ),
        "remaining_experiment_budget": experiment_budget_ceiling(observation.cash),
        "observation": observation.model_dump(mode="json"),
        "portfolio": (
            portfolio_context_for_proposals(portfolio) if portfolio is not None else None
        ),
        "newly_matured_outcomes": [
            _outcome_summary(outcome, individual_quality_floor=individual_floor)
            for outcome in newly_matured
        ],
        # Older matured outcomes are not re-sent: the prompt never referenced
        # the full history, each entry re-appeared inside it anyway, and the
        # portfolio's hypothesis statuses already carry what they settled.
        "active_commitment": active_program,
        "model_health_status": model_health_status,
        "purchased_insights": insight_summaries(insights),
        "weekly_trajectory": weekly_trajectory(decision_history),
    }


def _new_hypothesis(proposal: HypothesisProposalOutput) -> StrategicHypothesis:
    return StrategicHypothesis(
        hypothesis_id=proposal.hypothesis_id,
        causal_claim=proposal.causal_claim,
        target_outcome=proposal.target_outcome,
        levers=tuple(BusinessLever(lever) for lever in proposal.levers),
        segment=proposal.segment or None,
        channel=proposal.channel or None,
        status=HypothesisStatus.PROPOSED,
    )


def _append_unique(values: tuple[str, ...], addition: str) -> tuple[str, ...]:
    return values if addition in values else (*values, addition)


def apply_portfolio_update(
    previous: StrategicPortfolio | None,
    update: StrategyPortfolioUpdateOutput,
    *,
    as_of_week: int,
    objective: ObjectiveSpec,
    remaining_experiment_budget: float,
    outcomes: tuple[ExperimentOutcome, ...],
    individual_quality_floor: float | None = None,
) -> tuple[StrategicPortfolio, tuple[str, ...]]:
    """Deterministically apply the Executive's diff under append-only governance.

    Illegal edits are skipped with a diagnostic rather than applied: history is
    never rewritten, falsified stays falsified inside its envelope, and a
    non-falsifying outcome (for example no_exposure) can never justify
    falsification.
    """

    diagnostics: list[str] = []
    hypotheses: dict[str, StrategicHypothesis] = (
        {hypothesis.hypothesis_id: hypothesis for hypothesis in previous.hypotheses}
        if previous is not None
        else {}
    )
    outcome_by_commitment: dict[str, ExperimentOutcome] = {}
    for outcome in outcomes:
        current = outcome_by_commitment.get(outcome.commitment_id)
        if current is None or (outcome.measured_week or -1) >= (
            current.measured_week or -1
        ):
            outcome_by_commitment[outcome.commitment_id] = outcome

    falsification_supported: set[str] = set()
    for interpretation in update.outcome_interpretations:
        outcome = outcome_by_commitment.get(interpretation.commitment_id)
        if outcome is None:
            diagnostics.append(
                f"unknown_commitment_interpreted:{interpretation.commitment_id}"
            )
            continue
        effective = interpretation.interpretation
        if effective == "falsifies" and (
            outcome.outcome_status in NON_FALSIFYING_OUTCOME_STATUSES
        ):
            effective = "inconclusive"
            diagnostics.append(
                "falsification_downgraded_to_inconclusive:"
                f"{interpretation.commitment_id}:{outcome.outcome_status.value}"
            )
        if effective == "falsifies" and (
            outcome.outcome_status is ExperimentOutcomeStatus.EXPOSED_ZERO_CONVERSION
        ):
            # A quality-side hypothesis probed entirely below a measured
            # participation floor was never tested in its operative regime:
            # the regime converts nothing by construction, so its zero says
            # nothing about the lever. Downgraded, never silently rewritten.
            interpreted = hypotheses.get(interpretation.hypothesis_id)
            quality_side = interpreted is not None and any(
                lever in _QUALITY_SIDE_LEVERS for lever in interpreted.levers
            )
            if quality_side and (
                _operative_regime_reached(outcome, individual_quality_floor)
                is not True
            ):
                effective = "inconclusive"
                diagnostics.append(
                    "falsification_downgraded_below_participation_floor:"
                    f"{interpretation.commitment_id}"
                )
        if effective == "falsifies":
            falsification_supported.add(interpretation.hypothesis_id)
        hypothesis = hypotheses.get(interpretation.hypothesis_id)
        if hypothesis is not None:
            hypotheses[hypothesis.hypothesis_id] = hypothesis.model_copy(
                update={
                    "evidence_refs": _append_unique(
                        hypothesis.evidence_refs, f"outcome:{outcome.id}"
                    )
                }
            )

    for proposal in update.new_hypotheses:
        if proposal.hypothesis_id in hypotheses:
            diagnostics.append(f"duplicate_hypothesis_id:{proposal.hypothesis_id}")
            continue
        predecessor = None
        if proposal.predecessor_hypothesis_id:
            predecessor = hypotheses.get(proposal.predecessor_hypothesis_id)
            if predecessor is None:
                diagnostics.append(
                    f"unknown_predecessor:{proposal.predecessor_hypothesis_id}"
                )
                continue
        hypotheses[proposal.hypothesis_id] = _new_hypothesis(proposal)
        if predecessor is not None:
            hypotheses[predecessor.hypothesis_id] = predecessor.model_copy(
                update={
                    "successor_hypothesis_ids": _append_unique(
                        predecessor.successor_hypothesis_ids, proposal.hypothesis_id
                    )
                }
            )

    for status_update in update.status_updates:
        hypothesis = hypotheses.get(status_update.hypothesis_id)
        if hypothesis is None:
            diagnostics.append(f"unknown_hypothesis:{status_update.hypothesis_id}")
            continue
        new_status = HypothesisStatus(status_update.new_status)
        if new_status == hypothesis.status:
            continue
        if new_status not in ALLOWED_HYPOTHESIS_TRANSITIONS[hypothesis.status]:
            diagnostics.append(
                "illegal_status_transition:"
                f"{hypothesis.hypothesis_id}:{hypothesis.status.value}"
                f"->{new_status.value}"
            )
            continue
        changes: dict[str, Any] = {"status": new_status}
        if new_status is HypothesisStatus.FALSIFIED:
            if hypothesis.hypothesis_id not in falsification_supported:
                diagnostics.append(
                    "falsification_without_valid_exposure:"
                    f"{hypothesis.hypothesis_id}"
                )
                continue
            if hypothesis.falsification_reason is None:
                changes["falsification_reason"] = status_update.reason
        if new_status is HypothesisStatus.SUPERSEDED:
            successors = tuple(status_update.successor_hypothesis_ids)
            missing = [
                successor for successor in successors if successor not in hypotheses
            ]
            if not successors or missing:
                diagnostics.append(
                    f"superseded_without_successor:{hypothesis.hypothesis_id}"
                )
                continue
            merged = hypothesis.successor_hypothesis_ids
            for successor in successors:
                merged = _append_unique(merged, successor)
            changes["successor_hypothesis_ids"] = merged
        hypotheses[hypothesis.hypothesis_id] = hypothesis.model_copy(update=changes)

    open_statuses = {HypothesisStatus.PROPOSED, HypothesisStatus.RUNNING}
    requested_active = tuple(update.active_hypothesis_ids) or (
        previous.active_hypothesis_ids if previous is not None else ()
    )
    active: list[str] = []
    for hypothesis_id in requested_active:
        hypothesis = hypotheses.get(hypothesis_id)
        if hypothesis is None or hypothesis.status not in open_statuses:
            diagnostics.append(f"inactive_hypothesis_dropped:{hypothesis_id}")
            continue
        if hypothesis_id not in active:
            active.append(hypothesis_id)

    portfolio = StrategicPortfolio(
        as_of_week=as_of_week,
        objective=objective,
        binding_constraint=update.binding_constraint,
        active_hypothesis_ids=tuple(active),
        hypotheses=tuple(hypotheses.values()),
        remaining_experiment_budget=remaining_experiment_budget,
        unresolved_questions=tuple(update.unresolved_questions),
        prior_portfolio_hash=(
            previous.portfolio_hash if previous is not None else None
        ),
    )
    return portfolio, tuple(diagnostics)


async def update_strategic_portfolio(
    *,
    run: RunRecord,
    observation: ObservationSnapshot,
    executive: object,
    strategy_repository: StrategyRepository,
    decision_history: tuple[DecisionRecord, ...] = (),
    model_health_status: str | None = None,
) -> PortfolioUpdateResult | None:
    """Run the first Executive stage exactly once per planning week.

    Replay and crash-retry reuse the persisted revision for the week instead of
    calling the provider again, so restart reconstructs the same revision.
    Returns None when the engine does not implement the architect stage.
    """

    architect = getattr(executive, "update_strategy_portfolio", None)
    if not callable(architect):
        return None
    week = observation.day // 7
    latest = await strategy_repository.get_latest_portfolio_revision(run.id)
    if latest is not None and latest.week >= week:
        return PortfolioUpdateResult(revision=latest, diagnostics=(), replayed=True)
    outcomes = tuple(await strategy_repository.list_experiment_outcomes(run.id))
    insights = tuple(await strategy_repository.list_insight_records(run.id))
    previous = latest.portfolio if latest is not None else None
    brief = build_strategic_evidence_brief(
        run=run,
        observation=observation,
        portfolio=previous,
        outcomes=outcomes,
        decision_history=decision_history,
        model_health_status=model_health_status,
        insights=insights,
    )
    update = await architect(brief=brief)
    portfolio, diagnostics = apply_portfolio_update(
        previous,
        update,
        as_of_week=week,
        objective=ObjectiveSpec(horizon_day=run.horizon_days),
        remaining_experiment_budget=experiment_budget_ceiling(observation.cash),
        outcomes=outcomes,
        individual_quality_floor=measured_quality_floor_metrics(insights).get(
            "measured_quality_floor_individual"
        ),
    )
    revision_number = latest.revision + 1 if latest is not None else 1
    revision = StrategicPortfolioRevision(
        id=portfolio_revision_id(run.id, revision_number),
        run_id=run.id,
        week=week,
        revision=revision_number,
        portfolio=portfolio,
        created_at=utc_now(),
    )
    persisted = await strategy_repository.append_portfolio_revision(revision)
    return PortfolioUpdateResult(
        revision=persisted,
        diagnostics=diagnostics,
        replayed=False,
    )


__all__ = [
    "NON_FALSIFYING_OUTCOME_STATUSES",
    "PortfolioUpdateResult",
    "StrategyArchitectEngine",
    "apply_portfolio_update",
    "build_strategic_evidence_brief",
    "experiment_budget_ceiling",
    "portfolio_context_for_proposals",
    "update_strategic_portfolio",
]
