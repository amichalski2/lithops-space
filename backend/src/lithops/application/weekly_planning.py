"""Translate observations into robust executable plans and cash forecast commitments."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil
from statistics import fmean, median
from typing import Protocol

from pydantic import ValidationError

from lithops.application.executive_selection import (
    ExecutiveAuthorityContext,
    build_evaluation_cards,
    commitment_verdict_from_choice,
    is_operational_candidate,
    parameters_never_measured,
    parameters_still_on_priors,
    quality_position_facts,
    record_commitment_review,
    run_executive_selection,
)
from lithops.application.strategy_portfolio import experiment_budget_ceiling
from lithops.domain.economics import PeriodicMoney, RatePeriod
from lithops.domain.evaluation import CashSensitivityEstimate
from lithops.domain.executable_model import CompanyModelPredictRequest, FittedModel
from lithops.domain.insights import InformationRequest
from lithops.domain.models import (
    ActionCommand,
    ActionPlan,
    CandidateEvaluationRecord,
    CashForecast,
    CashForecasts,
    DecisionRecord,
    ExperimentProgram,
    ObservationSnapshot,
    ProposalBatch,
    ProposalRejection,
    RunRecord,
    construction_veto_codes,
)
from lithops.domain.ports.executable_model import ExecutableCompanyModel
from lithops.domain.world_model import WorldModelVersion
from lithops.evaluation.interval_math import nearest_rank_quantile
from lithops.model_runtime.invariants import evaluate_model_outcomes
from lithops.simulator import (
    PendingResearch,
    ResearchTierFacts,
    SimulationAction,
    SimulationState,
    TargetedAdAllocation,
    TargetedDevelopmentAllocation,
    simulate,
)
from lithops.simulator.invariants import evaluate_simulation_action
from lithops.simulator.state_transition import advance_simulation_week
from lithops.simulator.strategy_search import (
    CandidateSimulation,
    ExplorationAdmission,
    ExplorationMemory,
    FunnelRegimeEvidence,
    RobustnessLevel,
    StrategySearchResult,
    assess_controlled_exploration,
    cash_robust_utility,
    experiment_hypothesis_key,
    generate_candidate_actions,
    search_strategies,
    select_robust_strategy,
)

FORECAST_WEEKS = {7: 1, 28: 4, 84: 12, 182: 26}
MAX_WEEKLY_SPEND = 70_000.0
INITIAL_MONTHLY_PRICES = (25.0, 69.0, 179.0)
INITIAL_WEEKLY_AVERAGE_PRICE = fmean(INITIAL_MONTHLY_PRICES) * 7.0 / 30.0


def _remaining_horizon_days(run: RunRecord, observation: ObservationSnapshot) -> int:
    """Whole-week terminal horizon for the benchmark's actual remaining objective."""

    remaining = max(7, run.horizon_days - observation.day)
    return max(7, ceil(remaining / 7) * 7)


class ExecutiveProposalEngine(Protocol):
    async def decide(
        self,
        *,
        run: RunRecord,
        observation: ObservationSnapshot,
    ) -> tuple[ActionPlan, CashForecasts]: ...


async def _request_executive_plans(
    executive: ExecutiveProposalEngine,
    *,
    run: RunRecord,
    observation: ObservationSnapshot,
    decision_history: tuple[DecisionRecord, ...],
    portfolio_context: dict | None = None,
    rejection_feedback: tuple[dict, ...] | None = None,
) -> ProposalBatch:
    proposer = getattr(executive, "propose_actions", None)
    if callable(proposer):
        parameters = inspect.signature(proposer).parameters
        accepts_extra = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )

        def accepts(name: str) -> bool:
            return name in parameters or accepts_extra

        kwargs: dict[str, object] = {
            "run": run,
            "observation": observation,
            "decision_history": decision_history,
        }
        if portfolio_context is not None and accepts("portfolio_context"):
            kwargs["portfolio_context"] = portfolio_context
        if rejection_feedback and accepts("rejection_feedback"):
            kwargs["rejection_feedback"] = rejection_feedback
        proposed = await proposer(**kwargs)
        batch = (
            proposed
            if isinstance(proposed, ProposalBatch)
            else ProposalBatch(plans=tuple(proposed or ()))
        )
        # A batch that refused every proposal is an answer, not a silence: falling
        # through to decide() would spend another call and erase the refusals the
        # author needs to see. The deterministic pool carries the week instead.
        if batch.plans or batch.rejections:
            return batch
    plan, _ = await executive.decide(run=run, observation=observation)
    return ProposalBatch(plans=(plan,))


def _with_proposal_lineage(
    selected_plan: ActionPlan,
    proposal: ActionPlan | None,
) -> ActionPlan:
    if proposal is None:
        return selected_plan
    return selected_plan.model_copy(
        update={
            "proposal_kind": proposal.proposal_kind,
            "hypothesis_id": proposal.hypothesis_id,
            "experiment_control": proposal.experiment_control,
            "evidence_regime": proposal.evidence_regime,
            "experiment_expires_week": proposal.experiment_expires_week,
            "experiment_program": proposal.experiment_program,
        },
        deep=True,
    )


# The width of ActionPlan.strategy_family; a disambiguated identity has to fit it.
MAX_STRATEGY_FAMILY_LENGTH = 80


def _ballot_signature(plan: ActionPlan) -> str:
    """What makes two proposals the same entry on one week's ballot."""

    program = plan.experiment_program
    return json.dumps(
        {
            "proposal_kind": plan.proposal_kind,
            "hypothesis_id": plan.hypothesis_id,
            "experiment_control": plan.experiment_control,
            "commands": [
                {"tool": command.tool, "arguments": command.arguments}
                for command in plan.commands
            ],
            "program": program.model_dump(mode="json") if program is not None else None,
        },
        sort_keys=True,
        default=str,
    )


def _with_distinct_candidate_identities(
    plans: Sequence[ActionPlan],
) -> tuple[tuple[ActionPlan, ...], tuple[dict[str, str], ...]]:
    """Give every plan on the week's ballot its own candidate identity.

    Two plans share a strategy family whenever the Executive re-proposes the
    experiment its own commitment is already running. An evaluation set
    addresses candidates by identity, so a collision makes one of them
    invisible: card, lineage and choice all resolve to whichever plan was
    written last. A repeat of an entry already on the ballot is dropped; a plan
    that differs keeps its place under a distinct identity and is judged on its
    own merits.
    """

    kept: list[ActionPlan] = []
    signatures: dict[str, str] = {}
    diagnostics: list[dict[str, str]] = []
    for plan in plans:
        family = plan.strategy_family
        if family not in signatures:
            signatures[family] = _ballot_signature(plan)
            kept.append(plan)
            continue
        if signatures[family] == _ballot_signature(plan):
            diagnostics.append({"candidate_id": family, "resolution": "dropped"})
            continue
        alternative = family
        ordinal = 2
        while alternative in signatures:
            suffix = f"__alt{ordinal}"
            alternative = f"{family[: MAX_STRATEGY_FAMILY_LENGTH - len(suffix)]}{suffix}"
            ordinal += 1
        renamed = plan.model_copy(update={"strategy_family": alternative})
        signatures[alternative] = _ballot_signature(renamed)
        kept.append(renamed)
        diagnostics.append(
            {
                "candidate_id": family,
                "resolution": "renamed",
                "assigned_candidate_id": alternative,
            }
        )
    return tuple(kept), tuple(diagnostics)


def _stage_experiment_plan(plan: ActionPlan, *, current_week: int) -> ActionPlan:
    """Materialize the command phase encoded by a delayed experiment program."""

    program = plan.experiment_program
    staged_development = (
        program is not None
        and program.control in {"development", "targeted_development"}
        and program.acquisition_probe_weekly_spend > 0.0
    )
    if not staged_development or program is None:
        return plan
    build_phase = current_week < program.minimum_maturity_week
    commands: list[ActionCommand] = []
    for command in plan.commands:
        arguments = command.arguments
        if command.tool == "set_daily_spend" and program.control == "development":
            arguments = {
                **arguments,
                "development": (
                    program.treatment_value if build_phase else program.baseline_value
                )
                / 7.0,
            }
        elif command.tool == "set_targeted_dev_spend" and (
            program.control == "targeted_development"
        ):
            arguments = {
                "targeted_spend": (
                    program.treatment_targeted_development
                    if build_phase
                    else program.baseline_targeted_development
                )
            }
        elif command.tool == "set_targeted_ad_spend":
            arguments = {
                "targeted_spend": (
                    program.baseline_targeted_ad_spend
                    if build_phase
                    else {
                        str(program.target_channel): {
                            str(program.target_segment): (
                                program.acquisition_probe_weekly_spend / 7.0
                            )
                        }
                    }
                )
            }
        commands.append(
            command.model_copy(
                update={
                    "arguments": arguments,
                    "idempotency_key": (
                        f"{command.idempotency_key}:stage-week-{current_week}"
                    ),
                },
                deep=True,
            )
        )
    return plan.model_copy(update={"commands": commands}, deep=True)


def _active_experiment_plan(
    decisions: tuple[DecisionRecord, ...],
    *,
    current_week: int,
) -> ActionPlan | None:
    """Return the latest still-binding intervention, never an older abandoned one."""

    committed = tuple(
        decision
        for decision in sorted(decisions, key=lambda item: (item.week, str(item.id)))
        if decision.actual_outcome is not None
    )
    if not committed:
        return None
    plan = committed[-1].action_plan
    program = plan.experiment_program
    if program is None or current_week >= program.maximum_end_week:
        return None
    elapsed_cost = max(0.0, program.treatment_value - program.baseline_value) * max(
        0,
        current_week - program.started_week,
    )
    if elapsed_cost >= program.maximum_cumulative_downside:
        return None
    return _stage_experiment_plan(plan, current_week=current_week)


def _targeted_development_scale_variants(
    plans: tuple[ActionPlan, ...],
) -> tuple[ActionPlan, ...]:
    """Let the world model price magnitudes around an Executive causal hypothesis.

    Gemini still chooses the segment, channel, hypothesis, and initial magnitude. The
    deterministic planner only creates bounded counterfactual magnitudes, just as it
    evaluates multiple stochastic outcomes for every other continuous control.
    """

    expanded: list[ActionPlan] = []
    for plan in plans:
        program = plan.experiment_program
        if program is None or program.control != "targeted_development":
            expanded.append(plan)
            continue
        treatment = program.treatment_targeted_development
        if len(treatment) != 1:
            expanded.append(plan)
            continue
        segment, proposed_daily = next(iter(treatment.items()))
        proposed_daily = float(proposed_daily)
        if proposed_daily <= 0.0:
            expanded.append(plan)
            continue
        duration_weeks = max(1, program.minimum_maturity_week - program.started_week)
        baseline_daily = program.baseline_value / 7.0
        affordable_daily = min(
            10_000.0,
            baseline_daily
            + program.maximum_cumulative_downside / (7.0 * duration_weeks),
        )
        bounded_proposed_daily = min(proposed_daily, affordable_daily)
        magnitudes = tuple(
            sorted(
                {
                    bounded_proposed_daily,
                    min(affordable_daily, bounded_proposed_daily * 4.0),
                    affordable_daily,
                }
            )
        )
        for index, daily in enumerate(magnitudes, start=1):
            scaled_program = program.model_copy(
                update={
                    "treatment_value": daily * 7.0,
                    "treatment_targeted_development": {segment: daily},
                },
                deep=True,
            )
            commands = [
                command.model_copy(
                    update={
                        "arguments": {"targeted_spend": {segment: daily}},
                    },
                    deep=True,
                )
                if command.tool == "set_targeted_dev_spend"
                else command
                for command in plan.commands
            ]
            suffix = f"_m{index}"
            strategy = f"{plan.strategy_family[: 80 - len(suffix)]}{suffix}"
            expanded.append(
                plan.model_copy(
                    update={
                        "name": f"{plan.name[:112]}{suffix}",
                        "strategy_family": strategy,
                        "commands": commands,
                        "experiment_program": scaled_program,
                        "rationale": (
                            f"{plan.rationale} Planner magnitude counterfactual: "
                            f"{daily:.2f} USD/day for {duration_weeks} weeks; the "
                            "executable world model, not a rule, must price this scale."
                        )[:4_000],
                    },
                    deep=True,
                )
            )
    return tuple(expanded)


def _experiment_reversion_action(
    decisions: tuple[DecisionRecord, ...],
    *,
    state: SimulationState,
) -> SimulationAction | None:
    committed = tuple(
        decision
        for decision in sorted(decisions, key=lambda item: (item.week, str(item.id)))
        if decision.actual_outcome is not None
    )
    if not committed:
        return None
    program = committed[-1].action_plan.experiment_program
    if program is None:
        return None
    if program.is_standing_commitment:
        # A strategy that reaches the end of its window does not roll back: the
        # configuration it established is simply how the company now operates.
        # Rolling it back by default is what made every held direction look like
        # a probe that failed.
        return None
    elapsed_cost = max(0.0, program.treatment_value - program.baseline_value) * max(
        0,
        state.week - program.started_week,
    )
    budget_exhausted = elapsed_cost >= program.maximum_cumulative_downside
    if state.week != program.maximum_end_week and not budget_exhausted:
        return None
    updates: dict[str, object] = {}
    if program.control == "price":
        prices = program.baseline_configuration.get("prices", {})
        if isinstance(prices, dict) and all(
            isinstance(prices.get(plan), int | float) for plan in "ABC"
        ):
            updates["price_per_customer_weekly"] = (
                fmean(float(prices[plan]) for plan in "ABC") * 7.0 / 30.0
            )
    elif program.control == "tier":
        tiers = program.baseline_configuration.get("model_tiers", {})
        if isinstance(tiers, dict):
            for plan in "ABC":
                value = tiers.get(plan)
                if isinstance(value, int | float):
                    updates[f"model_tier_{plan.lower()}"] = int(value)
    elif program.control == "quota":
        quotas = program.baseline_configuration.get("usage_quotas", {})
        if isinstance(quotas, dict):
            for plan in "ABC":
                value = quotas.get(plan)
                if isinstance(value, int | float):
                    updates[f"usage_quota_{plan.lower()}"] = float(value)
    elif program.control == "marketing":
        updates["marketing_spend"] = program.baseline_value
    elif program.control == "development":
        baseline_ads = tuple(
            TargetedAdAllocation(
                channel=channel,
                segment=segment,
                daily_spend=float(amount),
            )
            for channel, groups in sorted(program.baseline_targeted_ad_spend.items())
            for segment, amount in sorted(groups.items())
        )
        updates.update(
            {
                "development_spend": program.baseline_value,
                "marketing_spend": 7.0
                * sum(allocation.daily_spend for allocation in baseline_ads),
                "targeted_ad_allocations": baseline_ads,
            }
        )
    elif program.control == "targeted_development":
        baseline_ads = tuple(
            TargetedAdAllocation(
                channel=channel,
                segment=segment,
                daily_spend=float(amount),
            )
            for channel, groups in sorted(program.baseline_targeted_ad_spend.items())
            for segment, amount in sorted(groups.items())
        )
        updates.update(
            {
                "marketing_spend": 7.0
                * sum(allocation.daily_spend for allocation in baseline_ads),
                "targeted_ad_allocations": baseline_ads,
                "targeted_development_allocations": tuple(
                    TargetedDevelopmentAllocation(
                        segment=segment,
                        daily_spend=float(amount),
                    )
                    for segment, amount in sorted(
                        program.baseline_targeted_development.items()
                    )
                ),
            }
        )
    else:
        updates["lead_promotion_monthly"] = program.baseline_value
    if program.control in {"price", "tier", "quota", "lead_promotion"}:
        # Offer-side probes hold measurement exposure identical in both arms, so
        # reverting the control alone would silently keep that exposure running.
        pre_experiment_ads = program.baseline_configuration.get(
            "pre_experiment_targeted_ad_spend"
        )
        pre_experiment_marketing = program.baseline_configuration.get(
            "pre_experiment_weekly_marketing_spend"
        )
        if isinstance(pre_experiment_marketing, int | float):
            baseline_ads = tuple(
                TargetedAdAllocation(
                    channel=channel,
                    segment=segment,
                    daily_spend=float(amount),
                )
                for channel, groups in sorted(
                    (pre_experiment_ads or {}).items()
                )
                if isinstance(groups, dict)
                for segment, amount in sorted(groups.items())
                if isinstance(amount, int | float)
            )
            updates.update(
                {
                    "marketing_spend": (
                        7.0 * sum(allocation.daily_spend for allocation in baseline_ads)
                        if baseline_ads
                        else float(pre_experiment_marketing)
                    ),
                    "targeted_ad_allocations": baseline_ads,
                }
            )
    # Rolling back the treatment must not disturb configuration the experiment
    # never touched, so every other control is carried over from the state.
    return _operating_action_from_state(
        state, name=f"experiment_revert_{program.commitment_id}"
    ).model_copy(update=updates)


def _matured_commitment_program(
    decisions: tuple[DecisionRecord, ...],
    *,
    state: SimulationState,
) -> ExperimentProgram | None:
    """The committed programme whose window closes this week, if any."""

    committed = tuple(
        decision
        for decision in sorted(decisions, key=lambda item: (item.week, str(item.id)))
        if decision.actual_outcome is not None
    )
    if not committed:
        return None
    program = committed[-1].action_plan.experiment_program
    if program is None or state.week != program.maximum_end_week:
        return None
    return program


def _operating_action_from_state(
    state: SimulationState, *, name: str
) -> SimulationAction:
    """The company's current configuration expressed as a plain action.

    Every control the caller does not deliberately change is carried over, so a
    rollback or an adoption cannot silently reset an unrelated setting.
    """

    return SimulationAction(
        name=name,
        price_per_customer_weekly=state.effective_catalog_price_per_customer_weekly,
        marketing_spend=state.marketing_spend,
        development_spend=state.development_spend,
        operations_spend=state.operations_spend,
        model_tier_a=state.model_tier_a,
        model_tier_b=state.model_tier_b,
        model_tier_c=state.model_tier_c,
        usage_quota_a=state.usage_quota_a,
        usage_quota_b=state.usage_quota_b,
        usage_quota_c=state.usage_quota_c,
        capacity_tier=state.capacity_tier,
        recurring_promotion_monthly=state.recurring_promotion_monthly,
        ads_strength=state.ads_strength,
        targeted_ops_spend=state.targeted_ops_spend,
        lead_promotion_monthly=state.lead_promotion_monthly,
    )


def _active_commitment_program(
    decisions: tuple[DecisionRecord, ...],
    *,
    state: SimulationState,
) -> ExperimentProgram | None:
    """The committed programme whose window covers this week, if any."""

    for decision in sorted(
        decisions, key=lambda item: (item.week, str(item.id)), reverse=True
    ):
        program = decision.action_plan.experiment_program
        if program is None:
            continue
        if program.started_week <= state.week <= program.maximum_end_week:
            return program
        return None
    return None


def _experiment_adoption_action(
    decisions: tuple[DecisionRecord, ...],
    *,
    state: SimulationState,
) -> SimulationAction | None:
    """Keep a matured treatment as the new operating configuration.

    A probe that earned its result should be able to become the way the company
    operates. Without this option every experiment ends in a rollback and the
    knowledge it bought is discarded.
    """

    program = _matured_commitment_program(decisions, state=state)
    if program is None or not program.treatment_configuration:
        return None
    treatment = program.treatment_configuration
    updates: dict[str, object] = {}
    prices = treatment.get("prices")
    if isinstance(prices, dict) and all(
        isinstance(prices.get(plan), int | float) for plan in "ABC"
    ):
        updates["price_per_customer_weekly"] = (
            fmean(float(prices[plan]) for plan in "ABC") * 7.0 / 30.0
        )
    tiers = treatment.get("model_tiers")
    if isinstance(tiers, dict):
        for plan in "ABC":
            value = tiers.get(plan)
            if isinstance(value, int | float):
                updates[f"model_tier_{plan.lower()}"] = int(value)
    quotas = treatment.get("usage_quotas")
    if isinstance(quotas, dict):
        for plan in "ABC":
            value = quotas.get(plan)
            if isinstance(value, int | float):
                updates[f"usage_quota_{plan.lower()}"] = float(value)
    marketing = treatment.get("weekly_marketing_spend")
    if isinstance(marketing, int | float):
        updates["marketing_spend"] = float(marketing)
    development = treatment.get("daily_development_spend")
    if isinstance(development, int | float):
        updates["development_spend"] = float(development) * 7.0
    promotion = treatment.get("lead_promotion_monthly")
    if isinstance(promotion, int | float):
        updates["lead_promotion_monthly"] = float(promotion)
    targeted = treatment.get("targeted_development_daily")
    if isinstance(targeted, dict) and targeted:
        updates["targeted_development_allocations"] = tuple(
            TargetedDevelopmentAllocation(segment=segment, daily_spend=float(amount))
            for segment, amount in sorted(targeted.items())
            if isinstance(amount, int | float)
        )
    # No expiry windows: an adopted treatment is simply how the company now runs.
    return _operating_action_from_state(
        state, name=f"experiment_adopt_{program.commitment_id}"
    ).model_copy(update=updates)


@dataclass(frozen=True, slots=True)
class RejectedCandidate:
    """One generated action that the independent economic gate refused."""

    strategy: str
    violation_codes: tuple[str, ...]
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class CandidatePoolFeasibility:
    """Evidence about how much of the generated action pool survived the economic gate."""

    generated_count: int
    feasible_count: int
    rejected: tuple[RejectedCandidate, ...]
    warning_codes: tuple[str, ...]

    @property
    def degraded(self) -> bool:
        return self.feasible_count < self.generated_count

    def as_payload(self) -> dict[str, object]:
        return {
            "generated_count": self.generated_count,
            "feasible_count": self.feasible_count,
            "warning_codes": list(self.warning_codes),
            "rejected": [
                {
                    "strategy": candidate.strategy,
                    "violation_codes": list(candidate.violation_codes),
                    **({"detail": candidate.detail} if candidate.detail else {}),
                }
                for candidate in self.rejected
            ],
        }


@dataclass(frozen=True, slots=True)
class WeeklyPlanningResult:
    action_plan: ActionPlan
    forecasts: CashForecasts
    search: StrategySearchResult
    candidate_records: tuple[CandidateEvaluationRecord, ...]
    sensitivities: tuple[CashSensitivityEstimate, ...]
    prompt_version: str
    assumptions: tuple[str, ...]
    evidence_references: tuple[str, ...]
    feasibility: CandidatePoolFeasibility
    exploration_admission: ExplorationAdmission
    # Information purchases authorized for this week. They change no company
    # configuration, so they travel beside the selected plan rather than inside it.
    information_requests: tuple[InformationRequest, ...] = ()
    # Read-only questions the Executive asked; their answers reach next week.
    data_queries: tuple[str, ...] = ()
    # The Executive's own note to itself, carried verbatim into the next week.
    journal: str = ""


def exploration_memory_from_decisions(
    decisions: tuple[DecisionRecord, ...],
    *,
    current_week: int,
) -> ExplorationMemory:
    committed = tuple(
        decision
        for decision in sorted(decisions, key=lambda item: (item.week, str(item.id)))
        if decision.actual_outcome is not None
        and (
            decision.action_plan.strategy_family.startswith(
                "controlled_exploration_"
            )
            or decision.action_plan.strategy_family.startswith(
                "executive_experiment_"
            )
        )
    )
    attempted = tuple(
        dict.fromkeys(decision.action_plan.strategy_family for decision in committed)
    )
    attempted_hypothesis_keys = tuple(
        dict.fromkeys(
            experiment_hypothesis_key(
                decision.action_plan.hypothesis_id,
                decision.action_plan.experiment_control,
                decision.action_plan.evidence_regime,
            )
            for decision in committed
            if decision.action_plan.hypothesis_id is not None
            and decision.action_plan.experiment_control is not None
            and decision.action_plan.evidence_regime is not None
        )
    )
    revert_marketing_spend: float | None = None
    revert_development_spend: float | None = None
    revert_lead_promotion_monthly: float | None = None
    if committed and committed[-1].week == current_week - 1:
        latest = committed[-1]
        selected = next(
            (
                candidate
                for candidate in latest.candidate_evaluations
                if candidate.strategy == latest.action_plan.strategy_family
            ),
            None,
        )
        if selected is not None:
            marketing = selected.action_parameters.get(
                "marketing_spend_after_experiment"
            )
            development = selected.action_parameters.get(
                "development_spend_after_experiment"
            )
            lead_promotion = selected.action_parameters.get(
                "lead_promotion_after_experiment"
            )
            if isinstance(marketing, int | float) and marketing >= 0:
                revert_marketing_spend = float(marketing)
            if isinstance(development, int | float) and development >= 0:
                revert_development_spend = float(development)
            if isinstance(lead_promotion, int | float) and lead_promotion >= 0:
                revert_lead_promotion_monthly = float(lead_promotion)
    regime_totals: dict[int, dict[str, float | int]] = {}
    for decision in decisions:
        outcome = decision.actual_outcome
        if outcome is None:
            continue
        metrics = outcome.metrics
        quality = metrics.get("product_quality")
        leads = metrics.get("weekly_leads")
        conversions = metrics.get("weekly_conversions")
        if not all(
            isinstance(value, int | float)
            for value in (quality, leads, conversions)
        ):
            continue
        quality_band = min(9, max(0, int(float(quality) * 10.0)))
        aggregate = regime_totals.setdefault(
            quality_band,
            {"leads": 0.0, "conversions": 0.0, "weeks": 0},
        )
        aggregate["leads"] = float(aggregate["leads"]) + max(0.0, float(leads))
        aggregate["conversions"] = float(aggregate["conversions"]) + max(
            0.0, float(conversions)
        )
        aggregate["weeks"] = int(aggregate["weeks"]) + 1
    regime_evidence = tuple(
        FunnelRegimeEvidence(
            quality_band=quality_band,
            leads=float(values["leads"]),
            conversions=float(values["conversions"]),
            weeks=int(values["weeks"]),
        )
        for quality_band, values in sorted(regime_totals.items())
    )
    return ExplorationMemory(
        attempted_strategies=attempted,
        attempted_hypothesis_keys=attempted_hypothesis_keys,
        revert_marketing_spend=revert_marketing_spend,
        revert_development_spend=revert_development_spend,
        revert_lead_promotion_monthly=revert_lead_promotion_monthly,
        funnel_regime_evidence=regime_evidence,
    )


def _with_refusals(
    feasibility: CandidatePoolFeasibility,
    *,
    rejections: tuple[ProposalRejection, ...] = (),
    dropped: tuple[RejectedCandidate, ...] = (),
    warning_codes: tuple[str, ...] = (),
) -> CandidatePoolFeasibility:
    """Fold refusals made before the economic gate into the week's pool evidence.

    A proposal refused while it was being built never reaches the gate, so without
    this it would vanish from the record. Counting it as generated-but-not-feasible
    keeps the pool visibly degraded and carries the named reason into the audit
    event the week already emits.
    """

    extra = tuple(
        RejectedCandidate(
            strategy=rejection.name,
            violation_codes=rejection.veto_codes,
            detail=rejection.detail or None,
        )
        for rejection in rejections
    ) + dropped
    if not extra and not warning_codes:
        return feasibility
    merged_warnings = list(feasibility.warning_codes)
    for code in warning_codes:
        if code not in merged_warnings:
            merged_warnings.append(code)
    return CandidatePoolFeasibility(
        generated_count=feasibility.generated_count + len(extra),
        feasible_count=feasibility.feasible_count,
        rejected=feasibility.rejected + extra,
        warning_codes=tuple(merged_warnings),
    )


def partition_feasible_actions(
    state: SimulationState,
    actions: tuple[SimulationAction, ...],
) -> tuple[tuple[SimulationAction, ...], CandidatePoolFeasibility]:
    """Split a generated pool into feasible actions and auditable rejection evidence."""

    feasible: list[SimulationAction] = []
    rejected: list[RejectedCandidate] = []
    warning_codes: list[str] = []
    for action in actions:
        report = evaluate_simulation_action(state, action)
        for warning in report.warnings:
            if warning.code.value not in warning_codes:
                warning_codes.append(warning.code.value)
        violation_codes = tuple(
            dict.fromkeys(violation.code.value for violation in report.violations)
        )
        if report.valid:
            feasible.append(action)
        elif is_operational_candidate(action.name):
            # Continuing as the company already runs is the floor the week stands
            # on. Refusing it would leave nothing to decide between, so its
            # violations are carried as warnings instead of removing the floor.
            feasible.append(action)
            for code in violation_codes:
                if code not in warning_codes:
                    warning_codes.append(code)
        else:
            rejected.append(
                RejectedCandidate(
                    strategy=action.name,
                    violation_codes=violation_codes,
                )
            )
    feasibility = CandidatePoolFeasibility(
        generated_count=len(actions),
        feasible_count=len(feasible),
        rejected=tuple(rejected),
        warning_codes=tuple(warning_codes),
    )
    return tuple(feasible), feasibility


def simulation_state_from_observation(observation: ObservationSnapshot) -> SimulationState:
    metrics = observation.metrics

    def number(name: str, default: float) -> float:
        value = metrics.get(name, default)
        return float(value) if isinstance(value, int | float) else default

    def optional_number(name: str) -> float | None:
        value = metrics.get(name)
        return max(0.0, float(value)) if isinstance(value, int | float) else None

    customers = max(0.0, number("active_customers", number("customers", 1_000.0)))
    revenue = max(0.0, number("weekly_revenue", number("revenue", 50_000.0)))
    inferred_price = (
        revenue / customers
        if customers > 0 and revenue > 0
        else INITIAL_WEEKLY_AVERAGE_PRICE
    )
    observed_arpu = number("price_per_customer_weekly", number("pricing", inferred_price))
    realized_arpu = observed_arpu if observed_arpu > 0 else inferred_price
    tier_prices = tuple(number(name, 0.0) for name in ("price_a", "price_b", "price_c"))
    catalog_price = (
        PeriodicMoney(amount=fmean(tier_prices), period=RatePeriod.MONTH_30_DAY)
        .per(RatePeriod.WEEK)
        .amount
        if all(price > 0 for price in tier_prices)
        else realized_arpu
    )
    # The catalog's range, not just its average: the cheapest plan is what a
    # customer carries by default and the dearest is what better delivery can
    # earn. Collapsing them to a mean makes quality look like a volume lever.
    def _weekly(monthly: float) -> float | None:
        if monthly <= 0.0:
            return None
        return (
            PeriodicMoney(amount=monthly, period=RatePeriod.MONTH_30_DAY)
            .per(RatePeriod.WEEK)
            .amount
        )

    catalog_entry = _weekly(min(tier_prices)) if all(tier_prices) else None
    catalog_premium = _weekly(max(tier_prices)) if all(tier_prices) else None
    capacity = max(customers + 1.0, number("capacity", max(1_000.0, customers * 1.5)))
    return SimulationState(
        week=observation.day // 7,
        cash=observation.cash,
        revenue_weekly=revenue,
        customers=customers,
        churn_rate=max(0.0, min(1.0, number("churn_rate", number("churn", 0.04)))),
        price_per_customer_weekly=realized_arpu,
        catalog_price_per_customer_weekly=catalog_price,
        catalog_price_entry_weekly=catalog_entry,
        catalog_price_premium_weekly=catalog_premium,
        entry_price_monthly=(tier_prices[0] if tier_prices[0] > 0 else None),
        lead_promotion_monthly=max(0.0, number("lead_promotion_monthly", 0.0)),
        price_realization_ratio=realized_arpu / catalog_price,
        weekly_acquisition=max(
            0.0,
            number(
                "weekly_conversions",
                number("weekly_acquisition", max(50.0, customers * 0.05)),
            ),
        ),
        weekly_leads=max(0.0, number("weekly_leads", 0.0)),
        weekly_conversions=max(0.0, number("weekly_conversions", 0.0)),
        weekly_lost_leads=max(0.0, number("weekly_lost_leads", 0.0)),
        total_leads=max(0.0, number("total_leads", 0.0)),
        total_conversions=max(0.0, number("total_conversions", 0.0)),
        total_lost_leads=max(0.0, number("total_lost_leads", 0.0)),
        marketing_spend=max(0.0, number("marketing_spend", 3_500.0)),
        marketing_spend_social_media_weekly=max(
            0.0, number("marketing_spend_social_media_weekly", 0.0)
        ),
        marketing_spend_search_ads_weekly=max(
            0.0, number("marketing_spend_search_ads_weekly", 0.0)
        ),
        marketing_spend_linkedin_weekly=max(
            0.0, number("marketing_spend_linkedin_weekly", 0.0)
        ),
        marketing_spend_content_marketing_weekly=max(
            0.0, number("marketing_spend_content_marketing_weekly", 0.0)
        ),
        marketing_spend_referral_program_weekly=max(
            0.0, number("marketing_spend_referral_program_weekly", 0.0)
        ),
        development_spend=max(0.0, number("development_spend", 1_750.0)),
        targeted_development_spend=max(
            0.0, number("targeted_development_spend", 0.0)
        ),
        operations_spend=max(0.0, number("operations_spend", 3_500.0)),
        capacity_spend_weekly=max(0.0, number("capacity_spend_weekly", 595.0)),
        product_quality=max(0.0, min(1.0, number("product_quality", 0.5))),
        capacity=capacity,
        reputation=max(0.0, min(1.0, number("reputation", 0.5))),
        operating_cost_per_customer_weekly=max(
            0.0,
            number("operating_cost_per_customer_weekly", 0.0),
        ),
        model_tier_a=round(number("model_tier_a", 1.0)),
        model_tier_b=round(number("model_tier_b", 1.0)),
        model_tier_c=round(number("model_tier_c", 1.0)),
        # An observation that does not report allowances carries no allowance
        # information; that is not the same as a zero allowance.
        usage_quota_a=optional_number("usage_quota_a"),
        usage_quota_b=optional_number("usage_quota_b"),
        usage_quota_c=optional_number("usage_quota_c"),
        daily_usage_per_customer=max(0.0, number("daily_usage_per_customer", 0.0)),
        estimated_usage_demand_per_day=optional_number(
            "estimated_usage_demand_per_day"
        ),
        # Purchased participation floors: absent until research is bought.
        measured_quality_floor_individual=optional_number(
            "measured_quality_floor_individual"
        ),
        measured_quality_floor_enterprise=optional_number(
            "measured_quality_floor_enterprise"
        ),
        revealed_quality_bar_lower_bound=optional_number(
            "revealed_quality_bar_lower_bound"
        ),
        capacity_tier=max(0, round(number("capacity_tier", 0.0))),
        recurring_promotion_monthly=max(
            0.0, number("recurring_promotion_monthly", 0.0)
        ),
        ads_strength=min(1.0, max(0.0, number("ads_strength", 0.0))),
        ads_revenue_weekly=max(0.0, number("ads_revenue_weekly", 0.0)),
        targeted_ops_spend=max(0.0, number("targeted_ops_spend", 0.0)),
        social_posts_weekly=max(0.0, number("social_posts_weekly", 0.0)),
        active_seats=max(0.0, number("active_seats", 0.0)),
        open_enterprise_threads=max(0.0, number("open_enterprise_threads", 0.0)),
        enterprise_revenue_weekly=max(0.0, number("enterprise_revenue_weekly", 0.0)),
        open_enterprise_seats=open_enterprise_seats_from_inbox(
            str(metrics.get("enterprise_inbox") or "")
        ),
        research_catalog=research_catalog_from_metrics(
            str(metrics.get("research_catalog_json") or "[]")
        ),
        research_tiers_in_progress=research_in_progress_from_metrics(
            str(metrics.get("research_catalog_json") or "[]")
        ),
        # A tier the environment reports as running was started by an earlier,
        # separate decision; this rollout must neither restart nor re-charge it.
        research_tiers_started=tuple(
            item.tier
            for item in research_in_progress_from_metrics(
                str(metrics.get("research_catalog_json") or "[]")
            )
        ),
    )


def research_catalog_from_metrics(payload: str) -> tuple[ResearchTierFacts, ...]:
    """The environment's own R&D price list, as the adapter observed it.

    Observed facts flow into the state that forecasts candidates, so a
    programme's exact charge and published means replace generic priors the
    moment they are read. Malformed entries are skipped, not fatal, and an
    empty list means unread — the state simply carries no catalog.
    """

    try:
        entries = json.loads(payload)
    except (TypeError, ValueError):
        return ()
    facts: list[ResearchTierFacts] = []
    known = ("tier", "cost", "mean_weeks", "mean_quality_boost")
    for entry in entries if isinstance(entries, list) else ():
        if not isinstance(entry, dict):
            continue
        try:
            facts.append(
                ResearchTierFacts.model_validate(
                    {key: entry[key] for key in known if key in entry}
                )
            )
        except ValidationError:
            continue
    return tuple(facts)


def research_in_progress_from_metrics(payload: str) -> tuple[PendingResearch, ...]:
    """Tiers the environment reports as already maturing.

    Seeding them into the forecast state keeps a candidate that repeats the
    action from restarting — and re-charging — a programme that is already
    running. Remaining time is not published, so the listed mean stands in.
    """

    try:
        entries = json.loads(payload)
    except (TypeError, ValueError):
        return ()
    pending: list[PendingResearch] = []
    for entry in entries if isinstance(entries, list) else ():
        if not isinstance(entry, dict) or not entry.get("in_progress"):
            continue
        try:
            pending.append(
                PendingResearch(
                    tier=int(entry["tier"]),
                    weeks_remaining=max(1, min(52, int(entry.get("mean_weeks") or 4))),
                )
            )
        except (KeyError, TypeError, ValueError, ValidationError):
            continue
    return tuple(pending)


def open_enterprise_seats_from_inbox(inbox: str) -> float:
    """Seats waiting in open negotiation threads, from the observed inbox line.

    The inbox is a compact ``customer:seats:day`` list; malformed entries are
    skipped rather than failing the week.
    """

    total = 0.0
    for entry in inbox.split(","):
        parts = entry.split(":")
        if len(parts) < 2:
            continue
        try:
            total += max(0.0, float(parts[1]))
        except ValueError:
            continue
    return total


def simulation_action_from_action_plan(
    plan: ActionPlan,
    state: SimulationState,
    *,
    name: str | None = None,
) -> SimulationAction:
    price = state.effective_catalog_price_per_customer_weekly
    marketing_spend = state.marketing_spend
    development_spend = state.development_spend
    operations_spend = state.operations_spend
    model_tier_a = state.model_tier_a
    model_tier_b = state.model_tier_b
    model_tier_c = state.model_tier_c
    lead_promotion_monthly = state.lead_promotion_monthly
    usage_quota_a = state.usage_quota_a
    usage_quota_b = state.usage_quota_b
    usage_quota_c = state.usage_quota_c
    capacity_tier = state.capacity_tier
    recurring_promotion_monthly = state.recurring_promotion_monthly
    ads_strength = state.ads_strength
    targeted_ops_spend = state.targeted_ops_spend
    social_posts = 0
    research_project_tier: int | None = None
    targeted_ad_allocations: tuple[TargetedAdAllocation, ...] = ()
    targeted_development_allocations: tuple[TargetedDevelopmentAllocation, ...] = ()
    for command in plan.commands:
        if command.tool == "set_prices":
            prices = [float(command.arguments[key]) for key in ("A", "B", "C")]
            price = PeriodicMoney(
                amount=fmean(prices),
                period=RatePeriod.MONTH_30_DAY,
            ).per(RatePeriod.WEEK).amount
        elif command.tool == "set_daily_spend":
            operations_spend = float(command.arguments.get("operations", 0.0)) * 7
            development_spend = float(command.arguments.get("development", 0.0)) * 7
        elif command.tool == "set_targeted_ad_spend":
            raw_spend = command.arguments.get("targeted_spend", {})
            if isinstance(raw_spend, dict):
                targeted_ad_allocations = tuple(
                    TargetedAdAllocation(
                        channel=str(channel),
                        segment=str(segment),
                        daily_spend=float(amount),
                    )
                    for channel, groups in sorted(raw_spend.items())
                    if isinstance(groups, dict)
                    for segment, amount in sorted(groups.items())
                    if isinstance(amount, int | float)
                )
                marketing_spend = 7.0 * sum(
                    allocation.daily_spend
                    for allocation in targeted_ad_allocations
                )
        elif command.tool == "set_targeted_dev_spend":
            raw_spend = command.arguments.get("targeted_spend", {})
            if isinstance(raw_spend, dict):
                targeted_development_allocations = tuple(
                    TargetedDevelopmentAllocation(
                        segment=str(segment),
                        daily_spend=float(amount),
                    )
                    for segment, amount in sorted(raw_spend.items())
                    if isinstance(amount, int | float)
                )
        elif command.tool == "set_model_tiers":
            model_tier_a = int(command.arguments["A"])
            model_tier_b = int(command.arguments["B"])
            model_tier_c = int(command.arguments["C"])
        elif command.tool == "set_lead_promotion":
            lead_promotion_monthly = float(
                command.arguments.get("global_promotion", 0.0)
            )
        elif command.tool == "set_usage_quotas":
            usage_quota_a = float(command.arguments["A"])
            usage_quota_b = float(command.arguments["B"])
            usage_quota_c = float(command.arguments["C"])
        elif command.tool == "set_capacity_tier":
            capacity_tier = int(command.arguments["tier"])
        elif command.tool == "set_promotion":
            recurring_promotion_monthly = float(
                command.arguments.get("global_promotion", 0.0)
            )
        elif command.tool == "set_ads_strength":
            ads_strength = float(command.arguments.get("global_strength", 0.0))
        elif command.tool == "set_targeted_ops_spend":
            raw_spend = command.arguments.get("targeted_spend", {})
            if isinstance(raw_spend, dict):
                targeted_ops_spend = 7.0 * sum(
                    float(amount)
                    for amount in raw_spend.values()
                    if isinstance(amount, int | float)
                )
        elif command.tool == "post_social_media":
            social_posts = 1
        elif command.tool == "start_research_project":
            research_project_tier = int(command.arguments["tier"])
    return SimulationAction(
        name=name or plan.strategy_family,
        price_per_customer_weekly=max(0.01, price),
        marketing_spend=min(MAX_WEEKLY_SPEND, max(0.0, marketing_spend)),
        targeted_ad_allocations=targeted_ad_allocations,
        development_spend=min(MAX_WEEKLY_SPEND, max(0.0, development_spend)),
        targeted_development_allocations=targeted_development_allocations,
        operations_spend=min(MAX_WEEKLY_SPEND, max(0.0, operations_spend)),
        model_tier_a=model_tier_a,
        model_tier_b=model_tier_b,
        model_tier_c=model_tier_c,
        usage_quota_a=usage_quota_a,
        usage_quota_b=usage_quota_b,
        usage_quota_c=usage_quota_c,
        capacity_tier=capacity_tier,
        recurring_promotion_monthly=recurring_promotion_monthly,
        ads_strength=ads_strength,
        targeted_ops_spend=min(MAX_WEEKLY_SPEND, max(0.0, targeted_ops_spend)),
        social_posts=social_posts,
        research_project_tier=research_project_tier,
        enterprise_engage=plan.enterprise_engage,
        enterprise_target_price_per_seat=plan.enterprise_target_price_per_seat,
        enterprise_floor_price_per_seat=plan.enterprise_floor_price_per_seat,
        enterprise_max_new_seats=plan.enterprise_max_new_seats,
        lead_promotion_monthly=lead_promotion_monthly,
    )


def _executable_candidates(
    state: SimulationState,
    executive_plans: ActionPlan | tuple[ActionPlan, ...],
    exploration_memory: ExplorationMemory | None = None,
    *,
    active_commitment: bool = False,
    deterministic_exploration: bool = True,
) -> tuple[tuple[SimulationAction, ...], tuple[RejectedCandidate, ...]]:
    dropped: list[RejectedCandidate] = []
    plans = (
        (executive_plans,)
        if isinstance(executive_plans, ActionPlan)
        else executive_plans
    )
    generated = [
        action.model_copy(
            update={
                # P0 execution maps price and spend only; do not simulate an unexecuted
                # segment-targeting advantage.
                "segment_focus": 1.0,
            }
        )
        for action in generate_candidate_actions(state, exploration_memory)
    ]
    executive_v2 = len(plans) >= 2
    if executive_v2:
        # The learned Executive owns business opportunity generation. Deterministic
        # code keeps only attributable safety recovery and the single intervention
        # explicitly admitted by the epistemic controller. Hand-authored growth and
        # pricing templates remain a legacy-engine fallback, but admission must not
        # generate a candidate that this translation boundary silently discards.
        generated = [
            action
            for action in generated
            if action.name == "unit_cost_recovery"
            or action.name.startswith("controlled_exploration_")
        ]
    if not deterministic_exploration:
        # Executive authority v2: deterministic code keeps only operational
        # recovery. It never authors an exploration or growth candidate of its
        # own, even when the Executive proposal stage returned nothing.
        generated = [
            action for action in generated if action.name == "unit_cost_recovery"
        ]
    attempted = set(
        (exploration_memory or ExplorationMemory()).attempted_strategies
    )
    attempted_hypotheses = set(
        (exploration_memory or ExplorationMemory()).attempted_hypothesis_keys
    )
    proposals: list[SimulationAction] = []
    for plan in plans:
        # Under executive authority a repeat proposal is answered by the
        # portfolio's falsification veto, which says so in the audit trail.
        # Dropping it here instead would remove a candidate silently.
        if deterministic_exploration and (
            not active_commitment
            and plan.proposal_kind == "experiment"
            and plan.hypothesis_id is not None
            and plan.experiment_control is not None
            and plan.evidence_regime is not None
            and experiment_hypothesis_key(
                plan.hypothesis_id,
                plan.experiment_control,
                plan.evidence_regime,
            )
            in attempted_hypotheses
        ):
            continue
        if (
            deterministic_exploration
            and plan.strategy_family.startswith("executive_experiment_")
            and plan.strategy_family in attempted
            and plan.proposal_kind != "experiment"
        ):
            continue
        proposal = simulation_action_from_action_plan(
            plan,
            state,
            name=plan.strategy_family,
        )
        program = plan.experiment_program
        if plan.strategy_family.startswith(
            ("executive_experiment_marketing_", "controlled_exploration_marketing_")
        ):
            proposal = proposal.model_copy(
                update={
                    "marketing_spend_until_week": (
                        program.maximum_end_week if program is not None else state.week + 1
                    ),
                    "marketing_spend_after_experiment": (
                        program.baseline_value if program is not None else state.marketing_spend
                    ),
                }
            )
        elif plan.strategy_family.startswith("executive_experiment_targeted_development_"):
            if program is None or program.target_segment is None or program.target_channel is None:
                # Refused by name rather than dropped in silence: without a segment
                # and channel the probe has nothing to measure.
                dropped.append(
                    RejectedCandidate(
                        strategy=plan.strategy_family,
                        violation_codes=("measurement_target_missing",),
                    )
                )
                continue
            proposal = proposal.model_copy(
                update={
                    "marketing_spend": program.acquisition_probe_weekly_spend,
                    "targeted_ad_allocations": (
                        TargetedAdAllocation(
                            channel=program.target_channel,
                            segment=program.target_segment,
                            daily_spend=program.acquisition_probe_weekly_spend / 7.0,
                        ),
                    ),
                    "marketing_spend_start_week": program.minimum_maturity_week,
                    "marketing_spend_until_week": program.maximum_end_week,
                    "marketing_spend_after_experiment": 7.0
                    * sum(
                        float(amount)
                        for groups in program.baseline_targeted_ad_spend.values()
                        for amount in groups.values()
                    ),
                    "targeted_development_allocations": tuple(
                        TargetedDevelopmentAllocation(
                            segment=segment,
                            daily_spend=float(amount),
                        )
                        for segment, amount in sorted(
                            program.treatment_targeted_development.items()
                        )
                    ),
                    "targeted_development_spend_until_week": (
                        program.minimum_maturity_week
                    ),
                    "targeted_development_spend_after_experiment": (
                        program.baseline_value
                    ),
                }
            )
        elif plan.strategy_family.startswith(
            ("executive_experiment_development_", "controlled_exploration_development_")
        ):
            updates: dict[str, object] = {
                "development_spend_until_week": (
                    program.maximum_end_week if program is not None else state.week + 1
                ),
                "development_spend_after_experiment": (
                    program.baseline_value if program is not None else state.development_spend
                ),
            }
            if (
                program is not None
                and program.acquisition_probe_weekly_spend > 0.0
                and program.target_segment is not None
                and program.target_channel is not None
            ):
                updates.update(
                    {
                        "marketing_spend": program.acquisition_probe_weekly_spend,
                        "targeted_ad_allocations": (
                            TargetedAdAllocation(
                                channel=program.target_channel,
                                segment=program.target_segment,
                                daily_spend=(
                                    program.acquisition_probe_weekly_spend / 7.0
                                ),
                            ),
                        ),
                        "marketing_spend_start_week": program.minimum_maturity_week,
                        "marketing_spend_until_week": program.maximum_end_week,
                        "marketing_spend_after_experiment": 7.0
                        * sum(
                            float(amount)
                            for groups in program.baseline_targeted_ad_spend.values()
                            for amount in groups.values()
                        ),
                        "development_spend_until_week": program.minimum_maturity_week,
                    }
                )
            proposal = proposal.model_copy(update=updates)
        elif plan.strategy_family.startswith(
            (
                "executive_experiment_lead_promotion_",
                "controlled_exploration_lead_promotion_",
            )
        ):
            proposal = proposal.model_copy(
                update={
                    "lead_promotion_until_week": (
                        program.maximum_end_week if program is not None else state.week + 1
                    ),
                    "lead_promotion_after_experiment": (
                        program.baseline_value
                        if program is not None
                        else state.lead_promotion_monthly
                    ),
                }
            )
        if (
            plan.proposal_kind == "experiment"
            and plan.experiment_control in {"price", "tier", "quota", "lead_promotion"}
            and program is not None
        ):
            pre_experiment_marketing = program.baseline_configuration.get(
                "pre_experiment_weekly_marketing_spend"
            )
            if isinstance(pre_experiment_marketing, int | float) and (
                proposal.marketing_spend > float(pre_experiment_marketing)
            ):
                # Measurement exposure ends with the probe; a forecast must not
                # carry that spend all the way to the terminal horizon.
                proposal = proposal.model_copy(
                    update={
                        "marketing_spend_until_week": program.maximum_end_week,
                        "marketing_spend_after_experiment": float(
                            pre_experiment_marketing
                        ),
                    }
                )
        proposals.append(proposal)
    if active_commitment:
        return tuple(proposals), tuple(dropped)
    candidates = [
        SimulationAction(
            name="continuation",
            price_per_customer_weekly=(
                state.effective_catalog_price_per_customer_weekly
            ),
            marketing_spend=state.marketing_spend,
            development_spend=state.development_spend,
            operations_spend=state.operations_spend,
            model_tier_a=state.model_tier_a,
            model_tier_b=state.model_tier_b,
            model_tier_c=state.model_tier_c,
            lead_promotion_monthly=state.lead_promotion_monthly,
        ),
        *generated,
        *proposals,
    ]
    unique: list[SimulationAction] = []
    signatures: set[str] = set()
    for candidate in candidates:
        signature = candidate.model_dump_json(exclude={"name"})
        if signature in signatures:
            # Two candidates that simulate identically are one choice. Keeping
            # the first preserves the Executive's own naming for it.
            continue
        signatures.add(signature)
        unique.append(candidate)
    return tuple(unique), tuple(dropped)


def _quantile(values: list[float], probability: float) -> float:
    return nearest_rank_quantile(values, probability)


def sandbox_action_payload(
    action: SimulationAction,
    state: SimulationState,
    *,
    horizon_weeks: int,
) -> dict[str, object]:
    """Translate action lifetime without turning persistent controls into one-week tests."""

    if horizon_weeks < 1:
        raise ValueError("sandbox action horizon must be positive")
    marketing_duration_weeks = (
        None
        if action.marketing_spend_until_week is None
        else max(
            1,
            min(horizon_weeks, action.marketing_spend_until_week - state.week),
        )
    )
    development_duration_weeks = (
        None
        if action.development_spend_until_week is None
        else max(
            1,
            min(horizon_weeks, action.development_spend_until_week - state.week),
        )
    )
    experiment_duration_weeks = (
        marketing_duration_weeks or development_duration_weeks or horizon_weeks
    )
    targeted_development_duration_weeks = (
        horizon_weeks
        if action.targeted_development_spend_until_week is None
        else max(
            0,
            min(
                horizon_weeks,
                action.targeted_development_spend_until_week - state.week,
            ),
        )
    )
    channel_spend = {
        "social_media": 0.0,
        "search_ads": 0.0,
        "linkedin": 0.0,
        "content_marketing": 0.0,
        "referral_program": 0.0,
    }
    if action.targeted_ad_allocations:
        for allocation in action.targeted_ad_allocations:
            channel_spend[allocation.channel] += allocation.daily_spend * 7.0
    else:
        observed = {
            "social_media": state.marketing_spend_social_media_weekly,
            "search_ads": state.marketing_spend_search_ads_weekly,
            "linkedin": state.marketing_spend_linkedin_weekly,
            "content_marketing": state.marketing_spend_content_marketing_weekly,
            "referral_program": state.marketing_spend_referral_program_weekly,
        }
        observed_total = sum(observed.values())
        if observed_total > 0:
            channel_spend = {
                channel: action.marketing_spend * amount / observed_total
                for channel, amount in observed.items()
            }
    return {
        **action.model_dump(mode="json"),
        "targeted_development_spend_weekly": 7.0
        * sum(
            allocation.daily_spend
            for allocation in action.targeted_development_allocations
        ),
        "targeted_development_duration_weeks": (
            targeted_development_duration_weeks
        ),
        "targeted_development_spend_after_experiment": (
            state.targeted_development_spend
            if action.targeted_development_spend_after_experiment is None
            else action.targeted_development_spend_after_experiment
        ),
        "marketing_spend_start_after_weeks": (
            0
            if action.marketing_spend_start_week is None
            else max(0, action.marketing_spend_start_week - state.week)
        ),
        **{
            f"marketing_spend_{channel}_weekly": amount
            for channel, amount in channel_spend.items()
        },
        # The generated-model v1 protocol has one duration shared by marketing and
        # development. A persistent control remains exactly representable: use the
        # bounded control's stop and set the persistent control's post-experiment
        # value to its treatment value. Only two genuinely different explicit stops
        # are unrepresentable and rejected above.
        "experiment_duration_weeks": experiment_duration_weeks,
        "development_spend_duration_weeks": (
            development_duration_weeks
            if development_duration_weeks is not None
            else experiment_duration_weeks
        ),
        "marketing_spend_after_experiment": (
            action.marketing_spend
            if action.marketing_spend_after_experiment is None
            else action.marketing_spend_after_experiment
        ),
        "development_spend_after_experiment": (
            action.development_spend
            if action.development_spend_after_experiment is None
            else action.development_spend_after_experiment
        ),
        "lead_promotion_monthly": (
            state.lead_promotion_monthly
            if action.lead_promotion_monthly is None
            else action.lead_promotion_monthly
        ),
        "lead_promotion_duration_weeks": (
            horizon_weeks
            if action.lead_promotion_until_week is None
            else max(
                1,
                min(horizon_weeks, action.lead_promotion_until_week - state.week),
            )
        ),
        "lead_promotion_after_experiment": (
            state.lead_promotion_monthly
            if action.lead_promotion_after_experiment is None
            else action.lead_promotion_after_experiment
        ),
    }


def _rollout_forecasts(
    state: SimulationState,
    action: SimulationAction,
    world_model: WorldModelVersion,
    *,
    n_rollouts: int,
    seed: int,
) -> CashForecasts:
    outcomes = simulate(
        state=state,
        action=action,
        world_model=world_model,
        horizon_weeks=max(FORECAST_WEEKS.values()),
        n_rollouts=n_rollouts,
        seed=seed,
    )
    items = []
    for horizon_days, week in FORECAST_WEEKS.items():
        cash = [outcome.states[week].cash for outcome in outcomes]
        items.append(
            CashForecast(
                horizon_days=horizon_days,
                point=median(cash),
                lower=_quantile(cash, 0.025),
                upper=_quantile(cash, 0.975),
            )
        )
    return CashForecasts(items=items)


def _deterministic_cash_path(
    state: SimulationState,
    action: SimulationAction,
    parameters: dict,
) -> dict[int, float]:
    current = state
    cash_by_week: dict[int, float] = {}
    for week in range(1, max(FORECAST_WEEKS.values()) + 1):
        current = advance_simulation_week(current, action, parameters)
        cash_by_week[week] = current.cash
    return cash_by_week


def estimate_cash_sensitivities(
    state: SimulationState,
    action: SimulationAction,
    world_model: WorldModelVersion,
) -> tuple[CashSensitivityEstimate, ...]:
    base_parameters = {
        parameter.name: parameter.estimate for parameter in world_model.parameters
    }
    base_cash = _deterministic_cash_path(state, action, base_parameters)
    estimates: list[CashSensitivityEstimate] = []
    for parameter in world_model.parameters:
        interval_width = parameter.upper_bound - parameter.lower_bound
        positive_delta = max(1e-4, interval_width * 0.01)
        delta = (
            positive_delta
            if parameter.estimate + positive_delta <= parameter.upper_bound
            else -positive_delta
        )
        perturbed_parameters = dict(base_parameters)
        perturbed_parameters[parameter.name] = parameter.estimate + delta
        perturbed_cash = _deterministic_cash_path(state, action, perturbed_parameters)
        evidence_reference = (
            f"finite-difference:{world_model.id}:{action.name}:"
            f"{parameter.name.value}:delta={delta:.8g}"
        )
        for horizon_days, week in FORECAST_WEEKS.items():
            estimates.append(
                CashSensitivityEstimate(
                    parameter_name=parameter.name,
                    horizon_days=horizon_days,
                    cash_sensitivity_per_unit=(perturbed_cash[week] - base_cash[week]) / delta,
                    evidence_reference=evidence_reference,
                )
            )
    return tuple(estimates)


def action_plan_from_simulation(
    run: RunRecord,
    observation: ObservationSnapshot,
    selected: SimulationAction,
    selection_reason: str,
    proposal_lineage: ActionPlan | None = None,
) -> ActionPlan:
    week = observation.day // 7
    metrics = observation.metrics
    monthly_average_price = PeriodicMoney(
        amount=selected.price_per_customer_weekly,
        period=RatePeriod.WEEK,
    ).per(RatePeriod.MONTH_30_DAY).amount
    tier_prices = tuple(
        float(metrics.get(name, default))
        if isinstance(metrics.get(name, default), int | float)
        else default
        for name, default in zip(
            ("price_a", "price_b", "price_c"),
            INITIAL_MONTHLY_PRICES,
            strict=True,
        )
    )
    if any(price <= 0 for price in tier_prices):
        tier_prices = INITIAL_MONTHLY_PRICES
    tier_average = fmean(tier_prices)
    selected_tier_prices = tuple(
        monthly_average_price * price / tier_average for price in tier_prices
    )
    operations_weekly = (
        selected.operations_spend
        if selected.operations_spend is not None
        else metrics.get("operations_spend", 3_500.0)
    )
    operations_daily = max(
        0.0,
        float(operations_weekly) / 7.0
        if isinstance(operations_weekly, int | float)
        else 0.0,
    )
    raw_segments = str(metrics.get("known_segments") or "S1").split(",")
    segments = tuple(
        segment.strip()
        for segment in raw_segments
        if re.fullmatch(r"(?:S[1-3]|E[1-3]|D_[SE]\d{2})", segment.strip())
    ) or ("S1",)
    selected_quotas = (
        selected.usage_quota_a
        if selected.usage_quota_a is not None
        else float(metrics.get("usage_quota_a", 0.0) or 0.0),
        selected.usage_quota_b
        if selected.usage_quota_b is not None
        else float(metrics.get("usage_quota_b", 0.0) or 0.0),
        selected.usage_quota_c
        if selected.usage_quota_c is not None
        else float(metrics.get("usage_quota_c", 0.0) or 0.0),
    )
    selected_capacity_tier = (
        selected.capacity_tier
        if selected.capacity_tier is not None
        else int(float(metrics.get("capacity_tier", 0.0) or 0.0))
    )
    targeted_spend: dict[str, dict[str, float]] = {}
    if selected.targeted_ad_allocations:
        for allocation in selected.targeted_ad_allocations:
            targeted_spend.setdefault(allocation.channel, {})[
                allocation.segment
            ] = allocation.daily_spend
    else:
        daily_per_segment = selected.marketing_spend / 7.0 / len(segments)
        for segment in segments:
            channel = "linkedin" if segment.startswith(("E", "D_E")) else "search_ads"
            targeted_spend.setdefault(channel, {})[segment] = daily_per_segment

    commands = [
        ActionCommand(
            tool="set_prices",
            arguments={
                "A": selected_tier_prices[0],
                "B": selected_tier_prices[1],
                "C": selected_tier_prices[2],
            },
            idempotency_key=f"{run.id}:week-{week}:prices",
        ),
        ActionCommand(
            tool="set_model_tiers",
            arguments={
                "A": selected.model_tier_a or int(metrics.get("model_tier_a", 1)),
                "B": selected.model_tier_b or int(metrics.get("model_tier_b", 1)),
                "C": selected.model_tier_c or int(metrics.get("model_tier_c", 1)),
            },
            idempotency_key=f"{run.id}:week-{week}:model-tiers",
        ),
        ActionCommand(
            tool="set_daily_spend",
            arguments={
                "operations": max(0.0, operations_daily),
                "development": selected.development_spend / 7,
            },
            idempotency_key=f"{run.id}:week-{week}:spend",
        ),
        ActionCommand(
            tool="set_targeted_ad_spend",
            arguments={"targeted_spend": targeted_spend},
            idempotency_key=f"{run.id}:week-{week}:acquisition",
        ),
        ActionCommand(
            tool="set_targeted_dev_spend",
            arguments={
                "targeted_spend": {
                    allocation.segment: allocation.daily_spend
                    for allocation in selected.targeted_development_allocations
                }
            },
            idempotency_key=f"{run.id}:week-{week}:targeted-development",
        ),
    ]
    # Service allowance and capacity are only sent when the action actually
    # changes them. Restating them would be harmless; emitting a zero the caller
    # never chose would not, because it would switch the service off.
    observed_quotas = tuple(
        float(metrics.get(f"usage_quota_{plan}", 0.0) or 0.0) for plan in "abc"
    )
    if any(
        abs(selected - observed) > 1e-9
        for selected, observed in zip(selected_quotas, observed_quotas, strict=True)
    ):
        commands.append(
            ActionCommand(
                tool="set_usage_quotas",
                arguments={
                    "A": int(round(selected_quotas[0])),
                    "B": int(round(selected_quotas[1])),
                    "C": int(round(selected_quotas[2])),
                },
                idempotency_key=f"{run.id}:week-{week}:usage-quotas",
            )
        )
    if selected_capacity_tier != int(float(metrics.get("capacity_tier", 0.0) or 0.0)):
        commands.append(
            ActionCommand(
                tool="set_capacity_tier",
                arguments={"tier": selected_capacity_tier},
                idempotency_key=f"{run.id}:week-{week}:capacity-tier",
            )
        )
    # Controls that would restate the observed configuration are not sent: the
    # benchmark carries them forward unchanged, so an unchanged control needs no
    # command and leaves nothing for the fidelity check to disagree about.
    observed_recurring_promotion = float(
        metrics.get("recurring_promotion_monthly", 0.0) or 0.0
    )
    selected_recurring_promotion = (
        selected.recurring_promotion_monthly
        if selected.recurring_promotion_monthly is not None
        else observed_recurring_promotion
    )
    if abs(selected_recurring_promotion - observed_recurring_promotion) > 1e-9:
        commands.append(
            ActionCommand(
                tool="set_promotion",
                arguments={"global_promotion": selected_recurring_promotion},
                idempotency_key=f"{run.id}:week-{week}:recurring-promotion",
            )
        )
    observed_ads_strength = float(metrics.get("ads_strength", 0.0) or 0.0)
    selected_ads_strength = (
        selected.ads_strength
        if selected.ads_strength is not None
        else observed_ads_strength
    )
    if abs(selected_ads_strength - observed_ads_strength) > 1e-9:
        commands.append(
            ActionCommand(
                tool="set_ads_strength",
                arguments={"global_strength": selected_ads_strength},
                idempotency_key=f"{run.id}:week-{week}:ads-strength",
            )
        )
    observed_targeted_ops = float(metrics.get("targeted_ops_spend", 0.0) or 0.0)
    selected_targeted_ops = (
        selected.targeted_ops_spend
        if selected.targeted_ops_spend is not None
        else observed_targeted_ops
    )
    if abs(selected_targeted_ops - observed_targeted_ops) > 1e-9:
        commands.append(
            ActionCommand(
                tool="set_targeted_ops_spend",
                arguments={
                    "targeted_spend": {
                        segment: selected_targeted_ops / 7.0 / len(segments)
                        for segment in segments
                    }
                },
                idempotency_key=f"{run.id}:week-{week}:targeted-ops",
            )
        )
    selected_lead_promotion = (
        float(metrics.get("lead_promotion_monthly", 0.0))
        if selected.lead_promotion_monthly is None
        else selected.lead_promotion_monthly
    )
    observed_lead_promotion = metrics.get("lead_promotion_monthly", 0.0)
    observed_lead_promotion = (
        float(observed_lead_promotion)
        if isinstance(observed_lead_promotion, int | float)
        else 0.0
    )
    if abs(selected_lead_promotion - observed_lead_promotion) > 1e-9:
        commands.append(
            ActionCommand(
                tool="set_lead_promotion",
                arguments={"global_promotion": selected_lead_promotion},
                idempotency_key=f"{run.id}:week-{week}:lead-promotion",
            )
        )

    # One-shot actions: they carry no observed level to compare against, so they
    # are emitted exactly when the selected candidate declares them. Omitting
    # them here made both structurally unexecutable — the field survived the
    # simulation and then vanished at execution without a veto code, which is
    # the silent substitution this layer exists to prevent.
    if selected.research_project_tier:
        commands.append(
            ActionCommand(
                tool="start_research_project",
                arguments={"tier": int(selected.research_project_tier)},
                idempotency_key=f"{run.id}:week-{week}:research-project",
            )
        )
    if selected.social_posts:
        # The post is authored by the Executive, so its words come from the
        # proposal rather than being invented here.
        content = next(
            (
                str(command.arguments.get("content", "")).strip()
                for command in (
                    proposal_lineage.commands if proposal_lineage is not None else ()
                )
                if command.tool == "post_social_media"
            ),
            "",
        )
        if content:
            commands.append(
                ActionCommand(
                    tool="post_social_media",
                    arguments={"content": content[:280]},
                    idempotency_key=f"{run.id}:week-{week}:social-post",
                )
            )

    experiment_control = None
    experiment_expires_week = None
    experiment_program = None
    hypothesis_id = None
    evidence_regime = None
    if proposal_lineage is not None and proposal_lineage.experiment_program is not None:
        # A committed multi-week program keeps its original absolute clock. Rebuilding
        # it from the current week's SimulationAction would reinterpret the stored
        # maturity week as a new program's expiry and becomes invalid exactly when
        # build transitions into probe (started_week == minimum_maturity_week).
        experiment_program = proposal_lineage.experiment_program
        experiment_control = proposal_lineage.experiment_control
        experiment_expires_week = proposal_lineage.experiment_expires_week
        hypothesis_id = proposal_lineage.hypothesis_id
        evidence_regime = proposal_lineage.evidence_regime
    elif selected.name.startswith("controlled_exploration_"):
        quality_match = re.search(r"_q(\d)$", selected.name)
        quality_band = int(quality_match.group(1)) if quality_match else 0
        evidence_regime = f"quality_band_{quality_band}"
        if selected.name.startswith("controlled_exploration_marketing_"):
            experiment_control = "marketing"
            experiment_expires_week = selected.marketing_spend_until_week
            baseline_value = selected.marketing_spend_after_experiment
            treatment_value = selected.marketing_spend
            hypothesis_id = f"lead_support_q{quality_band}"
        elif selected.name.startswith("controlled_exploration_development_"):
            experiment_control = "development"
            experiment_expires_week = selected.development_spend_until_week
            baseline_value = selected.development_spend_after_experiment
            treatment_value = selected.development_spend
            hypothesis_id = f"quality_support_q{quality_band}"
        else:
            experiment_control = "lead_promotion"
            experiment_expires_week = selected.lead_promotion_until_week
            baseline_value = selected.lead_promotion_after_experiment
            treatment_value = selected.lead_promotion_monthly
            hypothesis_id = f"promotion_support_q{quality_band}"
        if (
            experiment_expires_week is not None
            and baseline_value is not None
            and treatment_value is not None
        ):
            staged_quality_probe = experiment_control == "development"
            target_segment = segments[0] if staged_quality_probe else None
            target_channel = (
                "linkedin"
                if target_segment is not None
                and target_segment.startswith(("E", "D_E"))
                else "search_ads"
            ) if staged_quality_probe else None
            # Size the probe from available risk capital, not from a hidden
            # conversion threshold. It is deliberately bounded so a failed probe
            # remains informative and reversible.
            acquisition_probe_weekly_spend = (
                min(5_000.0, max(1_000.0, observation.cash * 0.0025))
                if staged_quality_probe
                else 0.0
            )
            minimum_maturity_week = experiment_expires_week
            if staged_quality_probe:
                experiment_expires_week += 1
            experiment_program = ExperimentProgram(
                commitment_id=f"{hypothesis_id}-{week}",
                control=experiment_control,
                started_week=week,
                minimum_maturity_week=minimum_maturity_week,
                maximum_end_week=experiment_expires_week,
                baseline_value=float(baseline_value),
                treatment_value=float(treatment_value),
                maximum_cumulative_downside=min(
                    30_000.0,
                    max(1_000.0, observation.cash * 0.03),
                ),
                expected_observation=(
                    "Create outcome exposure outside the currently observed funnel support."
                ),
                falsification_condition=(
                    "After development matures, a fresh acquisition cohort at the new "
                    "quality regime produces no decision-relevant conversion evidence by "
                    f"week {experiment_expires_week}."
                ),
                target_segment=target_segment,
                target_channel=target_channel,
                acquisition_probe_weekly_spend=acquisition_probe_weekly_spend,
                baseline_targeted_ad_spend={
                    channel: {
                        segment: float(amount)
                        for segment, amount in groups.items()
                        if float(amount) > 0.0
                    }
                    for channel, groups in targeted_spend.items()
                    if any(float(amount) > 0.0 for amount in groups.values())
                },
            )

    return ActionPlan(
        name=selected.name,
        strategy_family=selected.name,
        rationale=selection_reason,
        commands=commands,
        proposal_kind="experiment" if experiment_program is not None else None,
        hypothesis_id=hypothesis_id if experiment_program is not None else None,
        experiment_control=(
            experiment_control if experiment_program is not None else None
        ),
        evidence_regime=evidence_regime if experiment_program is not None else None,
        experiment_expires_week=(
            experiment_expires_week if experiment_program is not None else None
        ),
        experiment_program=experiment_program,
        enterprise_engage=selected.enterprise_engage,
        enterprise_target_price_per_seat=selected.enterprise_target_price_per_seat,
        enterprise_floor_price_per_seat=selected.enterprise_floor_price_per_seat,
        enterprise_max_new_seats=selected.enterprise_max_new_seats,
    )


async def prepare_weekly_plan(
    *,
    run: RunRecord,
    observation: ObservationSnapshot,
    world_model: WorldModelVersion,
    executive: ExecutiveProposalEngine,
    n_rollouts: int = 200,
    simulation_seed: int | None = None,
    decision_history: tuple[DecisionRecord, ...] = (),
) -> WeeklyPlanningResult:
    state = simulation_state_from_observation(observation)
    active_experiment = _active_experiment_plan(
        decision_history,
        current_week=state.week,
    )
    reversion = _experiment_reversion_action(decision_history, state=state)
    reversion_plan = (
        action_plan_from_simulation(
            run,
            observation,
            reversion,
            "Explicit rollback after the committed experiment maturity window.",
        )
        if reversion is not None
        else None
    )
    forced_plan = active_experiment or reversion_plan
    legacy_batch = ProposalBatch()
    if forced_plan is None:
        legacy_batch = await _request_executive_plans(
            executive,
            run=run,
            observation=observation,
            decision_history=decision_history,
        )
    executive_plans = (
        (forced_plan,)
        if forced_plan is not None
        else _targeted_development_scale_variants(legacy_batch.plans)
    )
    exploration_memory = exploration_memory_from_decisions(
        decision_history, current_week=state.week
    )
    exploration_admission = assess_controlled_exploration(state, exploration_memory)
    if active_experiment is not None and active_experiment.experiment_program is not None:
        active_program = active_experiment.experiment_program
        exploration_admission = exploration_admission.model_copy(
            update={
                "active_commitment_strategy": active_experiment.strategy_family,
                "active_commitment_phase": (
                    "build"
                    if state.week < active_program.minimum_maturity_week
                    else "probe"
                ),
            }
        )
    legacy_actions, legacy_dropped = _executable_candidates(
        state,
        executive_plans,
        exploration_memory,
        active_commitment=forced_plan is not None,
    )
    candidates, feasibility = partition_feasible_actions(state, legacy_actions)
    feasibility = _with_refusals(
        feasibility,
        rejections=legacy_batch.rejections,
        dropped=legacy_dropped,
    )
    if not candidates:
        raise ValueError("weekly planning requires at least one economically valid candidate")
    seed = (
        simulation_seed
        if simulation_seed is not None
        else observation.day + world_model.version * 10_000
    )
    search = search_strategies(
        state=state,
        world_model=world_model,
        actions=candidates,
        horizon_weeks=_remaining_horizon_days(run, observation) // 7,
        n_rollouts=n_rollouts,
        seed=seed,
        prefer_bounded_exploration=(
            exploration_admission.admitted
            or any(
                action.name.startswith("executive_experiment_")
                for action in candidates
            )
        ),
    )
    selected = search.selected.action
    executive_rationale_by_name = {
        plan.strategy_family: plan.rationale for plan in executive_plans
    }
    executive_plan_by_name = {
        plan.strategy_family: plan for plan in executive_plans
    }
    forecasts = _rollout_forecasts(
        state,
        selected,
        world_model,
        n_rollouts=n_rollouts,
        seed=seed,
    )
    action_plan = action_plan_from_simulation(
        run,
        observation,
        selected,
        (
            search.selection_reason
            + (
                " Executive hypothesis: "
                + executive_rationale_by_name[selected.name]
                if selected.name in executive_rationale_by_name
                else ""
            )
        )[:4_000],
        proposal_lineage=executive_plan_by_name.get(selected.name),
    )
    action_plan = _with_proposal_lineage(
        action_plan,
        executive_plan_by_name.get(selected.name),
    )
    action_plan = _stage_experiment_plan(action_plan, current_week=state.week)
    records = tuple(
        CandidateEvaluationRecord(
            strategy=candidate.action.name,
            expected_ending_cash=candidate.expected_ending_cash,
            downside_ending_cash=candidate.downside_ending_cash,
            bankruptcy_probability=candidate.bankruptcy_probability,
            going_concern_failure_probability=(
                candidate.going_concern_failure_probability
            ),
            expected_customer_growth=candidate.expected_customer_growth,
            robustness=candidate.robustness,
            robust_utility=candidate.robust_utility,
            rollout_count=candidate.rollout_count,
            action_parameters=candidate.action.model_dump(mode="json"),
            proposal_rationale=executive_rationale_by_name.get(
                candidate.action.name
            ),
            proposal_source=(
                "gemini_executive"
                if candidate.action.name in executive_rationale_by_name
                else "deterministic_kernel"
            ),
            hypothesis_id=(
                executive_plan_by_name[candidate.action.name].hypothesis_id
                if candidate.action.name in executive_plan_by_name
                else None
            ),
            evidence_regime=(
                executive_plan_by_name[candidate.action.name].evidence_regime
                if candidate.action.name in executive_plan_by_name
                else None
            ),
        )
        for candidate in search.candidates
    )
    prompt_version = getattr(executive, "prompt_version", executive.__class__.__name__)
    return WeeklyPlanningResult(
        action_plan=action_plan,
        forecasts=forecasts,
        search=search,
        candidate_records=records,
        sensitivities=estimate_cash_sensitivities(state, selected, world_model),
        prompt_version=str(prompt_version),
        feasibility=feasibility,
        exploration_admission=exploration_admission,
        assumptions=(
            "The fixed P0 transition framework remains valid over the forecast horizon.",
            "World-model parameter intervals represent the current plausible worlds.",
        ),
        evidence_references=(
            f"world-model:{world_model.id}",
            f"observation:{run.id}:{observation.day}",
            *(f"executive-proposal:{plan.name}" for plan in executive_plans),
        ),
    )


def _executable_candidate_summary(
    *,
    state: SimulationState,
    action: SimulationAction,
    samples: tuple,
    horizon_days: int,
) -> CandidateSimulation:
    terminal = tuple(sample for sample in samples if sample.horizon_days == horizon_days)
    if not terminal:
        raise ValueError("executable model returned no terminal candidate samples")
    ending_cash = [sample.cash for sample in terminal]
    expected_cash = fmean(ending_cash)
    downside_cash = _quantile(ending_cash, 0.10)
    bankruptcy_probability = sum(cash < 0 for cash in ending_cash) / len(ending_cash)
    going_concern_failure_probability = sum(
        sample.customers < 1.0 or sample.revenue_weekly <= 0.0
        for sample in terminal
    ) / len(terminal)
    expected_growth = fmean(sample.customers - state.customers for sample in terminal)
    robust_utility = cash_robust_utility(
        initial_cash=state.cash,
        expected_ending_cash=expected_cash,
        downside_ending_cash=downside_cash,
        bankruptcy_probability=bankruptcy_probability,
    )
    if bankruptcy_probability <= 0.05 and downside_cash >= 0:
        robustness = RobustnessLevel.HIGH
    elif bankruptcy_probability <= 0.20:
        robustness = RobustnessLevel.MEDIUM
    else:
        robustness = RobustnessLevel.LOW
    return CandidateSimulation(
        action=action,
        expected_ending_cash=expected_cash,
        downside_ending_cash=downside_cash,
        upside_ending_cash=_quantile(ending_cash, 0.90),
        bankruptcy_probability=bankruptcy_probability,
        going_concern_failure_probability=going_concern_failure_probability,
        expected_customer_growth=expected_growth,
        robustness=robustness,
        robust_utility=robust_utility,
        rollout_count=len(terminal),
    )


async def prepare_executable_weekly_plan(
    *,
    run: RunRecord,
    observation: ObservationSnapshot,
    world_model: WorldModelVersion,
    executive: ExecutiveProposalEngine,
    runtime: ExecutableCompanyModel,
    fitted_model: FittedModel,
    n_rollouts: int = 200,
    simulation_seed: int | None = None,
    decision_history: tuple[DecisionRecord, ...] = (),
    portfolio_context: dict | None = None,
    rejection_feedback: tuple[dict, ...] | None = None,
    authority: ExecutiveAuthorityContext | None = None,
) -> WeeklyPlanningResult:
    """Rank bounded actions using the exact fitted executable model used for forecasts.

    With `authority` set (executive authority v2), Python only evaluates and
    vetoes: the final strategic selection among eligible candidates is made by
    the Executive's second stage, and every fallback resolves to a safe
    operational candidate.
    """

    state = simulation_state_from_observation(observation)
    active_experiment = _active_experiment_plan(
        decision_history,
        current_week=state.week,
    )
    reversion = _experiment_reversion_action(decision_history, state=state)
    reversion_plan = (
        action_plan_from_simulation(
            run,
            observation,
            reversion,
            "Explicit rollback after the committed experiment maturity window.",
        )
        if reversion is not None
        else None
    )
    adoption = (
        _experiment_adoption_action(decision_history, state=state)
        if authority is not None
        else None
    )
    adoption_plan = (
        action_plan_from_simulation(
            run,
            observation,
            adoption,
            "Adopt the matured treatment as the new operating configuration.",
        )
        if adoption is not None
        else None
    )
    # Under executive authority the company's own commitment is presented as a
    # choice, not imposed: continue it, roll it back, or adopt what it proved.
    # Deterministic code keeps the veto, never the verdict.
    commitment_options = tuple(
        option
        for option in (active_experiment, reversion_plan, adoption_plan)
        if option is not None
    )
    forced_plan = (
        None if authority is not None else (active_experiment or reversion_plan)
    )
    proposal_batch = ProposalBatch()
    proposal_stage_warnings: tuple[str, ...] = ()
    if forced_plan is not None:
        executive_plans: tuple[ActionPlan, ...] = (forced_plan,)
    else:
        for proposal_attempt in range(3):
            try:
                proposal_batch = await _request_executive_plans(
                    executive,
                    run=run,
                    observation=observation,
                    decision_history=decision_history,
                    portfolio_context=portfolio_context,
                    rejection_feedback=rejection_feedback,
                )
                break
            except Exception as error:
                # A failed proposal stage is a missing opinion, not a failed
                # week — but a single transient provider error must not cost
                # the Executive its whole authorship turn, so the stage is
                # retried before the degradation is accepted. The deterministic
                # pool still carries a fully failed stage in either authority
                # mode, and the degradation is named rather than silent.
                # Deterministic code never invents a replacement experiment.
                if authority is not None:
                    await authority.emit_event(
                        "executive.unavailable",
                        {
                            "week": state.week,
                            "stage": "action_proposals",
                            "attempt": proposal_attempt + 1,
                            "error": str(error)[:500],
                        },
                    )
                if proposal_attempt < 2:
                    await asyncio.sleep(2.0 * (proposal_attempt + 1))
                    continue
                proposal_stage_warnings = ("executive_proposal_stage_failed",)
        proposed = proposal_batch.plans
        executive_plans = (
            # Magnitude variants are deterministic authorship of a business
            # decision, so under executive authority the proposal stands as made.
            commitment_options + tuple(proposed)
            if authority is not None
            else _targeted_development_scale_variants(proposed)
        )
        if authority is not None:
            executive_plans, ballot_diagnostics = _with_distinct_candidate_identities(
                executive_plans
            )
            for diagnostic in ballot_diagnostics:
                await authority.emit_event(
                    "executive.duplicate_candidate",
                    {"week": state.week, **diagnostic},
                )
    exploration_memory = exploration_memory_from_decisions(
        decision_history, current_week=state.week
    )
    exploration_admission = assess_controlled_exploration(state, exploration_memory)
    if active_experiment is not None and active_experiment.experiment_program is not None:
        active_program = active_experiment.experiment_program
        exploration_admission = exploration_admission.model_copy(
            update={
                "active_commitment_strategy": active_experiment.strategy_family,
                "active_commitment_phase": (
                    "build"
                    if state.week < active_program.minimum_maturity_week
                    else "probe"
                ),
            }
        )
    executable_actions, executable_dropped = _executable_candidates(
        state,
        executive_plans,
        exploration_memory,
        active_commitment=forced_plan is not None,
        deterministic_exploration=authority is None,
    )
    actions, feasibility = partition_feasible_actions(state, executable_actions)
    feasibility = _with_refusals(
        feasibility,
        rejections=proposal_batch.rejections,
        dropped=executable_dropped,
        warning_codes=proposal_stage_warnings,
    )
    if not actions:
        raise ValueError("executable planning requires at least one economically valid candidate")
    seed = (
        simulation_seed
        if simulation_seed is not None
        else observation.day + world_model.version * 10_000
    )
    selection_horizon_days = _remaining_horizon_days(run, observation)
    horizons = tuple(sorted({*FORECAST_WEEKS, selection_horizon_days}))
    distributions = []
    evaluations = []
    for action in actions:
        action_payload = (
            sandbox_action_payload(
                action,
                state,
                horizon_weeks=selection_horizon_days // 7,
            )
            if runtime.artifact.runtime_kind.value == "sandboxed_python"
            else action.model_dump(mode="json")
        )
        distribution = runtime.predict(
            CompanyModelPredictRequest(
                fitted_model=fitted_model,
                state=state.model_dump(mode="json"),
                action=action_payload,
                horizons_days=horizons,
                n_rollouts=n_rollouts,
                # Common random numbers make candidate differences attributable to
                # actions instead of unrelated rollout draws.
                seed=seed,
            )
        )
        invariant_report = evaluate_model_outcomes(distribution)
        if not invariant_report.valid:
            raise ValueError("executable planning candidate failed accounting invariants")
        distributions.append(distribution)
        evaluations.append(
            _executable_candidate_summary(
                state=state,
                action=action,
                samples=distribution.samples,
                horizon_days=selection_horizon_days,
            )
        )
    executive_rationale_by_name = {
        plan.strategy_family: plan.rationale for plan in executive_plans
    }
    executive_plan_by_name = {
        plan.strategy_family: plan for plan in executive_plans
    }
    v2_action_plan: ActionPlan | None = None
    information_requests: tuple[InformationRequest, ...] = ()
    data_queries: tuple[str, ...] = ()
    journal = ""
    if authority is not None:
        (
            search,
            v2_action_plan,
            information_requests,
            data_queries,
            journal,
        ) = await _select_with_executive_authority(
            run=run,
            observation=observation,
            state=state,
            evaluations=tuple(evaluations),
            distributions=distributions,
            executive=executive,
            authority=authority,
            executive_plan_by_name=executive_plan_by_name,
            commitment_candidate_id=(
                active_experiment.strategy_family
                if active_experiment is not None
                else None
            ),
            decision_history=decision_history,
            parameters_on_priors=parameters_still_on_priors(world_model),
            parameters_unmeasured=parameters_never_measured(world_model),
        )
    else:
        search = select_robust_strategy(
            tuple(evaluations),
            prefer_bounded_exploration=(
                exploration_admission.admitted
                or any(
                    action.name.startswith("executive_experiment_")
                    for action in actions
                )
            ),
            inherited_going_concern_failure=(
                state.customers < 1.0 or state.revenue_weekly <= 0.0
            ),
        )
    selected_index = next(
        index
        for index, candidate in enumerate(evaluations)
        if candidate.action.name == search.selected.action.name
    )
    selected_distribution = distributions[selected_index]
    forecasts = CashForecasts(
        items=[
            CashForecast(
                horizon_days=horizon,
                point=median(
                    sample.cash
                    for sample in selected_distribution.samples
                    if sample.horizon_days == horizon
                ),
                lower=_quantile(
                    [
                        sample.cash
                        for sample in selected_distribution.samples
                        if sample.horizon_days == horizon
                    ],
                    0.025,
                ),
                upper=_quantile(
                    [
                        sample.cash
                        for sample in selected_distribution.samples
                        if sample.horizon_days == horizon
                    ],
                    0.975,
                ),
            )
            for horizon in FORECAST_WEEKS
        ]
    )
    if v2_action_plan is not None:
        action_plan = v2_action_plan
    else:
        action_plan = action_plan_from_simulation(
            run,
            observation,
            search.selected.action,
            (
                search.selection_reason
                + (
                    " Executive hypothesis: "
                    + executive_rationale_by_name[search.selected.action.name]
                    if search.selected.action.name in executive_rationale_by_name
                    else ""
                )
            )[:4_000],
            proposal_lineage=executive_plan_by_name.get(search.selected.action.name),
        )
        action_plan = _with_proposal_lineage(
            action_plan,
            executive_plan_by_name.get(search.selected.action.name),
        )
        action_plan = _stage_experiment_plan(action_plan, current_week=state.week)
    records = tuple(
        CandidateEvaluationRecord(
            strategy=candidate.action.name,
            expected_ending_cash=candidate.expected_ending_cash,
            downside_ending_cash=candidate.downside_ending_cash,
            bankruptcy_probability=candidate.bankruptcy_probability,
            going_concern_failure_probability=(
                candidate.going_concern_failure_probability
            ),
            expected_customer_growth=candidate.expected_customer_growth,
            robustness=candidate.robustness,
            robust_utility=candidate.robust_utility,
            rollout_count=candidate.rollout_count,
            action_parameters=candidate.action.model_dump(mode="json"),
            proposal_rationale=executive_rationale_by_name.get(
                candidate.action.name
            ),
            proposal_source=(
                "gemini_executive"
                if candidate.action.name in executive_rationale_by_name
                else "deterministic_kernel"
            ),
            hypothesis_id=(
                executive_plan_by_name[candidate.action.name].hypothesis_id
                if candidate.action.name in executive_plan_by_name
                else None
            ),
            evidence_regime=(
                executive_plan_by_name[candidate.action.name].evidence_regime
                if candidate.action.name in executive_plan_by_name
                else None
            ),
        )
        for candidate in search.candidates
    )
    executive_version = getattr(executive, "prompt_version", executive.__class__.__name__)
    return WeeklyPlanningResult(
        action_plan=action_plan,
        forecasts=forecasts,
        search=search,
        candidate_records=records,
        sensitivities=(),
        prompt_version=f"{executive_version}+{runtime.artifact.prompt_version}",
        feasibility=feasibility,
        exploration_admission=exploration_admission,
        information_requests=information_requests,
        data_queries=data_queries,
        journal=journal,
        assumptions=(
            "The +7 day forecast is the committed-action forecast.",
            "Longer planning forecasts are explicitly hold-action scenarios and are "
            "not scored against later replanning.",
            "Candidate ranking uses the benchmark's remaining "
            f"{selection_horizon_days}-day terminal horizon.",
            "All forecast samples passed the independent accounting invariant gate.",
        ),
        evidence_references=(
            f"world-model:{world_model.id}",
            f"model-artifact:{runtime.artifact.id}:{runtime.artifact.content_hash}",
            f"fitted-model:{fitted_model.id}:{fitted_model.state_hash}",
            f"observation:{run.id}:{observation.day}",
        ),
    )


def _latest_commitment_program(
    decision_history: tuple[DecisionRecord, ...],
    plans_by_candidate: dict[str, ActionPlan],
) -> ExperimentProgram | None:
    for plan in plans_by_candidate.values():
        if plan.experiment_program is not None:
            return plan.experiment_program
    for decision in reversed(decision_history):
        program = decision.action_plan.experiment_program
        if program is not None:
            return program
    return None


def _candidate_pool_diagnostics(
    cards: tuple,
    plans_by_candidate: dict[str, ActionPlan],
) -> tuple[str, ...]:
    """Generic epistemic diagnostics; Python names the gap, never fills it."""

    diagnostics: list[str] = []
    if any(
        code.startswith("configuration_incomplete")
        for card in cards
        for code in card.veto_codes
    ):
        diagnostics.append("configuration_incomplete")
    experiment_cards = [
        card
        for card in cards
        if plans_by_candidate[card.candidate_id].proposal_kind == "experiment"
    ]
    eligible_experiments = [card for card in experiment_cards if card.eligible]
    if experiment_cards and not eligible_experiments and any(
        "hypothesis_falsified_in_support" in card.veto_codes
        or "experiment_over_budget" in card.veto_codes
        for card in experiment_cards
    ):
        diagnostics.append("strategy_exhausted")
    if len(eligible_experiments) >= 2:
        signatures = {
            (
                plans_by_candidate[card.candidate_id].experiment_control,
                (
                    plans_by_candidate[card.candidate_id].experiment_program.target_segment
                    if plans_by_candidate[card.candidate_id].experiment_program
                    else None
                ),
                (
                    plans_by_candidate[card.candidate_id].experiment_program.target_channel
                    if plans_by_candidate[card.candidate_id].experiment_program
                    else None
                ),
            )
            for card in eligible_experiments
        }
        if len(signatures) == 1:
            diagnostics.append("low_causal_diversity")
    return tuple(diagnostics)


async def _select_with_executive_authority(
    *,
    run: RunRecord,
    observation: ObservationSnapshot,
    state: SimulationState,
    evaluations: tuple[CandidateSimulation, ...],
    distributions: list,
    executive: ExecutiveProposalEngine,
    authority: ExecutiveAuthorityContext,
    executive_plan_by_name: dict[str, ActionPlan],
    commitment_candidate_id: str | None,
    decision_history: tuple[DecisionRecord, ...],
    parameters_on_priors: tuple[str, ...] = (),
    parameters_unmeasured: frozenset[str] = frozenset(),
) -> tuple[
    StrategySearchResult, ActionPlan, tuple[InformationRequest, ...], tuple[str, ...], str
]:
    """Executive authority v2 selection: cards, veto, and Gemini's final choice."""

    plans_by_candidate: dict[str, ActionPlan] = {}
    surviving_evaluations: list[CandidateSimulation] = []
    surviving_distributions: list = []
    for candidate, distribution in zip(evaluations, distributions, strict=True):
        name = candidate.action.name
        lineage = executive_plan_by_name.get(name)
        rationale = (
            lineage.rationale
            if lineage is not None
            else "Deterministic operational candidate."
        )[:4_000]
        # A candidate whose executable plan cannot be built is one candidate fewer,
        # not a failed week. Operational candidates are built from live state and
        # always construct, so the floor of the ballot survives any refusal here.
        try:
            plan = action_plan_from_simulation(
                run,
                observation,
                candidate.action,
                rationale,
                proposal_lineage=lineage,
            )
            plan = _with_proposal_lineage(plan, lineage)
            plan = _stage_experiment_plan(plan, current_week=state.week)
        except (ValidationError, ValueError) as error:
            await authority.emit_event(
                "decision.candidate_construction_failed",
                {
                    "week": state.week,
                    "candidate_id": name,
                    "veto_codes": list(construction_veto_codes(error)),
                    "detail": str(error)[:500],
                },
            )
            continue
        plans_by_candidate[name] = plan
        surviving_evaluations.append(candidate)
        surviving_distributions.append(distribution)
    evaluations = tuple(surviving_evaluations)
    distributions = surviving_distributions

    horizon_cash_by_candidate = {
        candidate.action.name: {
            horizon: median(
                sample.cash
                for sample in distribution.samples
                if sample.horizon_days == horizon
            )
            for horizon in FORECAST_WEEKS
        }
        for candidate, distribution in zip(evaluations, distributions, strict=True)
    }
    # The same downside quantile the terminal gates use, but per horizon, so an
    # experiment's risk can be read at its own commitment window.
    horizon_downside_cash_by_candidate = {
        candidate.action.name: {
            horizon: _quantile(
                [
                    sample.cash
                    for sample in distribution.samples
                    if sample.horizon_days == horizon
                ],
                0.10,
            )
            for horizon in FORECAST_WEEKS
        }
        for candidate, distribution in zip(evaluations, distributions, strict=True)
    }
    portfolio = authority.portfolio
    experiment_budget = (
        portfolio.remaining_experiment_budget
        if portfolio is not None
        else experiment_budget_ceiling(state.cash)
    )
    cards = build_evaluation_cards(
        evaluations=evaluations,
        plans_by_candidate=plans_by_candidate,
        horizon_cash_by_candidate=horizon_cash_by_candidate,
        portfolio=portfolio,
        experiment_budget=experiment_budget,
        inherited_going_concern_failure=(
            state.customers < 1.0 or state.revenue_weekly <= 0.0
        ),
        observed_prices={
            plan: float(observation.metrics.get(f"configured_price_{plan.lower()}", 0.0) or 0.0)
            for plan in "ABC"
        },
        observed_quotas={
            plan: float(observation.metrics.get(f"usage_quota_{plan.lower()}", 0.0) or 0.0)
            for plan in "ABC"
        },
        horizon_downside_cash_by_candidate=horizon_downside_cash_by_candidate,
        parameters_on_priors=parameters_on_priors,
        parameters_unmeasured=parameters_unmeasured,
        state=state,
    )
    diagnostics = _candidate_pool_diagnostics(cards, plans_by_candidate)
    if "strategy_exhausted" in diagnostics:
        await authority.emit_event(
            "decision.strategy_exhausted",
            {
                "week": state.week,
                "vetoed_experiments": [
                    {
                        "candidate_id": card.candidate_id,
                        "veto_codes": list(card.veto_codes),
                    }
                    for card in cards
                    if not card.eligible
                ],
            },
        )
    if "low_causal_diversity" in diagnostics:
        await authority.emit_event(
            "decision.candidate_diversity_low",
            {
                "week": state.week,
                "eligible_experiment_candidates": [
                    card.candidate_id for card in cards if card.eligible
                ],
            },
        )
    selection = await run_executive_selection(
        run_id=run.id,
        week=state.week,
        executive=executive,
        authority=authority,
        cards=cards,
        diagnostics=diagnostics,
        cash=state.cash,
        known_insight_identities=authority.known_insight_identities,
        learned_costs=authority.learned_information_costs,
        quality_position=quality_position_facts(state),
    )
    selected = next(
        candidate
        for candidate in evaluations
        if candidate.action.name == selection.candidate_id
    )
    active_program = _active_commitment_program(decision_history, state=state)
    if active_program is not None:
        verdict, verdict_reason = commitment_verdict_from_choice(
            active_program,
            week=state.week,
            chosen_candidate_id=selection.candidate_id,
            commitment_candidate_id=commitment_candidate_id,
        )
        await record_commitment_review(
            run_id=run.id,
            week=state.week,
            program=active_program,
            verdict=verdict,
            reason=verdict_reason,
            authority=authority,
        )
        await authority.emit_event(
            "commitment.reviewed",
            {
                "week": state.week,
                "commitment_id": active_program.commitment_id,
                "verdict": verdict.value,
                "chosen_candidate_id": selection.candidate_id,
            },
        )
    chosen_plan = plans_by_candidate[selection.candidate_id]
    chosen_card = next(
        card for card in cards if card.candidate_id == selection.candidate_id
    )
    if chosen_plan.semantic_hash != chosen_card.plan_hash:
        raise ValueError(
            "selected plan hash changed between evaluation and execution: "
            f"{selection.candidate_id}"
        )
    search = StrategySearchResult(
        candidates=evaluations,
        selected=selected,
        selection_reason_code=selection.selection_reason_code,
        selection_reason=selection.selection_reason[:1_000],
    )
    return (
        search,
        chosen_plan,
        selection.information_requests,
        selection.data_queries,
        selection.journal,
    )
