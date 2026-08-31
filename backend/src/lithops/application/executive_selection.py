"""Executive authority v2: deterministic evaluation cards and the final choice.

Python evaluates and vetoes; Gemini selects one eligible, immutable candidate ID.
No fallback branch in this module may start an experiment: every failure path
resolves to the safest eligible operational candidate (continuation or the
required reversion) and leaves the unexecuted choice in the audit trail.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Container, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID

from lithops.agents.common import ExecutiveChoiceOutput, InformationRequestOutput
from lithops.domain.errors import ConflictError
from lithops.domain.insights import InformationRequest, InsightRecord
from lithops.domain.models import ActionCommand, ActionPlan, ExperimentProgram, utc_now
from lithops.domain.ports import StrategyRepository
from lithops.domain.public_instruments import MODEL_TIER_QUALITY_MULTIPLIER
from lithops.domain.strategy import (
    CandidateEvaluationCard,
    CandidateEvaluationSet,
    CommitmentReview,
    CommitmentReviewVerdict,
    ExecutiveChoice,
    HypothesisStatus,
    StrategicHypothesis,
    StrategicPortfolio,
    candidate_evaluation_set_id,
    commitment_review_id,
    executive_choice_id,
    validate_choice_against_set,
)
from lithops.domain.world_model import WorldModelParameterName
from lithops.simulator.models import SimulationAction, SimulationState
from lithops.simulator.state_transition import effective_participation_floor
from lithops.simulator.strategy_search import (
    MAX_ACCEPTABLE_BANKRUPTCY_PROBABILITY,
    MAX_ACCEPTABLE_GOING_CONCERN_FAILURE_PROBABILITY,
    CandidateSimulation,
)

NO_PORTFOLIO_HASH = "0" * 64
# How much insolvency risk a candidate may carry beyond continuation's before
# the gate treats it as risk that candidate *added* rather than risk the
# company was already carrying. Forecast noise between two rollout sets of the
# same trajectory is well inside this band.
BANKRUPTCY_RISK_TOLERANCE = 0.05
OPERATIONAL_CANDIDATE_PREFIXES = ("continuation", "unit_cost_recovery", "experiment_revert_")
# Information is bought out of the same runway as everything else; this bounds how
# much of it one week may consume without prescribing what to buy.
INFORMATION_BUDGET_CASH_FRACTION = 0.03
MAX_WEEKLY_INFORMATION_REQUESTS = 2

CONTROL_LEVERS = {
    "price": "price",
    "tier": "tier",
    "quota": "service_allowance",
    "marketing": "acquisition",
    "development": "development",
    "targeted_development": "targeted_development",
    "lead_promotion": "promotion",
}


def _controlled_configuration(program: ExperimentProgram, *, treatment: bool) -> object:
    configuration = (
        program.treatment_configuration
        if treatment
        else program.baseline_configuration
    )
    key = {
        "price": "prices",
        "tier": "model_tiers",
        "quota": "usage_quotas",
        "marketing": "weekly_marketing_spend",
        "development": "daily_development_spend",
        "targeted_development": "targeted_development_daily",
        "lead_promotion": "lead_promotion_monthly",
    }.get(program.control)
    return configuration.get(key) if key is not None else None

EmitEvent = Callable[[str, dict], Awaitable[object]]


class CandidateSelectionEngine(Protocol):
    async def select_candidate(self, *, brief: dict) -> ExecutiveChoiceOutput: ...


@dataclass(frozen=True, slots=True)
class ExecutiveAuthorityContext:
    """Everything the v2 selection stage needs beyond the planning inputs."""

    strategy_repository: StrategyRepository
    emit_event: EmitEvent
    portfolio: StrategicPortfolio | None = None
    # Information already bought in this run, so the same answer is not paid for twice.
    known_insight_identities: frozenset[str] = frozenset()
    # What each kind of purchase has cost so far, learned from the run itself.
    learned_information_costs: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SelectionOutcome:
    candidate_id: str
    selection_reason_code: str
    selection_reason: str
    choice: ExecutiveChoice | None
    information_requests: tuple[InformationRequest, ...] = ()
    data_queries: tuple[str, ...] = ()
    journal: str = ""


def admit_information_requests(
    requests: Sequence[InformationRequestOutput],
    *,
    cash: float,
    recent_identities: Container[str],
    learned_costs: Mapping[str, float] | None = None,
) -> tuple[tuple[InformationRequest, ...], tuple[str, ...]]:
    """Gate information purchases on affordability and non-duplication.

    Information is the one executed action that is not simulated first, because
    it changes nothing about the company. In exchange it must fit the same
    generic cash-fraction ceiling as any other spend, and it must resolve
    something the run has not already bought.

    Prices are not known in advance, so they are learned from what earlier
    purchases actually cost. A kind of purchase whose price has never been
    observed is allowed once per week: that is how its price becomes known,
    bounded so an unknown price cannot be paid twice in the same week.
    """

    remaining = information_budget_ceiling(cash)
    prices = learned_costs or {}
    admitted: list[InformationRequest] = []
    diagnostics: list[str] = []
    unpriced_admitted = 0
    for output in requests:
        request = InformationRequest(
            tool=output.tool,
            target_group=output.target_group or None,
            target_level=output.target_level or None,
            expected_information_value=output.expected_information_value,
        )
        if request.identity in recent_identities:
            diagnostics.append(f"duplicate_insight_request:{request.identity}")
            continue
        if len(admitted) >= MAX_WEEKLY_INFORMATION_REQUESTS:
            diagnostics.append(f"information_budget_exceeded:{request.identity}")
            continue
        known_price = prices.get(request.price_key)
        if known_price is None:
            if unpriced_admitted >= 1:
                diagnostics.append(f"information_price_unknown:{request.identity}")
                continue
            unpriced_admitted += 1
        elif known_price > remaining:
            diagnostics.append(f"information_budget_exceeded:{request.identity}")
            continue
        else:
            remaining -= known_price
        admitted.append(request)
    return tuple(admitted), tuple(diagnostics)


def learned_information_costs(
    records: Sequence[InsightRecord],
) -> dict[str, float]:
    """What each kind of purchase has actually cost this run, so far."""

    costs: dict[str, float] = {}
    for record in records:
        if record.cost <= 0.0:
            continue
        key = record.price_key
        costs[key] = max(costs.get(key, 0.0), record.cost)
    return costs


def information_budget_ceiling(cash: float) -> float:
    """A ceiling on weekly information spend, never a spend target."""

    return max(0.0, cash * INFORMATION_BUDGET_CASH_FRACTION)


def is_operational_candidate(candidate_id: str) -> bool:
    return candidate_id.startswith(OPERATIONAL_CANDIDATE_PREFIXES)


def configuration_completeness_violations(
    plan: ActionPlan,
    *,
    observed_prices: Mapping[str, float] | None = None,
    observed_quotas: Mapping[str, float] | None = None,
) -> tuple[str, ...]:
    """Controls the plan would leave in a state that cannot produce revenue.

    The judgement is on the configuration the week would end with, not on which
    commands the plan happens to send: a control the plan does not touch keeps
    its observed value. A generic serviceability rule, not a recommended level —
    a catalog priced at zero everywhere collects nothing, and a service
    allowance of zero everywhere delivers nothing to any customer. Python names
    the gap and never chooses the replacement value.
    """

    def numeric_arguments(command: ActionCommand) -> dict[str, float]:
        return {
            str(key): float(value)
            for key, value in command.arguments.items()
            if isinstance(value, int | float)
        }

    prices = dict(observed_prices or {})
    quotas = dict(observed_quotas or {})
    for command in plan.commands:
        if command.tool == "set_prices":
            prices = numeric_arguments(command)
        elif command.tool == "set_usage_quotas":
            quotas = numeric_arguments(command)
    codes: list[str] = []
    if prices and all(value <= 0.0 for value in prices.values()):
        codes.append("configuration_incomplete_prices")
    if quotas and all(value <= 0.0 for value in quotas.values()):
        codes.append("configuration_incomplete_service_allowance")
    return tuple(codes)


def _descends_from(
    hypothesis_id: str,
    ancestor_id: str,
    by_id: Mapping[str, StrategicHypothesis],
) -> bool:
    """Whether a hypothesis is reachable through an ancestor's successor chain.

    Each generation declares what makes it materially different from the one it
    replaces, so a descendant carries that justification forward however many
    steps back the falsified ancestor sits.
    """

    seen: set[str] = set()
    frontier = [ancestor_id]
    while frontier:
        current = frontier.pop()
        if current in seen:
            continue
        seen.add(current)
        hypothesis = by_id.get(current)
        if hypothesis is None:
            continue
        for successor in hypothesis.successor_hypothesis_ids:
            if successor == hypothesis_id:
                return True
            frontier.append(successor)
    return False


def assess_experiment(
    plan: ActionPlan,
    *,
    portfolio: StrategicPortfolio | None,
) -> tuple[str, ...]:
    """Evaluate a proposed experiment; return veto codes, never a candidate.

    This replaces deterministic exploration authorship: admission reports on the
    Executive's proposal instead of generating a probe of its own. Cost is not
    vetoed here: solvency has its own gates, the declared cumulative-downside cap
    is enforced weekly during execution, and the price of learning is a business
    judgement — the card states it and the Executive weighs it.
    """

    codes: list[str] = []
    program = plan.experiment_program
    if program is None:
        codes.append("experiment_program_missing")
        return tuple(codes)
    if program.is_standing_commitment:
        # A held direction owes a stop condition and a spending limit, both
        # enforced by the program itself. It owes no control arm and no
        # measurement plan, because it is not claiming to measure anything —
        # demanding those would force every strategy to be dressed as a probe.
        return tuple(codes)
    if program.protocol_version != "experiment-program-v2":
        codes.append("experiment_protocol_not_v2")
    elif _controlled_configuration(program, treatment=False) == _controlled_configuration(
        program, treatment=True
    ):
        codes.append("experiment_treatment_is_noop")
    measurement_sources = {item.source for item in program.measurement_plan}
    if "configuration" not in measurement_sources:
        codes.append("configuration_measurement_missing")
    cohort_measurements = [
        item for item in program.measurement_plan if item.source == "cohort"
    ]
    if not cohort_measurements or not any(
        item.decision_grade and item.minimum_exposure > 0
        for item in cohort_measurements
    ):
        codes.append("decision_grade_cohort_measurement_missing")
    if program.target_segment is None or program.target_channel is None:
        codes.append("measurement_target_missing")
    if plan.evidence_regime is not None and plan.evidence_regime.endswith(
        ":unobserved_segment"
    ):
        # The proposal targets a segment this run has never observed. Refusing it
        # is honest; quietly retargeting it would execute a decision nobody made.
        codes.append("target_segment_not_observed")
    if program.control in {"development", "targeted_development"}:
        if program.acquisition_probe_weekly_spend <= 0.0:
            codes.append("acquisition_probe_missing")
    elif program.control in {"price", "tier", "quota", "lead_promotion"}:
        planned_spend = program.treatment_configuration.get(
            "weekly_marketing_spend", 0.0
        )
        if not isinstance(planned_spend, int | float) or planned_spend <= 0.0:
            codes.append("no_planned_measurement_exposure")
    if portfolio is not None and plan.hypothesis_id is not None:
        by_id = {
            hypothesis.hypothesis_id: hypothesis
            for hypothesis in portfolio.hypotheses
        }
        own = by_id.get(plan.hypothesis_id)
        if own is not None and own.status is HypothesisStatus.FALSIFIED:
            codes.append("hypothesis_falsified_in_support")
        elif program is not None:
            lever = CONTROL_LEVERS.get(program.control)
            for hypothesis in portfolio.hypotheses:
                if (
                    hypothesis.status is HypothesisStatus.FALSIFIED
                    and hypothesis.hypothesis_id != plan.hypothesis_id
                    # A lineage may take several generations to reach the idea
                    # being proposed now. Following only the direct successor
                    # would let an ancestor veto its own descendants forever.
                    and not _descends_from(
                        plan.hypothesis_id, hypothesis.hypothesis_id, by_id
                    )
                    and hypothesis.segment == program.target_segment
                    and hypothesis.channel == program.target_channel
                    and any(item.value == lever for item in hypothesis.levers)
                ):
                    codes.append("hypothesis_falsified_in_support")
                    break
        if (
            "hypothesis_falsified_in_support" not in codes
            and portfolio.active_hypothesis_ids
            and plan.hypothesis_id not in portfolio.active_hypothesis_ids
        ):
            codes.append("hypothesis_not_active")
    return tuple(codes)


def _commitment_window_downside_cost(
    plan: ActionPlan,
    *,
    candidate_id: str,
    horizon_downside_cash_by_candidate: (
        Mapping[str, dict[int, float]] | None
    ),
) -> float | None:
    """What the experiment risks against continuation inside its own window.

    An experiment is a bounded commitment: the weekly review can stop it and the
    maturity window ends it, so the cash it puts at risk is the downside gap at
    the horizon nearest that window — not the permanent-operation counterfactual
    carried to the terminal day, which priced every standing-cost lever out of
    existence regardless of its return.
    """

    if horizon_downside_cash_by_candidate is None:
        return None
    program = plan.experiment_program
    if program is None:
        return None
    candidate_downside = horizon_downside_cash_by_candidate.get(candidate_id)
    continuation_downside = horizon_downside_cash_by_candidate.get("continuation")
    if not candidate_downside or not continuation_downside:
        return None
    window_days = max(7, (program.maximum_end_week - program.started_week) * 7)
    shared = sorted(set(candidate_downside) & set(continuation_downside))
    if not shared:
        return None
    horizon = next(
        (value for value in shared if value >= window_days), shared[-1]
    )
    return max(
        0.0, continuation_downside[horizon] - candidate_downside[horizon]
    )


def parameters_still_on_priors(world_model: object) -> tuple[str, ...]:
    """Parameters whose only evidence is a generic prior.

    A forecast built on these is an opinion wearing a number's clothes. Naming
    them lets the Executive weigh a confident-looking figure for what it is,
    instead of reading an uninformative starting value as a measurement.
    """

    names: list[str] = []
    for parameter in getattr(world_model, "parameters", ()):
        evidence = getattr(parameter, "evidence", ())
        if all(
            getattr(item, "kind", None) is None
            or str(getattr(item, "kind", "")) == "generic_prior"
            for item in evidence
        ):
            names.append(str(parameter.name.value))
    return tuple(sorted(names))


def parameters_never_measured(world_model: object) -> frozenset[str]:
    """Parameters the world model holds no value for at all.

    An absent parameter is not a zero. A candidate whose outcome depends on one
    is unquantified, and saying so is what lets the Executive treat finding out
    as a reason to act rather than reading silence as a forecast of nothing.
    """

    known = {
        str(parameter.name.value)
        for parameter in getattr(world_model, "parameters", ())
    }
    return frozenset(
        name.value for name in WorldModelParameterName if name.value not in known
    )


# Which learned parameters govern the payoff of each lever a candidate can
# exercise. Mechanism knowledge (this lever's return is priced by that
# parameter), not environment knowledge (no values, no thresholds).
_LEVER_GOVERNING_PARAMETERS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("research_programme", "research_project_tier", (
        "research_quality_per_tier",
        "research_lag_weeks_per_tier",
    )),
    ("in_product_ads", "ads_strength", (
        "ads_revenue_rate",
        "ads_quality_tradeoff",
    )),
    ("enterprise_deals", "enterprise_engage", (
        "enterprise_price_sensitivity",
    )),
    ("owned_channel_social_post", "social_posts", ("social_lead_response",)),
    ("targeted_operations", "targeted_ops_spend", ("ops_reliability_response",)),
)


def levers_on_prior_only_parameters(
    action: object, parameters_on_priors: tuple[str, ...]
) -> tuple[str, ...]:
    """The prior-only parameters this candidate's own levers lean on.

    Run-level `forecast_rests_on_priors` names the model's ignorance; this
    names the candidate that would turn a given piece of it into a measurement
    by being executed. Only levers the action actually exercises count.
    """

    on_priors = set(parameters_on_priors)
    named: list[str] = []
    for _, field_name, governing in _LEVER_GOVERNING_PARAMETERS:
        value = getattr(action, field_name, None)
        exercised = bool(value) if not isinstance(value, float) else value > 0.0
        if not exercised:
            continue
        named.extend(name for name in governing if name in on_priors)
    return tuple(dict.fromkeys(named))


def _candidate_delivered_quality(
    state: SimulationState, action: SimulationAction
) -> float:
    """Near-term delivered quality under the candidate's tiers.

    Composed from the current base quality and the environment's own published
    tier-multiplier table — the number a customer would judge on the candidate's
    best plan, before any lagged quality effects mature.
    """

    tiers = (
        state.model_tier_a if action.model_tier_a is None else action.model_tier_a,
        state.model_tier_b if action.model_tier_b is None else action.model_tier_b,
        state.model_tier_c if action.model_tier_c is None else action.model_tier_c,
    )
    return max(
        min(1.0, max(0.0, state.product_quality * MODEL_TIER_QUALITY_MULTIPLIER[tier]))
        for tier in tiers
    )


def _is_quality_side(action: SimulationAction) -> bool:
    """Whether a candidate's point is the quality customers receive."""

    return bool(
        action.model_tier_a is not None
        or action.model_tier_b is not None
        or action.model_tier_c is not None
        or action.usage_quota_a is not None
        or action.usage_quota_b is not None
        or action.usage_quota_c is not None
        or action.research_project_tier
        or action.targeted_development_allocations
    )


def build_evaluation_cards(
    *,
    evaluations: tuple[CandidateSimulation, ...],
    plans_by_candidate: Mapping[str, ActionPlan],
    horizon_cash_by_candidate: Mapping[str, dict[int, float]],
    portfolio: StrategicPortfolio | None,
    experiment_budget: float,
    inherited_going_concern_failure: bool,
    observed_prices: Mapping[str, float] | None = None,
    observed_quotas: Mapping[str, float] | None = None,
    horizon_downside_cash_by_candidate: Mapping[str, dict[int, float]] | None = None,
    parameters_on_priors: tuple[str, ...] = (),
    parameters_unmeasured: frozenset[str] = frozenset(),
    state: SimulationState | None = None,
) -> tuple[CandidateEvaluationCard, ...]:
    """Mark every simulated candidate eligible or vetoed; select nothing."""

    # What continuing already risks, so a candidate is not warned about danger
    # the company is in regardless of what it chooses.
    continuation_going_concern = next(
        (
            candidate.going_concern_failure_probability
            for candidate in evaluations
            if candidate.action.name == "continuation"
        ),
        None,
    )
    continuation_expected = next(
        (
            candidate.expected_ending_cash
            for candidate in evaluations
            if candidate.action.name == "continuation"
        ),
        None,
    )
    continuation_bankruptcy = next(
        (
            candidate.bankruptcy_probability
            for candidate in evaluations
            if candidate.action.name == "continuation"
        ),
        None,
    )
    cards: list[CandidateEvaluationCard] = []
    for candidate in evaluations:
        candidate_id = candidate.action.name
        plan = plans_by_candidate[candidate_id]
        operational = is_operational_candidate(candidate_id)
        veto_codes: list[str] = []
        warnings: list[str] = []
        if candidate.bankruptcy_probability > MAX_ACCEPTABLE_BANKRUPTCY_PROBABILITY:
            # Insolvency the company is already heading for is not a reason to
            # forbid the interventions that could avert it. As a flat veto this
            # gate closed the only door out of a burn trajectory: with no
            # revenue every candidate breaches the threshold, so the sole
            # eligible option became continuation — which burns on, keeps the
            # probability high, and vetoes the next week's escape too. One run
            # spent six consecutive weeks naming "zero exposure" as its binding
            # constraint while its own top-ranked marketing probe sat vetoed.
            # So the comparison is against continuation: risk the company
            # carries anyway is reported, risk a candidate *adds* is vetoed.
            carried = (
                operational
                or continuation_bankruptcy is None
                or candidate.bankruptcy_probability
                <= continuation_bankruptcy + BANKRUPTCY_RISK_TOLERANCE
            )
            if carried:
                warnings.append(
                    "bankruptcy_gate_exceeded_operational_fallback"
                    if operational
                    else "insolvency_risk_carried"
                )
            else:
                veto_codes.append("bankruptcy_gate")
        if (
            candidate.going_concern_failure_probability
            > MAX_ACCEPTABLE_GOING_CONCERN_FAILURE_PROBABILITY
        ):
            # Never a veto. Whether the company stays a going concern is the
            # judgement this Executive exists to make, and the card carries the
            # probability for it to weigh. As a veto it froze one run into
            # twelve consecutive weeks with a single eligible candidate: the
            # gate tightened as the company weakened, so it forbade precisely
            # the interventions that might have reversed the decline, and
            # relaxed only once there was nothing left to save. Insolvency
            # stays a hard veto because it is terminal; this is a forecast.
            carried = (
                inherited_going_concern_failure
                or operational
                or (
                    continuation_going_concern is not None
                    and candidate.going_concern_failure_probability
                    <= continuation_going_concern + 0.05
                )
            )
            warnings.append(
                "going_concern_risk_carried" if carried else "going_concern_risk_added"
            )
        veto_codes.extend(
            configuration_completeness_violations(
                plan,
                observed_prices=observed_prices,
                observed_quotas=observed_quotas,
            )
        )
        if getattr(candidate.action, "enterprise_engage", False) and (
            state is None or state.measured_quality_floor_enterprise is None
        ):
            # The forecast shows no seats because nothing is known about what
            # these buyers require, not because none would be won.
            warnings.append("enterprise_outcome_unmeasured")
        if state is not None and _is_quality_side(candidate.action):
            floor = effective_participation_floor(state)
            if floor is None:
                # The lever's payoff is unquantified until a participation floor
                # is bought: the forecast holds no cliff to price, and saying so
                # makes measuring the floor a reason to act, not a blind spot.
                warnings.append("participation_floor_unmeasured")
            elif _candidate_delivered_quality(state, candidate.action) < floor:
                # The number customers judge stays under the bar they reported;
                # whatever else this candidate buys, it does not buy conversion.
                warnings.append("delivered_quality_below_measured_floor")
        window_downside_cost: float | None = None
        if getattr(candidate.action, "research_project_tier", None):
            catalog_entry = None
            if state is not None:
                catalog_entry = next(
                    (
                        entry
                        for entry in getattr(state, "research_catalog", ())
                        if entry.tier == candidate.action.research_project_tier
                    ),
                    None,
                )
            if catalog_entry is not None:
                # The programme's downside is not open-ended: the environment's
                # own listing prices it exactly. Stated as the commitment-window
                # downside so an irreversible bet with a known worst case is
                # weighed as one, not as unbounded risk.
                window_downside_cost = catalog_entry.cost
                warnings.append("research_downside_bounded_by_catalog_cost")
            else:
                # No listing has been read, so neither the charge nor the return
                # is quantified. Naming that makes reading the catalog a reason
                # to act, not a silent gap.
                warnings.append("research_catalog_unread_cost_and_return_unquantified")
        if plan.proposal_kind == "experiment":
            experiment_window_cost = _commitment_window_downside_cost(
                plan,
                candidate_id=candidate_id,
                horizon_downside_cash_by_candidate=horizon_downside_cash_by_candidate,
            )
            if experiment_window_cost is not None:
                window_downside_cost = max(
                    window_downside_cost or 0.0, experiment_window_cost
                )
            if (
                experiment_window_cost is not None
                and experiment_window_cost > experiment_budget
            ):
                # A fact for the Executive, never a veto: the projected downside
                # of learning inside this commitment window exceeds the advisory
                # research budget. Solvency gates still bind independently.
                warnings.append("experiment_budget_pressure")
            veto_codes.extend(
                assess_experiment(
                    plan,
                    portfolio=portfolio,
                )
            )
        cards.append(
            CandidateEvaluationCard(
                candidate_id=candidate_id,
                plan_hash=plan.semantic_hash,
                hypothesis_id=plan.hypothesis_id,
                eligible=not veto_codes,
                veto_codes=tuple(dict.fromkeys(veto_codes)),
                expected_terminal_cash=candidate.expected_ending_cash,
                downside_terminal_cash=candidate.downside_ending_cash,
                upside_terminal_cash=candidate.upside_ending_cash,
                bankruptcy_probability=candidate.bankruptcy_probability,
                going_concern_failure_probability=(
                    candidate.going_concern_failure_probability
                ),
                horizon_expected_cash=dict(
                    horizon_cash_by_candidate.get(candidate_id, {})
                ),
                downside_cost_commitment_window=window_downside_cost,
                forecast_rests_on_priors=parameters_on_priors,
                levers_on_priors=levers_on_prior_only_parameters(
                    candidate.action, parameters_on_priors
                ),
                terminal_cash_versus_continuation=(
                    None
                    if continuation_expected is None
                    else candidate.expected_ending_cash - continuation_expected
                ),
                support_and_assumption_warnings=tuple(warnings),
            )
        )
    if not any(card.eligible for card in cards):
        # The company must still act. Promote the minimum-bankruptcy operational
        # candidate as an explicit emergency; experiments are never resurrected.
        emergency = min(
            (
                card
                for card in cards
                if is_operational_candidate(card.candidate_id)
            ),
            key=lambda card: card.bankruptcy_probability,
            default=None,
        )
        if emergency is not None:
            cards = [
                card.model_copy(
                    update={
                        "eligible": True,
                        "veto_codes": (),
                        "support_and_assumption_warnings": (
                            *card.support_and_assumption_warnings,
                            "emergency_minimum_risk",
                        ),
                    }
                )
                if card.candidate_id == emergency.candidate_id
                else card
                for card in cards
            ]
    return tuple(cards)


def fallback_candidate_id(cards: tuple[CandidateEvaluationCard, ...]) -> str:
    """The safest operational candidate — never, under any branch, an experiment."""

    def safest(pool: list[CandidateEvaluationCard]) -> str:
        return min(pool, key=lambda card: card.bankruptcy_probability).candidate_id

    eligible_operational = [
        card
        for card in cards
        if card.eligible and is_operational_candidate(card.candidate_id)
    ]
    for card in eligible_operational:
        if card.candidate_id == "continuation":
            return card.candidate_id
    if eligible_operational:
        return safest(eligible_operational)
    non_experiment = [
        card
        for card in cards
        if card.eligible and not card.candidate_id.startswith("executive_experiment_")
    ]
    if non_experiment:
        return safest(non_experiment)
    operational_any = [
        card for card in cards if is_operational_candidate(card.candidate_id)
    ]
    if operational_any:
        return safest(operational_any)
    return safest(
        [
            card
            for card in cards
            if not card.candidate_id.startswith("executive_experiment_")
        ]
        or list(cards)
    )


def quality_position_facts(state: SimulationState) -> dict:
    """The delivered-quality position: what customers judge versus the bar.

    Delivered per plan is base quality composed with the environment's own
    published tier multipliers; the floors are purchased measurements, reported
    as absent — never zero — until bought.
    """

    return {
        "delivered_quality_by_plan": {
            "A": min(
                1.0,
                max(0.0, state.product_quality)
                * MODEL_TIER_QUALITY_MULTIPLIER[state.model_tier_a],
            ),
            "B": min(
                1.0,
                max(0.0, state.product_quality)
                * MODEL_TIER_QUALITY_MULTIPLIER[state.model_tier_b],
            ),
            "C": min(
                1.0,
                max(0.0, state.product_quality)
                * MODEL_TIER_QUALITY_MULTIPLIER[state.model_tier_c],
            ),
        },
        "measured_quality_floors": {
            "individual": state.measured_quality_floor_individual,
            "enterprise": state.measured_quality_floor_enterprise,
        },
        # Where the run's own churn has proven the bar to be, regardless of
        # what any floor said at purchase time. None = nothing revealed yet.
        "churn_revealed_bar_lower_bound": state.revealed_quality_bar_lower_bound,
    }


def _selection_brief(
    *,
    week: int,
    portfolio: StrategicPortfolio | None,
    evaluation_set: CandidateEvaluationSet,
    diagnostics: tuple[str, ...],
    cash: float = 0.0,
    quality_position: dict | None = None,
) -> dict:
    # What carrying on is projected to cost between now and the horizon. Without
    # it the only magnitude on the page is the price of acting, and the cheapest
    # candidate wins by default however little it changes.
    continuation = next(
        (
            card
            for card in evaluation_set.cards
            if card.candidate_id == "continuation"
        ),
        None,
    )
    return {
        "week": week,
        "cost_of_standing_still": (
            None if continuation is None else cash - continuation.expected_terminal_cash
        ),
        "objective": (
            "risk-adjusted terminal cash subject to solvency and going concern"
        ),
        "portfolio": (
            {
                "binding_constraint": portfolio.binding_constraint,
                "active_hypothesis_ids": list(portfolio.active_hypothesis_ids),
                "remaining_experiment_budget": portfolio.remaining_experiment_budget,
                "unresolved_questions": list(portfolio.unresolved_questions),
            }
            if portfolio is not None
            else None
        ),
        "evaluation_set_hash": evaluation_set.set_hash,
        "eligible_candidate_ids": list(evaluation_set.eligible_candidate_ids()),
        "candidates": [card.model_dump(mode="json") for card in evaluation_set.cards],
        "diagnostics": list(diagnostics),
        "quality_position": quality_position,
    }


async def run_executive_selection(
    *,
    run_id: UUID,
    week: int,
    executive: object,
    authority: ExecutiveAuthorityContext,
    cards: tuple[CandidateEvaluationCard, ...],
    diagnostics: tuple[str, ...] = (),
    cash: float = 0.0,
    known_insight_identities: Container[str] = frozenset(),
    learned_costs: Mapping[str, float] | None = None,
    quality_position: dict | None = None,
) -> SelectionOutcome:
    """Persist the immutable evaluation set, then let Gemini choose from it.

    Replay reuses the persisted choice without another provider call. Provider
    failure or a twice-invalid choice falls back to safe continuation and emits
    the corresponding audit event.
    """

    portfolio = authority.portfolio
    evaluation_set = CandidateEvaluationSet(
        id=candidate_evaluation_set_id(run_id, week),
        run_id=run_id,
        week=week,
        portfolio_hash=(
            portfolio.portfolio_hash if portfolio is not None else NO_PORTFOLIO_HASH
        ),
        cards=cards,
        created_at=utc_now(),
    )
    repository = authority.strategy_repository
    try:
        evaluation_set = await repository.append_candidate_evaluation_set(evaluation_set)
    except ConflictError:
        stored = await repository.get_candidate_evaluation_set(run_id, week)
        if stored is None or stored.set_hash != evaluation_set.set_hash:
            await authority.emit_event(
                "decision.evaluation_set_conflict",
                {
                    "week": week,
                    "computed_set_hash": evaluation_set.set_hash,
                    "stored_set_hash": stored.set_hash if stored else None,
                },
            )
            stored_ids = (
                {card.candidate_id for card in stored.cards} if stored else None
            )
            computed_ids = {card.candidate_id for card in cards}
            if stored is not None and stored_ids == computed_ids:
                # The persisted set is the immutable artifact of record and the
                # replay proposes the same candidates, merely re-priced. Losing
                # the Executive's whole turn to a hash mismatch cost one run two
                # of the five weeks of its collapse; select against the stored
                # set instead, with the divergence on the audit trail above.
                evaluation_set = stored
            else:
                fallback = fallback_candidate_id(cards)
                return SelectionOutcome(
                    candidate_id=fallback,
                    selection_reason_code="evaluation_set_conflict_safe_continuation",
                    selection_reason=(
                        "The persisted evaluation set diverged from this replay "
                        "beyond a re-pricing of the same candidates; executing "
                        "the safe operational candidate."
                    ),
                    choice=None,
                )
        else:
            evaluation_set = stored

    existing = await repository.get_executive_choice(run_id, week)
    if existing is not None:
        validate_choice_against_set(existing, evaluation_set)
        return SelectionOutcome(
            candidate_id=existing.selected_candidate_id,
            selection_reason_code="executive_candidate_selected",
            selection_reason=existing.decision_thesis[:1_000],
            choice=existing,
        )

    selector = getattr(executive, "select_candidate", None)
    eligible_ids = evaluation_set.eligible_candidate_ids()
    if not callable(selector):
        fallback = fallback_candidate_id(cards)
        return SelectionOutcome(
            candidate_id=fallback,
            selection_reason_code="executive_selection_unsupported_safe_continuation",
            selection_reason=(
                "The decision engine does not implement candidate selection; "
                "executing the safe operational candidate."
            ),
            choice=None,
        )

    brief = _selection_brief(
        cash=cash,
        week=week,
        portfolio=portfolio,
        evaluation_set=evaluation_set,
        diagnostics=diagnostics,
        quality_position=quality_position,
    )
    last_error: str | None = None
    for attempt in range(2):
        output = None
        for provider_attempt in range(3):
            try:
                output = await selector(brief=brief)
                break
            except Exception as error:
                # One transient provider failure used to surrender the whole
                # week to safe continuation — twice, in the middle of the one
                # collapse the Executive existed to manage. The turn is worth
                # three attempts before the harness decides in its place.
                await authority.emit_event(
                    "executive.unavailable",
                    {
                        "week": week,
                        "stage": "candidate_selection",
                        "attempt": provider_attempt + 1,
                        "error": str(error)[:500],
                    },
                )
                if provider_attempt < 2:
                    await asyncio.sleep(2.0 * (provider_attempt + 1))
        if output is None:
            fallback = fallback_candidate_id(cards)
            return SelectionOutcome(
                candidate_id=fallback,
                selection_reason_code="executive_unavailable_safe_continuation",
                selection_reason=(
                    "The Executive provider stayed unavailable across three "
                    "selection attempts; executing the safe operational candidate."
                ),
                choice=None,
            )
        if output.selected_candidate_id in eligible_ids:
            information_requests, information_diagnostics = admit_information_requests(
                output.information_requests,
                cash=cash,
                recent_identities=known_insight_identities,
                learned_costs=learned_costs,
            )
            if information_diagnostics:
                await authority.emit_event(
                    "information.requests_gated",
                    {"week": week, "diagnostics": list(information_diagnostics)},
                )
            choice = ExecutiveChoice(
                id=executive_choice_id(run_id, week),
                run_id=run_id,
                week=week,
                evaluation_set_id=evaluation_set.id,
                evaluation_set_hash=evaluation_set.set_hash,
                selected_candidate_id=output.selected_candidate_id,
                decision_thesis=output.decision_thesis,
                evidence_refs=tuple(output.evidence_refs),
                rejected_candidate_ids=tuple(
                    candidate_id
                    for candidate_id in dict.fromkeys(output.rejected_candidate_ids)
                    if candidate_id != output.selected_candidate_id
                ),
                stop_or_pivot_condition=output.stop_or_pivot_condition,
                created_at=utc_now(),
            )
            try:
                choice = await repository.append_executive_choice(choice)
            except ConflictError:
                stored_choice = await repository.get_executive_choice(run_id, week)
                if stored_choice is None:
                    raise
                validate_choice_against_set(stored_choice, evaluation_set)
                choice = stored_choice
            await authority.emit_event(
                "executive_candidate_selected",
                {
                    "week": week,
                    "selected_candidate_id": choice.selected_candidate_id,
                    "evaluation_set_hash": choice.evaluation_set_hash,
                    "rejected_candidate_ids": list(choice.rejected_candidate_ids),
                    "stop_or_pivot_condition": choice.stop_or_pivot_condition,
                    "attempt": attempt + 1,
                },
            )
            return SelectionOutcome(
                candidate_id=choice.selected_candidate_id,
                selection_reason_code="executive_candidate_selected",
                selection_reason=choice.decision_thesis[:1_000],
                choice=choice,
                information_requests=information_requests,
                data_queries=tuple(output.data_queries),
                journal=output.journal,
            )
        last_error = output.selected_candidate_id
        brief = {
            **brief,
            "previous_invalid_choice": {
                "selected_candidate_id": output.selected_candidate_id,
                "error": "the selected candidate is not an eligible candidate_id",
            },
        }

    await authority.emit_event(
        "executive.choice_invalid",
        {
            "week": week,
            "invalid_candidate_id": last_error,
            "eligible_candidate_ids": list(eligible_ids),
        },
    )
    fallback = fallback_candidate_id(cards)
    return SelectionOutcome(
        candidate_id=fallback,
        selection_reason_code="executive_choice_invalid_safe_continuation",
        selection_reason=(
            "The Executive twice selected an ineligible candidate; "
            "executing the safe operational candidate."
        ),
        choice=None,
    )


def commitment_verdict_from_choice(
    program: ExperimentProgram,
    *,
    week: int,
    chosen_candidate_id: str,
    commitment_candidate_id: str | None,
) -> tuple[CommitmentReviewVerdict, str]:
    """Read the weekly verdict off what the Executive chose to run.

    The verdict is recorded, not decided here: continuing, rolling back,
    adopting the treatment or moving on are all the Executive's calls, and each
    leaves a different mark on the commitment's history.
    """

    if chosen_candidate_id.startswith("experiment_adopt_"):
        return (
            CommitmentReviewVerdict.ADOPTED,
            "The matured treatment was adopted as the operating configuration.",
        )
    if chosen_candidate_id.startswith("experiment_revert_"):
        return (
            CommitmentReviewVerdict.REVERT,
            "The committed window closed and the treatment was rolled back.",
        )
    if commitment_candidate_id is not None and (
        chosen_candidate_id == commitment_candidate_id
    ):
        if week >= program.minimum_maturity_week:
            return (
                CommitmentReviewVerdict.MATURE_AND_PROBE,
                "The commitment reached its maturity window and the probe ran.",
            )
        return (
            CommitmentReviewVerdict.CONTINUE,
            "The commitment continued inside its build window.",
        )
    return (
        CommitmentReviewVerdict.ABANDONED,
        "The Executive moved to a different action before the window closed.",
    )


def deterministic_commitment_review(
    program: ExperimentProgram,
    *,
    week: int,
    reverting: bool,
) -> tuple[CommitmentReviewVerdict, str]:
    """The deterministic part of the weekly commitment review.

    Safety stops (budget exhaustion, cash floors) are enforced upstream by the
    reversion planner; here the record states which phase the commitment is in.
    """

    if reverting:
        return (
            CommitmentReviewVerdict.REVERT,
            "The committed window ended or its downside budget was exhausted; "
            "executing the explicit reversion.",
        )
    if week >= program.minimum_maturity_week:
        return (
            CommitmentReviewVerdict.MATURE_AND_PROBE,
            "The commitment reached its maturity window; running the committed probe.",
        )
    return (
        CommitmentReviewVerdict.CONTINUE,
        "The commitment is inside its build window and its weekly safety review passed.",
    )


async def record_commitment_review(
    *,
    run_id: UUID,
    week: int,
    program: ExperimentProgram,
    verdict: CommitmentReviewVerdict,
    reason: str,
    authority: ExecutiveAuthorityContext,
) -> CommitmentReview:
    review = CommitmentReview(
        id=commitment_review_id(run_id, program.commitment_id, week),
        run_id=run_id,
        commitment_id=program.commitment_id,
        week=week,
        verdict=verdict,
        reason=reason,
        created_at=utc_now(),
    )
    return await authority.strategy_repository.append_commitment_review(review)
