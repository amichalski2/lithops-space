"""Generate bounded strategies and rank them across uncertain plausible worlds."""

from __future__ import annotations

from enum import StrEnum
from math import sqrt
from statistics import fmean

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lithops.domain.world_model import WorldModelVersion
from lithops.simulator.engine import simulate
from lithops.simulator.invariants import evaluate_simulation_action
from lithops.simulator.models import RolloutOutcome, SimulationAction, SimulationState

MAX_ACCEPTABLE_BANKRUPTCY_PROBABILITY = 0.10
MAX_ACCEPTABLE_GOING_CONCERN_FAILURE_PROBABILITY = 0.10
EXPECTED_CASH_WEIGHT = 0.45
DOWNSIDE_CASH_WEIGHT = 0.55
BANKRUPTCY_PENALTY = 1.50
CONTROLLED_EXPLORATION_MIN_INCREMENT_WEEKLY = 1_000.0
CONTROLLED_EXPLORATION_MAX_DEVELOPMENT_INCREMENT_WEEKLY = 5_000.0
CONTROLLED_EXPLORATION_DURATION_WEEKS = 1
DELAYED_DEVELOPMENT_EXPERIMENT_WEEKS = 5
CONTROLLED_EXPLORATION_LEAD_PROMOTION_FRACTION = 0.20
MAX_LEAD_PROMOTION_FRACTION = 0.25
MAX_CONTROLLED_EXPLORATION_DOWNSIDE_FRACTION = 0.005
MAX_CONTROLLED_EXPLORATION_DOWNSIDE_ABSOLUTE = 5_000.0
MAX_DELAYED_EXPERIMENT_DOWNSIDE_FRACTION = 0.03
MAX_DELAYED_EXPERIMENT_DOWNSIDE_ABSOLUTE = 30_000.0
MAX_SUPPORT_FRONTIER_DOWNSIDE_FRACTION = 0.30
MAX_SUPPORT_FRONTIER_DOWNSIDE_ABSOLUTE = 300_000.0
EXPLORATION_CONTINUITY_CUSTOMERS = 1.0
EXPLORATION_WILSON_Z = 1.96
QUALITY_NULL_MAX_CONVERSION_UPPER = 0.05
QUALITY_NULL_MIN_DISTINCT_REGIMES = 3
QUALITY_NULL_MIN_BAND_SPAN = 2


class RobustnessLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CandidateSimulation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    action: SimulationAction
    expected_ending_cash: float
    downside_ending_cash: float
    # Upper decile of the same rollouts. Without it a widened parameter prior
    # widens only the visible downside, and an unmeasured lever reads as pure
    # risk — the asymmetry that starves exploration.
    upside_ending_cash: float | None = None
    bankruptcy_probability: float = Field(ge=0.0, le=1.0)
    going_concern_failure_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    expected_customer_growth: float
    robustness: RobustnessLevel
    robust_utility: float
    rollout_count: int = Field(ge=1)


class StrategySearchResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidates: tuple[CandidateSimulation, ...] = Field(min_length=1)
    selected: CandidateSimulation
    selection_reason_code: str = "survival_gated_robust_utility"
    selection_reason: str = Field(min_length=1, max_length=1_000)


class NoViableStrategyError(RuntimeError):
    """Every simulated action ends the business with certainty."""

    def __init__(self, candidates: tuple[CandidateSimulation, ...]) -> None:
        self.candidate_risks = tuple(
            (candidate.action.name, candidate.going_concern_failure_probability)
            for candidate in candidates
        )
        summary = ", ".join(
            f"{name}={risk:.3f}" for name, risk in self.candidate_risks
        )
        super().__init__(
            "no viable strategy: every candidate has certain going-concern failure; "
            + summary
        )


class FunnelRegimeEvidence(BaseModel):
    """Observed funnel outcomes inside one covariate-support region."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    quality_band: int = Field(ge=0, le=9)
    leads: float = Field(ge=0.0)
    conversions: float = Field(ge=0.0)
    weeks: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_counts(self) -> FunnelRegimeEvidence:
        if self.conversions > self.leads:
            raise ValueError("funnel regime conversions cannot exceed leads")
        return self


class ExplorationMemory(BaseModel):
    """Immutable evidence of experiments already committed in this run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempted_strategies: tuple[str, ...] = ()
    attempted_hypothesis_keys: tuple[str, ...] = ()
    revert_marketing_spend: float | None = Field(default=None, ge=0.0)
    revert_development_spend: float | None = Field(default=None, ge=0.0)
    revert_lead_promotion_monthly: float | None = Field(default=None, ge=0.0)
    funnel_regime_evidence: tuple[FunnelRegimeEvidence, ...] = ()


def experiment_hypothesis_key(
    hypothesis_id: str,
    experiment_control: str,
    evidence_regime: str,
) -> str:
    """Stable identity of one causal test in one materially distinct evidence regime."""

    return f"{hypothesis_id}|{experiment_control}|{evidence_regime}"


class ExplorationAdmission(BaseModel):
    """Why an information-gathering action may or may not enter the pool."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    admitted: bool
    reason_code: str
    candidate_strategy: str | None = None
    conversion_lower: float = Field(default=0.0, ge=0.0, le=1.0)
    conversion_upper: float = Field(default=1.0, ge=0.0, le=1.0)
    lower_expected_conversions: float = Field(default=0.0, ge=0.0)
    upper_expected_conversions: float = Field(default=0.0, ge=0.0)
    attempted_strategies: tuple[str, ...] = ()
    evidence_quality_band: int | None = Field(default=None, ge=0, le=9)
    evidence_leads: float = Field(default=0.0, ge=0.0)
    evidence_conversions: float = Field(default=0.0, ge=0.0)
    active_commitment_strategy: str | None = None
    active_commitment_phase: str | None = Field(
        default=None, pattern=r"^(?:build|probe)$"
    )

    def as_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json")


def _bounded_spend(value: float) -> float:
    return max(0.0, min(1_000_000.0, value))


def _wilson_interval(successes: float, trials: float) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 1.0
    rate = successes / trials
    z2 = EXPLORATION_WILSON_Z**2
    denominator = 1.0 + z2 / trials
    centre = (rate + z2 / (2.0 * trials)) / denominator
    margin = (
        EXPLORATION_WILSON_Z
        * sqrt(rate * (1.0 - rate) / trials + z2 / (4.0 * trials**2))
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def assess_controlled_exploration(
    state: SimulationState,
    memory: ExplorationMemory | None = None,
) -> ExplorationAdmission:
    memory = memory or ExplorationMemory()
    attempted = memory.attempted_strategies
    expected_weekly_losses = state.customers * state.churn_rate
    if state.weekly_acquisition > expected_weekly_losses:
        return ExplorationAdmission(
            admitted=False,
            reason_code="funnel_replaces_expected_churn",
            attempted_strategies=attempted,
        )
    quality_band = min(9, max(0, int(state.product_quality * 10.0)))
    regime = next(
        (
            item
            for item in memory.funnel_regime_evidence
            if item.quality_band == quality_band
        ),
        None,
    )
    # A fresh run has no durable regime memory yet, but the current observation
    # is still valid evidence.  Never fall back to lifetime totals here: they
    # pool observations collected under materially different product quality.
    regime_leads = (
        regime.leads if regime is not None else max(0.0, state.weekly_leads)
    )
    regime_conversions = (
        regime.conversions
        if regime is not None
        else max(0.0, state.weekly_conversions)
    )
    if regime_leads <= 0:
        strategy = f"controlled_exploration_marketing_q{quality_band}"
        if strategy in attempted:
            return ExplorationAdmission(
                admitted=False,
                reason_code="marketing_experiment_already_attempted",
                attempted_strategies=attempted,
                evidence_quality_band=quality_band,
            )
        return ExplorationAdmission(
            admitted=True,
            reason_code="funnel_has_no_observational_evidence",
            candidate_strategy=strategy,
            upper_expected_conversions=(
                max(
                    CONTROLLED_EXPLORATION_MIN_INCREMENT_WEEKLY,
                    state.marketing_spend * 0.25,
                )
                / 200.0
                * CONTROLLED_EXPLORATION_DURATION_WEEKS
            ),
            attempted_strategies=attempted,
            evidence_quality_band=quality_band,
        )
    lower, upper = _wilson_interval(regime_conversions, regime_leads)
    regime_weeks = regime.weeks if regime is not None else 1
    weekly_leads = max(state.weekly_leads, regime_leads / regime_weeks)
    exposure = weekly_leads * CONTROLLED_EXPLORATION_DURATION_WEEKS
    lower_expected = exposure * lower
    upper_expected = exposure * upper
    decision_sensitive = (
        lower_expected < EXPLORATION_CONTINUITY_CUSTOMERS <= upper_expected
    )
    promotion_strategy = f"controlled_exploration_lead_promotion_q{quality_band}"
    statistically_resolved_null_regimes = tuple(
        item
        for item in memory.funnel_regime_evidence
        if item.conversions <= 0.0
        and item.leads > 0.0
        and _wilson_interval(item.conversions, item.leads)[1]
        <= QUALITY_NULL_MAX_CONVERSION_UPPER
    )
    quality_null_spans_regimes = (
        len(statistically_resolved_null_regimes)
        >= QUALITY_NULL_MIN_DISTINCT_REGIMES
        and max(
            item.quality_band for item in statistically_resolved_null_regimes
        )
        - min(item.quality_band for item in statistically_resolved_null_regimes)
        >= QUALITY_NULL_MIN_BAND_SPAN
    )
    if quality_null_spans_regimes and promotion_strategy not in attempted:
        return ExplorationAdmission(
            admitted=True,
            reason_code="cross_regime_quality_null_requires_price_probe",
            candidate_strategy=promotion_strategy,
            conversion_lower=lower,
            conversion_upper=upper,
            lower_expected_conversions=lower_expected,
            upper_expected_conversions=upper_expected,
            attempted_strategies=attempted,
            evidence_quality_band=quality_band,
            evidence_leads=regime_leads,
            evidence_conversions=regime_conversions,
        )
    if (
        regime_conversions <= 0
        and promotion_strategy not in attempted
        and decision_sensitive
    ):
        return ExplorationAdmission(
            admitted=True,
            reason_code="lead_promotion_uncertainty_changes_continuity_decision",
            candidate_strategy=promotion_strategy,
            conversion_lower=lower,
            conversion_upper=upper,
            lower_expected_conversions=lower_expected,
            upper_expected_conversions=upper_expected,
            attempted_strategies=attempted,
            evidence_quality_band=quality_band,
            evidence_leads=regime_leads,
            evidence_conversions=regime_conversions,
        )
    strategy = f"controlled_exploration_development_q{quality_band}"
    if strategy in attempted:
        return ExplorationAdmission(
            admitted=False,
            reason_code="development_experiment_already_attempted",
            attempted_strategies=attempted,
        )
    higher_quality_has_exposure = any(
        item.quality_band > quality_band and item.leads > 0
        for item in memory.funnel_regime_evidence
    )
    support_frontier_open = (
        regime_conversions <= 0
        and quality_band < 9
        and not higher_quality_has_exposure
    )
    admitted = decision_sensitive or support_frontier_open
    return ExplorationAdmission(
        admitted=admitted,
        reason_code=(
            "conversion_uncertainty_changes_continuity_decision"
            if decision_sensitive
            else (
                "higher_quality_support_could_change_continuity_decision"
                if support_frontier_open
                else "conversion_uncertainty_cannot_change_continuity_decision"
            )
        ),
        candidate_strategy=strategy if admitted else None,
        conversion_lower=lower,
        conversion_upper=upper,
        lower_expected_conversions=lower_expected,
        upper_expected_conversions=upper_expected,
        attempted_strategies=attempted,
        evidence_quality_band=quality_band,
        evidence_leads=regime_leads,
        evidence_conversions=regime_conversions,
    )


def generate_candidate_actions(
    state: SimulationState,
    exploration_memory: ExplorationMemory | None = None,
) -> tuple[SimulationAction, ...]:
    """A small P0 strategy set; Gemini may later tune values inside these bounds."""

    memory = exploration_memory or ExplorationMemory()
    catalog_price = state.effective_catalog_price_per_customer_weekly
    # A selected one-week experiment is a temporary intervention, not the new
    # operating baseline.  On the following decision all ordinary strategies
    # are therefore built from its recorded reversion values.  This makes the
    # commitment visible in receipts and prevents spend ratchets.
    baseline_marketing_spend = (
        state.marketing_spend
        if memory.revert_marketing_spend is None
        else memory.revert_marketing_spend
    )
    baseline_development_spend = (
        state.development_spend
        if memory.revert_development_spend is None
        else memory.revert_development_spend
    )
    baseline_lead_promotion = (
        state.lead_promotion_monthly
        if memory.revert_lead_promotion_monthly is None
        else memory.revert_lead_promotion_monthly
    )
    held_tiers = {
        "model_tier_a": state.model_tier_a,
        "model_tier_b": state.model_tier_b,
        "model_tier_c": state.model_tier_c,
    }
    candidates = [
        SimulationAction(
            name="aggressive_growth",
            price_per_customer_weekly=catalog_price,
            marketing_spend=_bounded_spend(
                max(1_000.0, baseline_marketing_spend * 1.5)
            ),
            development_spend=_bounded_spend(
                max(1_000.0, baseline_development_spend * 1.3)
            ),
            operations_spend=state.operations_spend,
            segment_focus=1.15,
            lead_promotion_monthly=baseline_lead_promotion,
            **held_tiers,
        ),
        SimulationAction(
            name="balanced_growth",
            price_per_customer_weekly=catalog_price,
            marketing_spend=_bounded_spend(baseline_marketing_spend),
            development_spend=_bounded_spend(baseline_development_spend),
            operations_spend=state.operations_spend,
            segment_focus=1.0,
            lead_promotion_monthly=baseline_lead_promotion,
            **held_tiers,
        ),
        SimulationAction(
            name="cash_preservation",
            price_per_customer_weekly=catalog_price * 1.05,
            marketing_spend=_bounded_spend(baseline_marketing_spend * 0.6),
            development_spend=_bounded_spend(baseline_development_spend * 0.5),
            operations_spend=state.operations_spend,
            segment_focus=0.9,
            lead_promotion_monthly=baseline_lead_promotion,
            **held_tiers,
        ),
        SimulationAction(
            name="pricing_efficiency",
            price_per_customer_weekly=catalog_price * 1.10,
            marketing_spend=_bounded_spend(baseline_marketing_spend * 0.8),
            development_spend=_bounded_spend(baseline_development_spend),
            operations_spend=state.operations_spend,
            segment_focus=1.05,
            lead_promotion_monthly=baseline_lead_promotion,
            **held_tiers,
        ),
    ]
    admission = assess_controlled_exploration(state, memory)
    if admission.admitted and admission.candidate_strategy is not None:
        marketing_spend = baseline_marketing_spend
        development_spend = baseline_development_spend
        if admission.candidate_strategy.startswith("controlled_exploration_marketing_"):
            marketing_spend += max(
                CONTROLLED_EXPLORATION_MIN_INCREMENT_WEEKLY,
                baseline_marketing_spend * 0.25,
            )
        elif admission.candidate_strategy.startswith("controlled_exploration_development_"):
            development_spend += min(
                CONTROLLED_EXPLORATION_MAX_DEVELOPMENT_INCREMENT_WEEKLY,
                max(1_000.0, state.cash * 0.005),
            )
        lead_promotion = baseline_lead_promotion
        lead_promotion_until_week = None
        lead_promotion_after_experiment = None
        if admission.candidate_strategy.startswith(
            "controlled_exploration_lead_promotion_"
        ):
            entry_price_monthly = state.entry_price_monthly or (
                catalog_price * 30.0 / 7.0
            )
            lead_promotion = min(
                entry_price_monthly * MAX_LEAD_PROMOTION_FRACTION,
                baseline_lead_promotion
                + entry_price_monthly
                * CONTROLLED_EXPLORATION_LEAD_PROMOTION_FRACTION,
            )
            lead_promotion_until_week = (
                state.week + CONTROLLED_EXPLORATION_DURATION_WEEKS
            )
            lead_promotion_after_experiment = baseline_lead_promotion
        candidates.append(
            SimulationAction(
                name=admission.candidate_strategy,
                price_per_customer_weekly=catalog_price,
                marketing_spend=_bounded_spend(marketing_spend),
                development_spend=_bounded_spend(development_spend),
                operations_spend=state.operations_spend,
                segment_focus=1.0,
                **held_tiers,
                marketing_spend_until_week=(
                    state.week + CONTROLLED_EXPLORATION_DURATION_WEEKS
                ),
                development_spend_until_week=(
                    state.week
                    + (
                        DELAYED_DEVELOPMENT_EXPERIMENT_WEEKS
                        if admission.candidate_strategy.startswith(
                            "controlled_exploration_development_"
                        )
                        else CONTROLLED_EXPLORATION_DURATION_WEEKS
                    )
                ),
                marketing_spend_after_experiment=baseline_marketing_spend,
                development_spend_after_experiment=baseline_development_spend,
                lead_promotion_monthly=lead_promotion,
                lead_promotion_until_week=lead_promotion_until_week,
                lead_promotion_after_experiment=lead_promotion_after_experiment,
            )
        )
    if (
        state.customers > 0
        and state.operating_cost_per_customer_weekly
        > state.price_per_customer_weekly
        and any(tier > 1 for tier in held_tiers.values())
    ):
        # Cost recovery is a separate, attributable intervention. It changes no
        # other control, so a lower tier must earn selection through its simulated
        # cash/quality trade-off instead of being imposed after ranking.
        candidates.append(
            SimulationAction(
                name="unit_cost_recovery",
                price_per_customer_weekly=catalog_price,
                marketing_spend=_bounded_spend(baseline_marketing_spend),
                development_spend=_bounded_spend(baseline_development_spend),
                operations_spend=state.operations_spend,
                model_tier_a=max(1, state.model_tier_a - 1),
                model_tier_b=max(1, state.model_tier_b - 1),
                model_tier_c=max(1, state.model_tier_c - 1),
                lead_promotion_monthly=baseline_lead_promotion,
            )
        )
    return tuple(candidates)


def _lower_quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = int((len(ordered) - 1) * probability)
    return ordered[index]


def cash_robust_utility(
    *,
    initial_cash: float,
    expected_ending_cash: float,
    downside_ending_cash: float,
    bankruptcy_probability: float,
) -> float:
    """Score the benchmark objective without substituting growth for cash.

    Customer, revenue, churn, and capacity outcomes remain available as explicit
    evidence and constraints. They must not silently improve candidate ranking when
    their cash consequences are negative.
    """

    capital_scale = max(abs(initial_cash), 1.0)
    expected_return = (expected_ending_cash - initial_cash) / capital_scale
    downside_return = (downside_ending_cash - initial_cash) / capital_scale
    return (
        EXPECTED_CASH_WEIGHT * expected_return
        + DOWNSIDE_CASH_WEIGHT * downside_return
        - BANKRUPTCY_PENALTY * bankruptcy_probability
    )


def _is_support_frontier(candidate: CandidateSimulation) -> bool:
    """Whether an admitted experiment creates product-quality evidence support."""

    return candidate.action.name.startswith(
        (
            "executive_experiment_targeted_development_",
            "controlled_exploration_development_",
        )
    )


def _support_frontier_intensity(candidate: CandidateSimulation) -> float:
    """Comparable weekly support movement, not an assumed commercial payoff.

    Targeted work uses the same documented effectiveness prior as the trusted
    transition. This score is used only after epistemic admission and a cash
    downside cap; it cannot make an ordinary operating plan look profitable.
    """

    targeted_weekly = 7.0 * sum(
        allocation.daily_spend
        for allocation in candidate.action.targeted_development_allocations
    )
    return candidate.action.development_spend + 5.0 * targeted_weekly


def _affordable_support_frontier(
    candidates: tuple[CandidateSimulation, ...],
    *,
    reference_downside_cash: float,
    downside_budget: float,
) -> tuple[CandidateSimulation, ...]:
    return tuple(
        candidate
        for candidate in candidates
        if max(0.0, reference_downside_cash - candidate.downside_ending_cash)
        <= downside_budget
    )


def summarize_candidate(
    initial_state: SimulationState,
    action: SimulationAction,
    outcomes: tuple[RolloutOutcome, ...],
) -> CandidateSimulation:
    if not outcomes:
        raise ValueError("candidate summary requires at least one rollout")

    ending_cash = [outcome.ending_cash for outcome in outcomes]
    customer_growth = [
        outcome.ending_customers - initial_state.customers for outcome in outcomes
    ]
    expected_cash = fmean(ending_cash)
    downside_cash = _lower_quantile(ending_cash, 0.10)
    upside_cash = _lower_quantile(ending_cash, 0.90)
    bankruptcy_probability = sum(outcome.bankrupt for outcome in outcomes) / len(outcomes)
    going_concern_failure_probability = sum(
        outcome.ending_customers < 1.0
        or outcome.states[-1].revenue_weekly <= 0.0
        for outcome in outcomes
    ) / len(outcomes)
    expected_growth = fmean(customer_growth)

    robust_utility = cash_robust_utility(
        initial_cash=initial_state.cash,
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
        upside_ending_cash=upside_cash,
        bankruptcy_probability=bankruptcy_probability,
        going_concern_failure_probability=going_concern_failure_probability,
        expected_customer_growth=expected_growth,
        robustness=robustness,
        robust_utility=robust_utility,
        rollout_count=len(outcomes),
    )


def select_robust_strategy(
    candidates: tuple[CandidateSimulation, ...],
    *,
    prefer_bounded_exploration: bool = False,
    inherited_going_concern_failure: bool = False,
) -> StrategySearchResult:
    if not candidates:
        raise ValueError("robust strategy selection requires at least one candidate")
    survival_eligible = tuple(
        candidate
        for candidate in candidates
        if candidate.bankruptcy_probability
        <= MAX_ACCEPTABLE_BANKRUPTCY_PROBABILITY
    )
    eligible = tuple(
        candidate
        for candidate in survival_eligible
        if candidate.going_concern_failure_probability
        <= MAX_ACCEPTABLE_GOING_CONCERN_FAILURE_PROBABILITY
    )
    if not eligible and all(
        candidate.going_concern_failure_probability >= 1.0 for candidate in candidates
    ) and not inherited_going_concern_failure:
        raise NoViableStrategyError(candidates)
    if eligible:
        utility_selected = max(
            eligible,
            key=lambda candidate: (
                candidate.robust_utility,
                candidate.downside_ending_cash,
                candidate.expected_ending_cash,
                candidate.action.name,
            ),
        )
        selected = utility_selected
        reason_code = "survival_gated_robust_utility"
        reason = (
            f"Selected {selected.action.name} among {len(eligible)}/{len(candidates)} "
            "survival-and-going-concern-eligible strategies: bankruptcy probability "
            f"{selected.bankruptcy_probability:.3f} is at or below "
            f"{MAX_ACCEPTABLE_BANKRUPTCY_PROBABILITY:.3f}; going-concern failure probability "
            f"{selected.going_concern_failure_probability:.3f}; highest downside-adjusted "
            f"utility {selected.robust_utility:.6f}, p10 cash "
            f"{selected.downside_ending_cash:.2f}."
        )
        if prefer_bounded_exploration:
            experiments = tuple(
                candidate
                for candidate in survival_eligible
                if candidate.action.name.startswith("controlled_exploration_")
                or candidate.action.name.startswith("executive_experiment_")
            )
            standard = tuple(candidate for candidate in eligible if candidate not in experiments)
            if experiments and standard:
                best_standard = max(
                    standard,
                    key=lambda candidate: (
                        candidate.robust_utility,
                        candidate.downside_ending_cash,
                        candidate.expected_ending_cash,
                        candidate.action.name,
                    ),
                )
                support_frontier = tuple(
                    candidate
                    for candidate in experiments
                    if _is_support_frontier(candidate)
                )
                downside_budget = min(
                    MAX_CONTROLLED_EXPLORATION_DOWNSIDE_ABSOLUTE,
                    max(0.0, best_standard.downside_ending_cash)
                    * MAX_CONTROLLED_EXPLORATION_DOWNSIDE_FRACTION,
                )
                experiment = max(
                    support_frontier or experiments,
                    key=lambda candidate: (
                        candidate.downside_ending_cash,
                        candidate.robust_utility,
                        candidate.action.name,
                    ),
                )
                experiment_end = max(
                    value
                    for value in (
                        experiment.action.marketing_spend_until_week,
                        experiment.action.development_spend_until_week,
                        experiment.action.targeted_development_spend_until_week,
                        experiment.action.lead_promotion_until_week,
                        0,
                    )
                    if value is not None
                )
                delayed_weeks = max(0, experiment_end - 1)
                if (
                    experiment.action.development_spend_until_week is not None
                    and delayed_weeks > 1
                ):
                    downside_budget = min(
                        MAX_DELAYED_EXPERIMENT_DOWNSIDE_ABSOLUTE,
                        max(0.0, best_standard.downside_ending_cash)
                        * MAX_DELAYED_EXPERIMENT_DOWNSIDE_FRACTION,
                    )
                if support_frontier:
                    downside_budget = min(
                        MAX_SUPPORT_FRONTIER_DOWNSIDE_ABSOLUTE,
                        max(0.0, best_standard.downside_ending_cash)
                        * MAX_SUPPORT_FRONTIER_DOWNSIDE_FRACTION,
                    )
                    affordable = _affordable_support_frontier(
                        support_frontier,
                        reference_downside_cash=best_standard.downside_ending_cash,
                        downside_budget=downside_budget,
                    )
                    if affordable:
                        experiment = max(
                            affordable,
                            key=lambda candidate: (
                                _support_frontier_intensity(candidate),
                                candidate.robust_utility,
                                candidate.downside_ending_cash,
                                candidate.action.name,
                            ),
                        )
                downside_cost = max(
                    0.0,
                    best_standard.downside_ending_cash
                    - experiment.downside_ending_cash,
                )
                decision_sensitive = experiment.action.name.startswith(
                    "controlled_exploration_"
                ) or bool(support_frontier) or (
                    min(
                        best_standard.expected_customer_growth,
                        experiment.expected_customer_growth,
                    )
                    < EXPLORATION_CONTINUITY_CUSTOMERS
                    <= max(
                        best_standard.expected_customer_growth,
                        experiment.expected_customer_growth,
                    )
                )
                if decision_sensitive and downside_cost <= downside_budget:
                    selected = experiment
                    reason_code = "decision_sensitive_bounded_exploration"
                    reason = (
                        f"Selected {selected.action.name} as a bounded controlled "
                        "experiment after the survival and policy gates: modeled p10 "
                        f"cash cost {downside_cost:.2f} is within the explicit information "
                        f"budget {downside_budget:.2f} versus {best_standard.action.name}. "
                        "The admission gate established that unresolved uncertainty can "
                        "change the continuity decision. Terminal going-concern failure "
                        f"probability {selected.going_concern_failure_probability:.3f} is "
                        "recorded but is not treated as a steady-policy forecast for this "
                        "expiring action; this selection does not assume a successful "
                        "experiment outcome."
                    )
    elif survival_eligible:
        selected = min(
            survival_eligible,
            key=lambda candidate: (
                candidate.going_concern_failure_probability,
                -candidate.downside_ending_cash,
                -candidate.robust_utility,
                candidate.action.name,
            ),
        )
        support_frontier = tuple(
            candidate
            for candidate in survival_eligible
            if _is_support_frontier(candidate)
        )
        if inherited_going_concern_failure and prefer_bounded_exploration and support_frontier:
            downside_budget = min(
                MAX_SUPPORT_FRONTIER_DOWNSIDE_ABSOLUTE,
                max(0.0, selected.downside_ending_cash)
                * MAX_SUPPORT_FRONTIER_DOWNSIDE_FRACTION,
            )
            affordable = _affordable_support_frontier(
                support_frontier,
                reference_downside_cash=selected.downside_ending_cash,
                downside_budget=downside_budget,
            )
            if affordable:
                experiment = max(
                    affordable,
                    key=lambda candidate: (
                        _support_frontier_intensity(candidate),
                        candidate.robust_utility,
                        candidate.downside_ending_cash,
                        candidate.action.name,
                    ),
                )
                downside_cost = max(
                    0.0,
                    selected.downside_ending_cash - experiment.downside_ending_cash,
                )
                selected = experiment
                reason_code = "inherited_continuity_support_frontier_experiment"
                reason = (
                    "The company entered planning with no customers or revenue, and all "
                    "survival-eligible candidates inherit that failure. Selected the "
                    f"bounded support-frontier program {selected.action.name}: its p10 cash "
                    f"cost {downside_cost:.2f} is within budget {downside_budget:.2f}. "
                    "Among affordable admitted programs it creates the strongest product-"
                    "support displacement. This is an information objective, not an assumed "
                    "conversion or commercial payoff."
                )
                return StrategySearchResult(
                    candidates=candidates,
                    selected=selected,
                    selection_reason_code=reason_code,
                    selection_reason=reason,
                )
        controlled_probes = tuple(
            candidate
            for candidate in survival_eligible
            if candidate.action.name.startswith("controlled_exploration_")
            and not _is_support_frontier(candidate)
        )
        if (
            inherited_going_concern_failure
            and prefer_bounded_exploration
            and controlled_probes
        ):
            downside_budget = min(
                MAX_CONTROLLED_EXPLORATION_DOWNSIDE_ABSOLUTE,
                max(0.0, selected.downside_ending_cash)
                * MAX_CONTROLLED_EXPLORATION_DOWNSIDE_FRACTION,
            )
            affordable = _affordable_support_frontier(
                controlled_probes,
                reference_downside_cash=selected.downside_ending_cash,
                downside_budget=downside_budget,
            )
            if affordable:
                experiment = max(
                    affordable,
                    key=lambda candidate: (
                        candidate.robust_utility,
                        candidate.downside_ending_cash,
                        candidate.expected_ending_cash,
                        candidate.action.name,
                    ),
                )
                downside_cost = max(
                    0.0,
                    selected.downside_ending_cash - experiment.downside_ending_cash,
                )
                selected = experiment
                reason_code = "inherited_continuity_bounded_exploration"
                reason = (
                    "The company entered planning without customers or revenue, but the "
                    "epistemic admission gate identified a bounded probe in the current "
                    f"support regime. Selected {selected.action.name}: modeled p10 cash "
                    f"cost {downside_cost:.2f} is within the measurement budget "
                    f"{downside_budget:.2f}. This purchases an outcome observation and "
                    "does not assume that the probe converts."
                )
                return StrategySearchResult(
                    candidates=candidates,
                    selected=selected,
                    selection_reason_code=reason_code,
                    selection_reason=reason,
                )
        if inherited_going_concern_failure:
            reason_code = "inherited_going_concern_minimum_failure"
            reason = (
                f"Inherited going-concern fallback selected {selected.action.name}: the "
                "company already has no customers or revenue, so terminal continuity risk "
                "is not a new violation caused by this plan. No survival-eligible strategy "
                "restores continuity within the modeled horizon; chose minimum failure "
                f"probability {selected.going_concern_failure_probability:.3f}, then "
                f"strongest p10 cash {selected.downside_ending_cash:.2f}. The degraded "
                "operating state remains explicit and is not reported as recovery."
            )
        else:
            reason_code = "going_concern_gate_emergency_minimum_failure"
            reason = (
                f"Going-concern fallback selected {selected.action.name}: all "
                f"{len(survival_eligible)} survival-eligible strategies exceed the maximum "
                "acceptable probability of ending without customers or revenue; chose "
                "minimum failure probability "
                f"{selected.going_concern_failure_probability:.3f}, then strongest p10 cash "
                f"{selected.downside_ending_cash:.2f}."
            )
    else:
        selected = min(
            candidates,
            key=lambda candidate: (
                candidate.bankruptcy_probability,
                candidate.going_concern_failure_probability,
                -candidate.downside_ending_cash,
                -candidate.robust_utility,
                candidate.action.name,
            ),
        )
        reason_code = "survival_gate_emergency_minimum_risk"
        reason = (
            f"Emergency fallback selected {selected.action.name}: all {len(candidates)} "
            "strategies exceed the maximum acceptable bankruptcy probability "
            f"{MAX_ACCEPTABLE_BANKRUPTCY_PROBABILITY:.3f}; chose minimum risk "
            f"{selected.bankruptcy_probability:.3f}, then strongest p10 cash "
            f"{selected.downside_ending_cash:.2f}."
        )
    return StrategySearchResult(
        candidates=candidates,
        selected=selected,
        selection_reason_code=reason_code,
        selection_reason=reason,
    )


def search_strategies(
    *,
    state: SimulationState,
    world_model: WorldModelVersion,
    actions: tuple[SimulationAction, ...] | None = None,
    horizon_weeks: int = 12,
    n_rollouts: int = 1_000,
    seed: int = 0,
    prefer_bounded_exploration: bool = False,
) -> StrategySearchResult:
    candidate_actions = actions or generate_candidate_actions(state)
    valid_actions = tuple(
        action
        for action in candidate_actions
        if evaluate_simulation_action(state, action).valid
    )
    if not valid_actions:
        raise ValueError(
            "robust strategy selection requires at least one economically valid candidate"
        )
    evaluations = tuple(
        summarize_candidate(
            state,
            action,
            simulate(
                state=state,
                action=action,
                world_model=world_model,
                horizon_weeks=horizon_weeks,
                n_rollouts=n_rollouts,
                seed=seed,
            ),
        )
        for action in valid_actions
    )
    return select_robust_strategy(
        evaluations,
        prefer_bounded_exploration=prefer_bounded_exploration,
        inherited_going_concern_failure=(
            state.customers < 1.0 or state.revenue_weekly <= 0.0
        ),
    )
