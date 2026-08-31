from __future__ import annotations

import re
from typing import Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from lithops.domain.models import (
    ActionCommand,
    ActionPlan,
    CashForecast,
    CashForecasts,
)


class StrictAgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SpendAllocation(StrictAgentOutput):
    operations: float = Field(ge=0, le=10_000)
    development: float = Field(ge=0, le=10_000)


class CashForecastOutput(StrictAgentOutput):
    # Google GenAI's Schema enum accepts strings only. Keep the exact values at
    # the provider boundary and convert them to domain integers after validation.
    horizon_days: Literal["7", "28", "84", "182"]
    point: float = Field(ge=-1_000_000_000, le=1_000_000_000)
    lower: float = Field(ge=-1_000_000_000, le=1_000_000_000)
    upper: float = Field(ge=-1_000_000_000, le=1_000_000_000)

    @model_validator(mode="after")
    def validate_interval(self) -> CashForecastOutput:
        if not self.lower <= self.point <= self.upper:
            raise ValueError("cash forecast must satisfy lower <= point <= upper")
        return self


class ExecutiveDecisionOutput(StrictAgentOutput):
    """Narrow P0 decision surface exposed to an LLM.

    The model selects only a bounded spend allocation. Lithops supplies the tool
    identity and idempotency key so provider output cannot bypass those controls.
    """

    name: str = Field(min_length=1, max_length=120)
    strategy_family: Literal[
        "aggressive_growth",
        "balanced_growth",
        "cash_preservation",
        "controlled_exploration",
    ]
    rationale: str = Field(min_length=1, max_length=4_000)
    daily_spend: SpendAllocation
    cash_forecasts: list[CashForecastOutput] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def validate_horizons(self) -> ExecutiveDecisionOutput:
        horizons = {item.horizon_days for item in self.cash_forecasts}
        if horizons != {"7", "28", "84", "182"}:
            raise ValueError("cash forecasts must contain exactly 7, 28, 84, and 182 days")
        return self

    def to_domain(self, *, run_id: UUID, week: int) -> tuple[ActionPlan, CashForecasts]:
        plan = ActionPlan(
            name=self.name,
            strategy_family=self.strategy_family,
            rationale=self.rationale,
            commands=[
                ActionCommand(
                    tool="set_daily_spend",
                    arguments=self.daily_spend.model_dump(mode="json"),
                    idempotency_key=f"{run_id}:week-{week}:executive-spend-0",
                )
            ],
        )
        forecasts = CashForecasts(
            items=[
                CashForecast(
                    horizon_days=int(item.horizon_days),
                    point=item.point,
                    lower=item.lower,
                    upper=item.upper,
                )
                for item in self.cash_forecasts
            ]
        )
        return plan, forecasts


class ExecutiveActionProposalOutput(StrictAgentOutput):
    """One semantic operating proposal; it contains no CEO-Bench tool names."""

    name: str = Field(min_length=1, max_length=120)
    hypothesis_id: str = Field(
        min_length=3,
        max_length=40,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    proposal_kind: Literal["operating", "experiment"]
    experiment_control: Literal[
        "none",
        "price",
        "tier",
        "quota",
        "promotion",
        "ads_strength",
        "marketing",
        "development",
        "targeted_development",
        "lead_promotion",
    ]
    experiment_duration_weeks: int = Field(default=1, ge=1, le=8)
    strategy_family: Literal[
        "growth",
        "efficiency",
        "pricing",
        "product_quality",
        "retention",
        "continuation",
    ]
    hypothesis: str = Field(min_length=1, max_length=1_000)
    expected_observation: str = Field(min_length=1, max_length=1_000)
    # What would make you stop. Required when an operating proposal is held for
    # more than one week, so a standing commitment can always be ended on
    # evidence rather than only on expiry.
    stop_condition: str = Field(default="", max_length=1_000)
    # Pre-registered readout for the commitment's maturity week: what to read,
    # the threshold that splits it, and the action either side selects. Your
    # future self inherits this rule with the result; leave it empty only when
    # the plan commits nothing beyond the current week.
    decision_rule: str = Field(default="", max_length=500)
    rationale: str = Field(min_length=1, max_length=2_000)
    catalog_price_multiplier: float = Field(ge=0.75, le=1.25)
    weekly_marketing_spend: float = Field(ge=0, le=70_000)
    daily_spend: SpendAllocation
    targeted_development_daily: float = Field(default=0.0, ge=0, le=10_000)
    model_tier_a: int = Field(ge=1, le=5)
    model_tier_b: int = Field(ge=1, le=5)
    model_tier_c: int = Field(ge=1, le=5)
    # Daily service allowance per customer on each plan. An allowance below what a
    # customer demands rations the value delivered to them.
    usage_quota_a: int = Field(default=0, ge=0, le=100_000)
    usage_quota_b: int = Field(default=0, ge=0, le=100_000)
    usage_quota_c: int = Field(default=0, ge=0, le=100_000)
    capacity_tier: int = Field(default=0, ge=0, le=7)
    lead_promotion_fraction: float = Field(ge=0, le=0.25)
    # Recurring discount off the listed entry price, unlike the first-invoice one.
    recurring_promotion_fraction: float = Field(default=0.0, ge=0, le=0.25)
    # In-product advertising: revenue per customer against perceived quality.
    ads_strength: float = Field(default=0.0, ge=0, le=1.0)
    # Support and reliability work directed at existing customers.
    targeted_ops_daily: float = Field(default=0.0, ge=0, le=10_000)
    # One owned-channel post per week at most.
    post_social_media: bool = False
    social_media_content: str = Field(default="", max_length=280)
    # The envelope you authorize for negotiating open enterprise threads. Offers
    # are made per thread after this plan executes, never outside this envelope.
    enterprise_engage: bool = False
    enterprise_target_price_per_seat: float = Field(default=0.0, ge=0, le=100_000)
    enterprise_floor_price_per_seat: float = Field(default=0.0, ge=0, le=100_000)
    enterprise_max_new_seats: int = Field(default=0, ge=0, le=100_000)
    # Start a research programme this week; its tier is its size. Irreversible
    # spend with a delayed, uncertain quality return. Zero starts nothing.
    research_project_tier: int = Field(default=0, ge=0, le=20)
    # The most this probe may cost before it is stopped. Zero leaves it to the
    # affordability ceiling; anything larger than that ceiling is capped by it.
    maximum_downside_budget: float = Field(default=0.0, ge=0, le=1_000_000)
    target_channel: Literal[
        "social_media",
        "search_ads",
        "linkedin",
        "content_marketing",
        "referral_program",
    ]
    target_segment: str = Field(
        min_length=2,
        max_length=8,
        pattern=r"^(?:S[1-3]|E[1-3]|D_[SE]\d{2})$",
    )

    @model_validator(mode="after")
    def validate_experiment_control(self) -> ExecutiveActionProposalOutput:
        if self.proposal_kind == "operating" and self.experiment_control != "none":
            raise ValueError("operating proposal cannot declare an experiment control")
        if (
            self.proposal_kind == "operating"
            and self.experiment_duration_weeks > 1
            and not self.stop_condition.strip()
        ):
            raise ValueError(
                "veto:standing_commitment_needs_a_stop_condition — holding a "
                "direction for several weeks requires naming the observation "
                "that would end it; durability without one is immunity"
            )
        if self.proposal_kind == "experiment" and self.experiment_control == "none":
            raise ValueError("experiment proposal requires one controlled intervention")
        if self.enterprise_engage:
            if self.enterprise_target_price_per_seat <= 0.0:
                raise ValueError("engaging enterprise threads requires a target price")
            if self.enterprise_floor_price_per_seat > (
                self.enterprise_target_price_per_seat
            ):
                raise ValueError("the enterprise floor cannot exceed the target price")
        # These mirror domain invariants that would otherwise only surface while the
        # plan is being constructed. Refusing here names the gap in the same
        # vocabulary the evaluation cards use and lets the provider's own retry carry
        # the reason back, so the proposal is remade by its author rather than
        # repaired by us.
        if self.research_project_tier and self.proposal_kind == "experiment":
            raise ValueError(
                "veto:research_programme_is_not_a_probe — a research programme is "
                "irreversible spend with a delayed, uncertain return, so it is "
                "governed as a portfolio commitment rather than a one-week "
                "controlled experiment; propose it on an operating candidate"
            )
        if (
            self.experiment_control == "targeted_development"
            and self.weekly_marketing_spend <= 0.0
        ):
            raise ValueError(
                "veto:acquisition_probe_missing — the acquisition probe of a "
                "targeted_development experiment is funded by weekly_marketing_spend; "
                "a zero value buys no probe exposure, so this candidate is refused"
            )
        if self.proposal_kind == "experiment":
            identity = (
                f"executive_experiment_{self.experiment_control}_{self.hypothesis_id}"
            )
            if len(identity) > 80:
                raise ValueError(
                    "veto:strategy_identity_too_long — an experiment is identified by "
                    f"executive_experiment_<control>_<hypothesis_id>, capped at 80 "
                    f"characters; this identity needs {len(identity)}"
                )
        return self


class ExecutiveProposalOutput(StrictAgentOutput):
    """A bounded set of competing hypotheses for deterministic evaluation."""

    decision_summary: str = Field(min_length=1, max_length=2_000)
    candidates: list[ExecutiveActionProposalOutput] = Field(min_length=2, max_length=4)

    @model_validator(mode="after")
    def validate_candidate_identity(self) -> ExecutiveProposalOutput:
        names = [candidate.name for candidate in self.candidates]
        if len(names) != len(set(names)):
            raise ValueError("executive proposal candidate names must be unique")
        return self


class ExperimentInterpretationOutput(StrictAgentOutput):
    """The Executive's reading of one newly matured experiment outcome."""

    commitment_id: str = Field(min_length=3, max_length=80)
    hypothesis_id: str = Field(min_length=1, max_length=60, pattern=r"^[a-z][a-z0-9_]*$")
    interpretation: Literal["supports", "falsifies", "inconclusive", "invalid"]
    reasoning: str = Field(min_length=1, max_length=1_000)


class HypothesisStatusUpdateOutput(StrictAgentOutput):
    hypothesis_id: str = Field(min_length=1, max_length=60, pattern=r"^[a-z][a-z0-9_]*$")
    new_status: Literal[
        "running", "supported", "falsified", "inconclusive", "superseded", "abandoned"
    ]
    reason: str = Field(min_length=1, max_length=1_000)
    successor_hypothesis_ids: list[str] = Field(default_factory=list, max_length=4)


class HypothesisProposalOutput(StrictAgentOutput):
    """One new causal hypothesis added to the strategic portfolio."""

    hypothesis_id: str = Field(min_length=3, max_length=60, pattern=r"^[a-z][a-z0-9_]*$")
    causal_claim: str = Field(min_length=1, max_length=2_000)
    target_outcome: str = Field(min_length=1, max_length=1_000)
    levers: list[
        Literal[
            "price",
            "tier",
            "promotion",
            "development",
            "targeted_development",
            "acquisition",
            "capacity_cost",
            "continuation",
        ]
    ] = Field(min_length=1, max_length=4)
    segment: str = Field(default="", max_length=8)
    # Keep the provider schema free of an empty enum member: Gemini rejects
    # response schemas whose enum contains "".  Empty still means that the
    # hypothesis is not channel-specific and is validated below.
    channel: str = Field(default="", max_length=32)
    predecessor_hypothesis_id: str = Field(default="", max_length=60)
    material_difference: str = Field(default="", max_length=1_000)
    competing_predictions: str = Field(min_length=1, max_length=1_000)
    decisive_observation: str = Field(min_length=1, max_length=1_000)

    @model_validator(mode="after")
    def validate_lineage(self) -> HypothesisProposalOutput:
        if self.segment and not re.fullmatch(r"(?:S[1-3]|E[1-3]|D_[SE]\d{2})", self.segment):
            raise ValueError("hypothesis segment must be an observed segment code")
        if self.channel and self.channel not in {
            "social_media",
            "search_ads",
            "linkedin",
            "content_marketing",
            "referral_program",
        }:
            raise ValueError("hypothesis channel must be an observed channel code")
        if len(set(self.levers)) != len(self.levers):
            raise ValueError("hypothesis levers must be unique")
        if self.predecessor_hypothesis_id and not self.material_difference:
            raise ValueError(
                "a successor hypothesis must state why it is materially different"
            )
        return self


class InformationRequestOutput(StrictAgentOutput):
    """One purchase of market information, justified by the unknown it resolves."""

    tool: Literal[
        "research_market",
        "research_group",
        "get_group_insights",
        "get_market_overview",
        "get_cost_info",
    ]
    target_group: str = Field(default="", max_length=8)
    target_level: int = Field(default=0, ge=0, le=5)
    expected_information_value: str = Field(min_length=1, max_length=1_000)

    @model_validator(mode="after")
    def validate_target(self) -> InformationRequestOutput:
        if self.target_group and not re.fullmatch(
            r"(?:S[1-3]|E[1-3]|D_[SE]\d{2})", self.target_group
        ):
            raise ValueError("information target must be an observed segment code")
        if self.tool in {"research_group", "get_group_insights"} and not self.target_group:
            raise ValueError(f"{self.tool} requires a target group")
        if self.tool == "research_group" and self.target_level < 2:
            raise ValueError("research_group requires a target level of 2 or higher")
        return self


class ExecutiveChoiceOutput(StrictAgentOutput):
    """Second-stage Executive output: one eligible candidate ID, with the case.

    The model chooses only among the immutable evaluation cards it was shown;
    deterministic Python validates the ID and plan hash before execution.
    """

    selected_candidate_id: str = Field(min_length=1, max_length=120)
    decision_thesis: str = Field(min_length=1, max_length=2_000)
    evidence_refs: list[str] = Field(default_factory=list, max_length=6)
    rejected_candidate_ids: list[str] = Field(default_factory=list, max_length=8)
    stop_or_pivot_condition: str = Field(min_length=1, max_length=1_000)
    # Information purchases are decided per week, not per candidate arm: they
    # change no configuration and so do not multiply the candidate set.
    information_requests: list[InformationRequestOutput] = Field(
        default_factory=list, max_length=2
    )

    @field_validator("information_requests", mode="before")
    @classmethod
    def tolerate_malformed_information_requests(cls, value: object) -> object:
        """A malformed side-request must not sink the selected candidate.

        The choice is the payload; an information request is an attachment that
        the admission gate re-judges anyway (budget, duplicates). One run lost
        whole weeks to safe-continuation because a single request named a
        segment as ``D_S1`` instead of ``D_S01`` and the entire choice failed
        validation three times. Light normalization first, then any entry that
        still cannot parse is dropped rather than fatal.
        """

        if not isinstance(value, list):
            return value
        admitted: list[object] = []
        for entry in value:
            if isinstance(entry, dict):
                target = entry.get("target_group")
                if isinstance(target, str):
                    normalized = target.strip().upper()
                    normalized = re.sub(
                        r"^(D_[SE])(\d)$", lambda m: f"{m.group(1)}0{m.group(2)}", normalized
                    )
                    entry = {**entry, "target_group": normalized}
            try:
                admitted.append(InformationRequestOutput.model_validate(entry))
            except ValidationError:
                continue
        return admitted
    # Read-only questions about the business database. The answers come back in
    # next week's brief, so the Executive can look at what it wants to look at
    # rather than only at what the observation contract happens to expose.
    data_queries: list[str] = Field(default_factory=list, max_length=3)
    # Carried verbatim into the next week. The portfolio holds hypotheses; this
    # is where intent that spans weeks survives between calls.
    journal: str = Field(default="", max_length=2_000)

    @model_validator(mode="after")
    def validate_queries(self) -> ExecutiveChoiceOutput:
        for query in self.data_queries:
            if len(query) > 2_000:
                raise ValueError("a data query must be under 2000 characters")
        return self

    @model_validator(mode="after")
    def validate_selection(self) -> ExecutiveChoiceOutput:
        if self.selected_candidate_id in self.rejected_candidate_ids:
            raise ValueError("the selected candidate cannot also be rejected")
        return self


class StrategyPortfolioUpdateOutput(StrictAgentOutput):
    """First-stage Executive output: an update to the causal portfolio.

    The Executive interprets evidence and revises hypotheses; deterministic
    Python applies this diff under append-only governance and computes hashes.
    """

    portfolio_thesis: str = Field(min_length=1, max_length=2_000)
    binding_constraint: str = Field(min_length=1, max_length=1_000)
    outcome_interpretations: list[ExperimentInterpretationOutput] = Field(
        default_factory=list, max_length=8
    )
    status_updates: list[HypothesisStatusUpdateOutput] = Field(
        default_factory=list, max_length=8
    )
    new_hypotheses: list[HypothesisProposalOutput] = Field(
        default_factory=list, max_length=4
    )
    active_hypothesis_ids: list[str] = Field(default_factory=list, max_length=6)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=6)

    @model_validator(mode="after")
    def validate_identity(self) -> StrategyPortfolioUpdateOutput:
        new_ids = [hypothesis.hypothesis_id for hypothesis in self.new_hypotheses]
        if len(new_ids) != len(set(new_ids)):
            raise ValueError("new hypothesis IDs must be unique")
        update_ids = [update.hypothesis_id for update in self.status_updates]
        if len(update_ids) != len(set(update_ids)):
            raise ValueError("at most one status update per hypothesis")
        return self
