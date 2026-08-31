from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import ValidationError

from lithops.agents.common import (
    ExecutiveActionProposalOutput,
    ExecutiveChoiceOutput,
    ExecutiveDecisionOutput,
    ExecutiveProposalOutput,
    StrategyPortfolioUpdateOutput,
)
from lithops.domain.models import (
    ActionCommand,
    ActionPlan,
    CashForecasts,
    DecisionRecord,
    ExperimentMeasurement,
    ExperimentProgram,
    ObservationSnapshot,
    ProposalBatch,
    ProposalRejection,
    RunRecord,
    construction_veto_codes,
)
from lithops.domain.ports import StructuredModelProvider
from lithops.evaluation.trajectory import weekly_trajectory

PROMPT_PATH = Path(__file__).with_name("prompt.txt")
PROPOSAL_PROMPT_PATH = Path(__file__).with_name("proposal_prompt.txt")
STRATEGY_ARCHITECT_PROMPT_PATH = Path(__file__).with_name("strategy_architect_prompt.txt")
CANDIDATE_SELECTION_PROMPT_PATH = Path(__file__).with_name(
    "candidate_selection_prompt.txt"
)
INITIAL_MONTHLY_PRICES = (25.0, 69.0, 179.0)


class ExecutiveDecisionEngine:
    """Turns observable company state into a bounded, schema-validated decision."""

    prompt_version = "executive-v8"
    strategy_architect_prompt_version = "strategy-architect-v4"
    candidate_selection_prompt_version = "candidate-selection-v5"

    def __init__(self, provider: StructuredModelProvider) -> None:
        self.provider = provider
        self.system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
        self.proposal_system_prompt = PROPOSAL_PROMPT_PATH.read_text(encoding="utf-8")
        self.strategy_architect_prompt = STRATEGY_ARCHITECT_PROMPT_PATH.read_text(
            encoding="utf-8"
        )
        self.candidate_selection_prompt = CANDIDATE_SELECTION_PROMPT_PATH.read_text(
            encoding="utf-8"
        )

    async def select_candidate(
        self,
        *,
        brief: dict,
    ) -> ExecutiveChoiceOutput:
        """Second Executive stage: pick one eligible candidate ID.

        Python validates the returned ID and plan hash against the immutable
        evaluation set before anything is persisted or executed.
        """

        return await self.provider.generate_structured(
            system_prompt=self.candidate_selection_prompt,
            user_prompt=json.dumps(brief, separators=(",", ":"), sort_keys=True),
            output_schema=ExecutiveChoiceOutput,
        )

    async def update_strategy_portfolio(
        self,
        *,
        brief: dict,
    ) -> StrategyPortfolioUpdateOutput:
        """First Executive stage: revise the causal portfolio from evidence.

        The brief is a compact deterministic reduction of current state — never
        an ever-growing conversational transcript. Governance and hashing of the
        resulting revision stay in deterministic Python.
        """

        return await self.provider.generate_structured(
            system_prompt=self.strategy_architect_prompt,
            user_prompt=json.dumps(brief, separators=(",", ":"), sort_keys=True),
            output_schema=StrategyPortfolioUpdateOutput,
        )

    async def propose_actions(
        self,
        *,
        run: RunRecord,
        observation: ObservationSnapshot,
        decision_history: tuple[DecisionRecord, ...] = (),
        portfolio_context: dict | None = None,
        rejection_feedback: tuple[dict, ...] | None = None,
    ) -> ProposalBatch:
        """Propose competing semantic actions without privileged tool selection."""

        observation_by_day = {
            decision.observation.day: decision.observation
            for decision in decision_history
        }
        observation_by_day[observation.day] = observation
        matured_residuals = [
            {
                "issued_day": decision.observation.day,
                "horizon_days": forecast.horizon_days,
                "predicted_cash": forecast.point,
                "lower_cash": forecast.lower,
                "upper_cash": forecast.upper,
                "actual_cash": observation_by_day[
                    decision.observation.day + forecast.horizon_days
                ].cash,
                "signed_error": (
                    observation_by_day[
                        decision.observation.day + forecast.horizon_days
                    ].cash
                    - forecast.point
                ),
                "interval_hit": (
                    forecast.lower
                    <= observation_by_day[
                        decision.observation.day + forecast.horizon_days
                    ].cash
                    <= forecast.upper
                ),
            }
            for decision in decision_history
            for forecast in decision.forecasts.items
            if decision.observation.day + forecast.horizon_days in observation_by_day
        ][-24:]
        # One compact line per past decision. The full dumps this used to carry
        # — two complete observation snapshots per entry, each with its own
        # market feed and evidence packet — weighed ~400 KB a week, were never
        # referenced by the prompt, and drowned the curated series the prompt
        # does point at. Metric history lives in `weekly_trajectory`; this list
        # answers only "what was decided, why, and what did it cost".
        history = [
            {
                "week": decision.week,
                "strategy": decision.action_plan.strategy_family,
                "selection_reason_code": decision.selection_reason_code,
                "realized_cash_change": (
                    round(decision.actual_outcome.cash - decision.observation.cash, 2)
                    if decision.actual_outcome is not None
                    else None
                ),
                "experiment": (
                    {
                        "commitment_id": program.commitment_id,
                        "control": program.control,
                        "started_week": program.started_week,
                        "maximum_end_week": program.maximum_end_week,
                    }
                    if (program := decision.action_plan.experiment_program) is not None
                    else None
                ),
            }
            for decision in decision_history[-12:]
        ]
        input_payload = {
            "run_id": str(run.id),
            "week": observation.day // 7,
            "remaining_days": max(0, run.horizon_days - observation.day),
            "objective": (
                "risk-adjusted terminal cash subject to solvency and going concern"
            ),
            "observation": observation.model_dump(mode="json"),
            "unconfigured_controls": self._unconfigured_controls(observation.metrics),
            # The run's own history as a series, so trends and rates are visible
            # rather than having to be inferred from one snapshot.
            "weekly_trajectory": weekly_trajectory(decision_history),
            "recent_decisions": history,
            "matured_cash_residuals": matured_residuals,
            "active_model_lineage": (
                {
                    "artifact_id": str(decision_history[-1].model_artifact_id),
                    "artifact_hash": decision_history[-1].model_artifact_hash,
                    "fitted_model_id": str(decision_history[-1].fitted_model_id),
                }
                if decision_history
                and decision_history[-1].model_artifact_id is not None
                else None
            ),
            # What the run can actually execute. This list is the model's whole
            # sense of what it may consider, so a stale entry silently removes a
            # lever: `enterprise_deals` sat under "unavailable" long after the
            # negotiation loop shipped, hiding the benchmark's largest revenue
            # lever, and research programmes were never listed at all.
            "semantic_capabilities": {
                "supported": [
                    "catalog_price",
                    "weekly_marketing_spend",
                    "marketing_channel_and_segment",
                    "daily_operations_spend",
                    "daily_development_spend",
                    "targeted_development",
                    "targeted_operations_spend",
                    "model_tiers",
                    "usage_quotas",
                    "capacity_tier",
                    "first_invoice_lead_promotion",
                    "recurring_promotion",
                    "in_product_ads_strength",
                    "owned_channel_social_post",
                    "enterprise_deals",
                    "research_programmes",
                ],
                "unavailable_until_modeled": [],
                "experiment_semantics": {
                    "immediate_controls": [
                        "price",
                        "tier",
                        "quota",
                        "marketing",
                        "lead_promotion",
                    ],
                    "delayed_controls": ["development", "targeted_development"],
                    "development_duration_weeks": [2, 8],
                    "weekly_safety_review": True,
                    "silent_renewal": False,
                },
            },
        }
        if portfolio_context is not None:
            input_payload["strategic_portfolio"] = portfolio_context
        if rejection_feedback:
            input_payload["prior_rejections"] = list(rejection_feedback)
        output = await self.provider.generate_structured(
            system_prompt=self.proposal_system_prompt,
            user_prompt=json.dumps(input_payload, separators=(",", ":"), sort_keys=True),
            output_schema=ExecutiveProposalOutput,
        )
        week = observation.day // 7
        plans: list[ActionPlan] = []
        rejections: list[ProposalRejection] = []
        for index, proposal in enumerate(output.candidates):
            # One unusable proposal is one refused candidate, never a failed week.
            # The remaining proposals and the deterministic pool still stand.
            try:
                plans.append(
                    self._proposal_plan(
                        proposal,
                        run=run,
                        observation=observation,
                        candidate_index=index,
                    )
                )
            except (ValidationError, ValueError) as error:
                rejections.append(
                    ProposalRejection(
                        week=week,
                        candidate_index=index,
                        name=proposal.name,
                        hypothesis_id=proposal.hypothesis_id,
                        stage="construction",
                        veto_codes=construction_veto_codes(error),
                        detail=str(error)[:500],
                    )
                )
        return ProposalBatch(plans=tuple(plans), rejections=tuple(rejections))

    @staticmethod
    def _unconfigured_controls(
        metrics: dict[str, float | int | str | bool | None],
    ) -> list[dict[str, str]]:
        """Controls still sitting at the environment default, never set by you.

        Each entry states the observed value and the mechanical consequence of
        leaving it there. It names the gap; choosing the value is the Executive's
        decision.
        """

        def number(name: str) -> float:
            value = metrics.get(name)
            return float(value) if isinstance(value, int | float) else 0.0

        entries: list[dict[str, str]] = []
        if all(number(f"configured_price_{plan}") <= 0.0 for plan in "abc"):
            entries.append(
                {
                    "control": "catalog_prices",
                    "observed": "every plan is priced at zero",
                    "consequence": "a plan priced at zero bills nothing to its subscribers",
                }
            )
        if all(number(f"usage_quota_{plan}") <= 0.0 for plan in "abc"):
            entries.append(
                {
                    "control": "usage_quotas",
                    "observed": "every plan has a zero daily service allowance",
                    "consequence": (
                        "an allowance of zero serves none of a customer's demanded "
                        "usage, so the value delivered on that plan is zero"
                    ),
                }
            )
        if number("capacity_tier") <= 0.0:
            entries.append(
                {
                    "control": "capacity_tier",
                    "observed": "capacity tier 0",
                    "consequence": (
                        "served capacity stays at the entry level; usage above it "
                        "degrades service"
                    ),
                }
            )
        tiers = {plan.upper(): int(number(f"model_tier_{plan}") or 1) for plan in "abc"}
        if len(set(tiers.values())) > 1:
            lowest = min(tiers, key=tiers.get)
            entries.append(
                {
                    "control": "model_tiers",
                    "observed": (
                        "plans run different model tiers: "
                        + ", ".join(f"{plan}={tier}" for plan, tier in tiers.items())
                    ),
                    "consequence": (
                        f"delivered quality differs per plan; customers on plan "
                        f"{lowest} judge the lowest-multiplied number regardless "
                        "of how far the base has risen"
                    ),
                }
            )
        measured_demand = metrics.get("estimated_usage_demand_per_day")
        if isinstance(measured_demand, int | float) and measured_demand > 0:
            rationed_plans = [
                plan.upper()
                for plan in "abc"
                if 0.0 < number(f"usage_quota_{plan}") < float(measured_demand)
            ]
            if rationed_plans:
                entries.append(
                    {
                        "control": "usage_quotas",
                        "observed": (
                            f"plans {', '.join(rationed_plans)} allow less daily "
                            f"usage than the measured demand of "
                            f"{float(measured_demand):.0f} units/day"
                        ),
                        "consequence": (
                            "an allowance below what a customer demands serves "
                            "only part of their usage, so the value those plans "
                            "deliver is degraded"
                        ),
                    }
                )
        return entries

    @staticmethod
    def _proposal_plan(
        proposal: ExecutiveActionProposalOutput,
        *,
        run: RunRecord,
        observation: ObservationSnapshot,
        candidate_index: int,
    ) -> ActionPlan:
        week = observation.day // 7
        metrics = observation.metrics
        prices = tuple(
            float(metrics.get(name, default))
            if isinstance(metrics.get(name, default), int | float)
            and float(metrics.get(name, default)) > 0
            else default
            for name, default in zip(
                ("price_a", "price_b", "price_c"),
                INITIAL_MONTHLY_PRICES,
                strict=True,
            )
        )
        known_segments = {
            segment.strip()
            for segment in str(metrics.get("known_segments") or "S1").split(",")
            if re.fullmatch(r"(?:S[1-3]|E[1-3]|D_[SE]\d{2})", segment.strip())
        }
        # Silently retargeting a proposal would execute a decision nobody made.
        # An unobserved segment is carried through and refused downstream instead.
        target_segment = proposal.target_segment
        segment_is_observed = not known_segments or target_segment in known_segments
        is_experiment = proposal.proposal_kind == "experiment"
        lead_count = float(metrics.get("total_leads", 0.0) or 0.0)
        lead_band = (
            "none"
            if lead_count <= 0
            else "lt30"
            if lead_count < 30
            else "lt100"
            if lead_count < 100
            else "gte100"
        )
        quality = float(metrics.get("product_quality", 0.0) or 0.0)
        quality_band = min(4, max(0, int(quality * 5)))
        customer_band = (
            "active"
            if float(metrics.get("active_customers", 0.0) or 0.0) > 0
            else "zero"
        )
        evidence_regime = (
            f"leads_{lead_band}:quality_{quality_band}:customers_{customer_band}"
        )
        current_marketing = float(metrics.get("marketing_spend", 0.0) or 0.0)
        current_operations_daily = (
            float(metrics.get("operations_spend", 0.0) or 0.0) / 7.0
        )
        current_development_daily = (
            float(metrics.get("development_spend", 0.0) or 0.0) / 7.0
        )
        try:
            current_targeted_development = json.loads(
                str(metrics.get("targeted_development_allocations_json") or "{}")
            )
        except (TypeError, ValueError):
            current_targeted_development = {}
        if not isinstance(current_targeted_development, dict):
            current_targeted_development = {}
        if isinstance(current_targeted_development.get("targeted_spend"), dict):
            current_targeted_development = current_targeted_development["targeted_spend"]
        current_targeted_development = {
            str(segment): float(amount)
            for segment, amount in current_targeted_development.items()
            if isinstance(amount, int | float) and float(amount) >= 0.0
        }
        try:
            current_targeted_ads = json.loads(
                str(metrics.get("targeted_ad_allocations_json") or "{}")
            )
        except (TypeError, ValueError):
            current_targeted_ads = {}
        if not isinstance(current_targeted_ads, dict):
            current_targeted_ads = {}
        if isinstance(current_targeted_ads.get("targeted_spend"), dict):
            current_targeted_ads = current_targeted_ads["targeted_spend"]
        price_multiplier = (
            proposal.catalog_price_multiplier
            if not is_experiment or proposal.experiment_control == "price"
            else 1.0
        )
        # Offer-side experiments still need measurement exposure: with no leads
        # arriving there is nothing to observe, so the probe could never mature.
        # The proposed exposure is held identical in both arms, which keeps the
        # contrast a single difference in the declared control.
        carries_measurement_exposure = is_experiment and proposal.experiment_control in {
            "price",
            "tier",
            "quota",
            "lead_promotion",
        }
        weekly_marketing = (
            proposal.weekly_marketing_spend
            if not is_experiment
            or proposal.experiment_control == "marketing"
            or carries_measurement_exposure
            else current_marketing
        )
        operations_daily = (
            proposal.daily_spend.operations
            if not is_experiment
            else current_operations_daily
        )
        development_daily = (
            proposal.daily_spend.development
            if not is_experiment
            or proposal.experiment_control == "development"
            else current_development_daily
        )
        current_tiers = {
            "A": int(metrics.get("model_tier_a", 1) or 1),
            "B": int(metrics.get("model_tier_b", 1) or 1),
            "C": int(metrics.get("model_tier_c", 1) or 1),
        }
        proposed_tiers = {
            "A": proposal.model_tier_a,
            "B": proposal.model_tier_b,
            "C": proposal.model_tier_c,
        }
        observed_quotas = {
            plan: int(float(metrics.get(f"usage_quota_{plan.lower()}", 0) or 0))
            for plan in ("A", "B", "C")
        }
        proposed_quotas = {
            "A": proposal.usage_quota_a,
            "B": proposal.usage_quota_b,
            "C": proposal.usage_quota_c,
        }
        # An experiment holds every control it does not test at its operating
        # level. A control that was never configured has no operating level to
        # hold, so both arms adopt the proposed one: the arms still differ only
        # in the declared control.
        current_quotas = (
            proposed_quotas
            if all(value <= 0 for value in observed_quotas.values())
            else observed_quotas
        )
        current_capacity_tier = int(float(metrics.get("capacity_tier", 0) or 0))
        observed_recurring_promotion = float(
            metrics.get("recurring_promotion_monthly", 0.0) or 0.0
        )
        observed_ads_strength = float(metrics.get("ads_strength", 0.0) or 0.0)
        observed_targeted_ops_daily = (
            float(metrics.get("targeted_ops_spend", 0.0) or 0.0) / 7.0
        )
        entry_price = min(prices)
        recurring_promotion = entry_price * proposal.recurring_promotion_fraction
        commands = [
            ActionCommand(
                tool="set_prices",
                arguments={
                    tier: price * price_multiplier
                    for tier, price in zip(("A", "B", "C"), prices, strict=True)
                },
                idempotency_key=(
                    f"{run.id}:week-{week}:executive-{candidate_index}:prices"
                ),
            ),
            ActionCommand(
                tool="set_model_tiers",
                arguments=(
                    proposed_tiers
                    if not is_experiment or proposal.experiment_control == "tier"
                    else current_tiers
                ),
                idempotency_key=(
                    f"{run.id}:week-{week}:executive-{candidate_index}:tiers"
                ),
            ),
            ActionCommand(
                tool="set_daily_spend",
                arguments={
                    "operations": operations_daily,
                    "development": development_daily,
                },
                idempotency_key=(
                    f"{run.id}:week-{week}:executive-{candidate_index}:spend"
                ),
            ),
            ActionCommand(
                tool="set_targeted_ad_spend",
                arguments={"targeted_spend": (
                    {
                        proposal.target_channel: {
                            target_segment: weekly_marketing / 7.0
                        }
                    }
                    if not is_experiment
                    or proposal.experiment_control == "marketing"
                    or carries_measurement_exposure
                    else current_targeted_ads
                )},
                idempotency_key=(
                    f"{run.id}:week-{week}:executive-{candidate_index}:marketing"
                ),
            ),
            ActionCommand(
                tool="set_targeted_dev_spend",
                arguments={"targeted_spend": (
                    {target_segment: proposal.targeted_development_daily}
                    if not is_experiment
                    or proposal.experiment_control == "targeted_development"
                    else current_targeted_development
                )},
                idempotency_key=(
                    f"{run.id}:week-{week}:executive-{candidate_index}:targeted-development"
                ),
            ),
            ActionCommand(
                tool="set_usage_quotas",
                arguments=(
                    proposed_quotas
                    if not is_experiment or proposal.experiment_control == "quota"
                    else current_quotas
                ),
                idempotency_key=(
                    f"{run.id}:week-{week}:executive-{candidate_index}:usage-quotas"
                ),
            ),
            ActionCommand(
                tool="set_capacity_tier",
                arguments={
                    "tier": (
                        proposal.capacity_tier
                        if not is_experiment
                        else current_capacity_tier
                    )
                },
                idempotency_key=(
                    f"{run.id}:week-{week}:executive-{candidate_index}:capacity-tier"
                ),
            ),
        ]
        # An unchanged control needs no command: the benchmark carries the last
        # configured value forward, so restating it only adds a call that the
        # fidelity check would then have to reconcile.
        selected_recurring_promotion = (
            recurring_promotion
            if not is_experiment or proposal.experiment_control == "promotion"
            else observed_recurring_promotion
        )
        if abs(selected_recurring_promotion - observed_recurring_promotion) > 1e-9:
            commands.append(
                ActionCommand(
                    tool="set_promotion",
                    arguments={"global_promotion": selected_recurring_promotion},
                    idempotency_key=(
                        f"{run.id}:week-{week}:executive-{candidate_index}"
                        ":recurring-promotion"
                    ),
                )
            )
        selected_ads_strength = (
            proposal.ads_strength
            if not is_experiment or proposal.experiment_control == "ads_strength"
            else observed_ads_strength
        )
        if abs(selected_ads_strength - observed_ads_strength) > 1e-9:
            commands.append(
                ActionCommand(
                    tool="set_ads_strength",
                    arguments={"global_strength": selected_ads_strength},
                    idempotency_key=(
                        f"{run.id}:week-{week}:executive-{candidate_index}:ads-strength"
                    ),
                )
            )
        selected_targeted_ops_daily = (
            proposal.targeted_ops_daily
            if not is_experiment
            else observed_targeted_ops_daily
        )
        if abs(selected_targeted_ops_daily - observed_targeted_ops_daily) > 1e-9:
            commands.append(
                ActionCommand(
                    tool="set_targeted_ops_spend",
                    arguments={
                        "targeted_spend": {target_segment: selected_targeted_ops_daily}
                    },
                    idempotency_key=(
                        f"{run.id}:week-{week}:executive-{candidate_index}:targeted-ops"
                    ),
                )
            )
        if proposal.research_project_tier:
            commands.append(
                ActionCommand(
                    tool="start_research_project",
                    arguments={"tier": proposal.research_project_tier},
                    idempotency_key=(
                        f"{run.id}:week-{week}:executive-{candidate_index}:research"
                    ),
                )
            )
        if proposal.post_social_media and proposal.social_media_content.strip():
            commands.append(
                ActionCommand(
                    tool="post_social_media",
                    arguments={"content": proposal.social_media_content.strip()},
                    idempotency_key=(
                        f"{run.id}:week-{week}:executive-{candidate_index}:social-post"
                    ),
                )
            )
        observed_promotion = metrics.get("lead_promotion_monthly", 0.0)
        observed_promotion = (
            float(observed_promotion)
            if isinstance(observed_promotion, int | float)
            else 0.0
        )
        promotion = (
            entry_price * proposal.lead_promotion_fraction
            if not is_experiment
            or proposal.experiment_control == "lead_promotion"
            else observed_promotion
        )
        if abs(promotion - observed_promotion) > 1e-9:
            commands.append(
                ActionCommand(
                    tool="set_lead_promotion",
                    arguments={"global_promotion": promotion},
                    idempotency_key=(
                        f"{run.id}:week-{week}:executive-{candidate_index}:promotion"
                    ),
                )
            )
        experiment_program = None
        experiment_expires_week = None
        if not is_experiment and proposal.experiment_duration_weeks > 1:
            # A direction held for several weeks. It gets the same weekly review,
            # the same declared spending limit and the same audit trail as an
            # experiment; what it does not get is a control arm or an automatic
            # rollback, because holding a direction is the decision itself.
            duration_weeks = max(2, min(8, proposal.experiment_duration_weeks))
            experiment_expires_week = week + duration_weeks
            # A held direction is a larger, longer bet than a probe, so its
            # ceiling is the share of capital a company can commit to a strategy
            # rather than the smaller one that bounds a week of learning.
            budget_ceiling = min(500_000.0, max(30_000.0, observation.cash * 0.40))
            experiment_program = ExperimentProgram(
                commitment_id=f"{proposal.hypothesis_id}-hold-{week}",
                control="strategy",
                protocol_version="operating-commitment-v1",
                started_week=week,
                minimum_maturity_week=week + duration_weeks,
                maximum_end_week=experiment_expires_week,
                baseline_value=0.0,
                treatment_value=0.0,
                maximum_cumulative_downside=(
                    min(proposal.maximum_downside_budget, budget_ceiling)
                    if proposal.maximum_downside_budget > 0.0
                    else budget_ceiling
                ),
                expected_observation=proposal.expected_observation,
                falsification_condition=proposal.stop_condition,
                decision_rule=proposal.decision_rule,
                target_segment=target_segment if segment_is_observed else None,
                target_channel=proposal.target_channel,
            )
        if is_experiment:
            # The declared duration stands. Delayed controls still need at least
            # two weeks to show anything, which is a property of the control, not
            # a preference about the plan.
            duration_weeks = (
                max(2, min(8, proposal.experiment_duration_weeks))
                if proposal.experiment_control in {"development", "targeted_development"}
                else max(1, min(8, proposal.experiment_duration_weeks))
            )
            maturity_week = week + duration_weeks
            experiment_expires_week = (
                maturity_week + 1
                if proposal.experiment_control in {"development", "targeted_development"}
                else maturity_week
            )
            baseline_value, treatment_value = {
                "price": (1.0, price_multiplier),
                "tier": (
                    sum(current_tiers.values()) / 3.0,
                    sum(proposed_tiers.values()) / 3.0,
                ),
                "quota": (
                    sum(current_quotas.values()) / 3.0,
                    sum(proposed_quotas.values()) / 3.0,
                ),
                "marketing": (current_marketing, weekly_marketing),
                "development": (
                    current_development_daily * 7.0,
                    development_daily * 7.0,
                ),
                "targeted_development": (
                    sum(current_targeted_development.values()) * 7.0,
                    proposal.targeted_development_daily * 7.0,
                ),
                "lead_promotion": (observed_promotion, promotion),
            }[proposal.experiment_control]
            baseline_configuration = {
                "prices": dict(zip(("A", "B", "C"), prices, strict=True)),
                "model_tiers": current_tiers,
                "usage_quotas": current_quotas,
                "weekly_marketing_spend": (
                    weekly_marketing if carries_measurement_exposure else current_marketing
                ),
                "daily_development_spend": current_development_daily,
                "targeted_development_daily": current_targeted_development,
                "lead_promotion_monthly": observed_promotion,
                # The operating level to return to. It differs from the control
                # arm whenever the probe adds measurement exposure to both arms.
                "pre_experiment_weekly_marketing_spend": current_marketing,
                "pre_experiment_targeted_ad_spend": current_targeted_ads,
            }
            treatment_configuration = {
                "prices": {
                    tier: price * (
                        price_multiplier
                        if proposal.experiment_control == "price"
                        else 1.0
                    )
                    for tier, price in zip(("A", "B", "C"), prices, strict=True)
                },
                "model_tiers": (
                    proposed_tiers
                    if proposal.experiment_control == "tier"
                    else current_tiers
                ),
                "usage_quotas": (
                    proposed_quotas
                    if proposal.experiment_control == "quota"
                    else current_quotas
                ),
                "weekly_marketing_spend": (
                    weekly_marketing
                    if proposal.experiment_control == "marketing"
                    or carries_measurement_exposure
                    else current_marketing
                ),
                "daily_development_spend": (
                    development_daily
                    if proposal.experiment_control == "development"
                    else current_development_daily
                ),
                "targeted_development_daily": (
                    {target_segment: proposal.targeted_development_daily}
                    if proposal.experiment_control == "targeted_development"
                    else current_targeted_development
                ),
                "lead_promotion_monthly": (
                    promotion
                    if proposal.experiment_control == "lead_promotion"
                    else observed_promotion
                ),
            }
            measurement_plan = (
                ExperimentMeasurement(
                    source="configuration",
                    metric=proposal.experiment_control,
                    target_segment=target_segment,
                    target_channel=proposal.target_channel,
                ),
                ExperimentMeasurement(
                    source="cohort",
                    metric="conversion_rate",
                    target_segment=target_segment,
                    target_channel=proposal.target_channel,
                    minimum_exposure=30,
                    attribution_window_weeks=1,
                ),
                ExperimentMeasurement(
                    source="quality",
                    metric="delivered_quality_proxy",
                    target_segment=target_segment,
                    attribution_window_weeks=duration_weeks,
                    decision_grade=False,
                ),
                ExperimentMeasurement(
                    source="ledger",
                    metric="cumulative_downside",
                    attribution_window_weeks=duration_weeks,
                ),
            )
            # This is a policy budget, not an assumed benefit. The robust planner may
            # still reject a program whose modeled downside is smaller than this cap.
            # A ceiling, not a level: the Executive's own budget stands whenever
            # it fits inside what the company can afford to lose.
            budget_ceiling = (
                min(300_000.0, max(30_000.0, observation.cash * 0.30))
                if proposal.experiment_control == "targeted_development"
                else min(30_000.0, max(1_000.0, observation.cash * 0.03))
            )
            downside_budget = (
                min(proposal.maximum_downside_budget, budget_ceiling)
                if proposal.maximum_downside_budget > 0.0
                else budget_ceiling
            )
            experiment_program = ExperimentProgram(
                commitment_id=f"{proposal.hypothesis_id}-{week}",
                control=proposal.experiment_control,
                protocol_version="experiment-program-v2",
                started_week=week,
                minimum_maturity_week=maturity_week,
                maximum_end_week=experiment_expires_week,
                baseline_value=baseline_value,
                treatment_value=treatment_value,
                maximum_cumulative_downside=downside_budget,
                expected_observation=proposal.expected_observation,
                falsification_condition=(
                    f"The expected observation does not occur by week "
                    f"{experiment_expires_week}."
                ),
                decision_rule=proposal.decision_rule,
                target_segment=target_segment,
                target_channel=proposal.target_channel,
                acquisition_probe_weekly_spend=(
                    proposal.weekly_marketing_spend
                    if proposal.experiment_control
                    in {"development", "targeted_development"}
                    else 0.0
                ),
                baseline_targeted_development=(
                    current_targeted_development
                    if proposal.experiment_control == "targeted_development"
                    else {}
                ),
                treatment_targeted_development=(
                    {target_segment: proposal.targeted_development_daily}
                    if proposal.experiment_control == "targeted_development"
                    else {}
                ),
                baseline_targeted_ad_spend=(
                    current_targeted_ads
                    if proposal.experiment_control == "targeted_development"
                    else {}
                ),
                baseline_configuration=baseline_configuration,
                treatment_configuration=treatment_configuration,
                measurement_plan=measurement_plan,
            )
        return ActionPlan(
            name=proposal.name,
            strategy_family=(
                f"executive_experiment_{proposal.experiment_control}_"
                f"{proposal.hypothesis_id}"
                if is_experiment
                else f"executive_{proposal.strategy_family}_{candidate_index}"
            ),
            rationale=(
                f"Hypothesis: {proposal.hypothesis} Expected observation: "
                f"{proposal.expected_observation} Rationale: {proposal.rationale}"
            ),
            commands=commands,
            proposal_kind=proposal.proposal_kind,
            hypothesis_id=proposal.hypothesis_id,
            experiment_control=(
                proposal.experiment_control if is_experiment else None
            ),
            evidence_regime=(
                evidence_regime
                if segment_is_observed
                else f"{evidence_regime}:unobserved_segment"
            ),
            experiment_expires_week=experiment_expires_week,
            experiment_program=experiment_program,
            enterprise_engage=proposal.enterprise_engage,
            enterprise_target_price_per_seat=(
                proposal.enterprise_target_price_per_seat
                if proposal.enterprise_engage
                else None
            ),
            enterprise_floor_price_per_seat=(
                proposal.enterprise_floor_price_per_seat or None
                if proposal.enterprise_engage
                else None
            ),
            enterprise_max_new_seats=(
                float(proposal.enterprise_max_new_seats)
                if proposal.enterprise_engage and proposal.enterprise_max_new_seats
                else None
            ),
        )

    async def decide(
        self,
        *,
        run: RunRecord,
        observation: ObservationSnapshot,
    ) -> tuple[ActionPlan, CashForecasts]:
        input_payload = {
            "run_id": str(run.id),
            "week": observation.day // 7,
            "horizon_days": run.horizon_days,
            "observation": observation.model_dump(mode="json"),
        }
        output = await self.provider.generate_structured(
            system_prompt=self.system_prompt,
            user_prompt=json.dumps(input_payload, separators=(",", ":"), sort_keys=True),
            output_schema=ExecutiveDecisionOutput,
        )
        return output.to_domain(run_id=run.id, week=observation.day // 7)
