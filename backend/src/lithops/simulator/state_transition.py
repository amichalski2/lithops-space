"""One fixed, inspectable weekly business state transition."""

from __future__ import annotations

from dataclasses import replace

from lithops.domain.world_model import WorldModelParameterName
from lithops.simulator.components import (
    BASELINE_TRANSITION_ASSEMBLY,
    TransitionModelAssembly,
)
from lithops.simulator.models import (
    PendingQualityEffect,
    PendingResearch,
    SimulationAction,
    SimulationState,
    WeeklyShock,
)

MODEL_TIER_COST_PER_USAGE_UNIT = {1: 0.0003, 2: 0.002, 3: 0.006, 4: 0.012, 5: 0.03}

# Neutral fallbacks for artifacts fitted before these parameters existed. They
# mirror the bootstrap priors so an older world model keeps transitioning.
_DEFAULT_QUOTA_SATURATION = 1.0
_DEFAULT_PREMIUM_QUALITY_SCALE = 0.60
_DEFAULT_DEVELOPMENT_QUALITY_RESPONSE = 0.01
_DEFAULT_DEVELOPMENT_QUALITY_EXPONENT = 0.55
_DEFAULT_QUOTA_DEMAND_REFERENCE = 200.0
_DEFAULT_CAPACITY_TIER_STEP = 2.5
_DEFAULT_ADS_REVENUE_RATE = 0.5
_DEFAULT_ADS_QUALITY_TRADEOFF = 0.15
_DEFAULT_OPS_RELIABILITY_RESPONSE = 0.2
_DEFAULT_SOCIAL_LEAD_RESPONSE = 0.1
_DEFAULT_ENTERPRISE_PRICE_SENSITIVITY = 1.0
_DEFAULT_RESEARCH_LAG_WEEKS_PER_TIER = 2.0
_DEFAULT_RESEARCH_QUALITY_PER_TIER = 0.02
_DEFAULT_PARTICIPATION_CONVERSION_RATE = 0.15
_DEFAULT_PARTICIPATION_SOFTNESS = 0.05


def _saturating(value: float, reference: float) -> float:
    """Bounded diminishing response in [0, 1) against a measured reference."""

    if reference <= 0.0 or value <= 0.0:
        return 0.0
    return value / (value + reference)


def _bounded(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _served_usage_ratio(quota: float, demand_reference: float) -> float:
    """Share of demanded usage an allowance actually serves."""

    if demand_reference <= 0.0:
        return 1.0
    return _bounded(quota / demand_reference, 0.0, 1.0)


def effective_participation_floor(state: SimulationState) -> float | None:
    """The highest participation bar the run has actually measured.

    Two instruments feed it: the purchased floor (the bar at purchase time) and
    the churn-revealed lower bound (where the bar has provably moved since).
    Both absent means unmeasured — never zero.
    """

    measured = state.measured_quality_floor_individual
    revealed = state.revealed_quality_bar_lower_bound
    if measured is None and revealed is None:
        return None
    return max(value for value in (measured, revealed) if value is not None)


def advance_simulation_week(
    state: SimulationState,
    action: SimulationAction,
    parameters: dict[WorldModelParameterName, float],
    shock: WeeklyShock | None = None,
    assembly: TransitionModelAssembly = BASELINE_TRANSITION_ASSEMBLY,
) -> SimulationState:
    price_elasticity = parameters[WorldModelParameterName.PRICE_ELASTICITY]
    marketing_saturation = parameters[WorldModelParameterName.MARKETING_SATURATION]
    churn_sensitivity = parameters[WorldModelParameterName.CHURN_SENSITIVITY]
    quality_lag_weeks = max(
        1,
        round(parameters[WorldModelParameterName.QUALITY_LAG_WEEKS]),
    )
    segment_response = parameters[WorldModelParameterName.SEGMENT_RESPONSE]
    marketing_started = (
        action.marketing_spend_start_week is None
        or state.week >= action.marketing_spend_start_week
    )
    marketing_spend = (
        state.marketing_spend
        if not marketing_started
        else action.marketing_spend
        if action.marketing_spend_until_week is None
        or state.week < action.marketing_spend_until_week
        else (action.marketing_spend_after_experiment or 0.0)
    )
    targeting_active = (
        marketing_started
        and (
            action.marketing_spend_until_week is None
            or state.week < action.marketing_spend_until_week
        )
    )
    channel_spend = {
        "social_media": 0.0,
        "search_ads": 0.0,
        "linkedin": 0.0,
        "content_marketing": 0.0,
        "referral_program": 0.0,
    }
    if targeting_active and action.targeted_ad_allocations:
        for allocation in action.targeted_ad_allocations:
            channel_spend[allocation.channel] += allocation.daily_spend * 7.0
    else:
        current_channel_spend = {
            "social_media": state.marketing_spend_social_media_weekly,
            "search_ads": state.marketing_spend_search_ads_weekly,
            "linkedin": state.marketing_spend_linkedin_weekly,
            "content_marketing": state.marketing_spend_content_marketing_weekly,
            "referral_program": state.marketing_spend_referral_program_weekly,
        }
        observed_total = sum(current_channel_spend.values())
        if observed_total > 0:
            channel_spend = {
                channel: marketing_spend * amount / observed_total
                for channel, amount in current_channel_spend.items()
            }
    development_spend = (
        action.development_spend
        if action.development_spend_until_week is None
        or state.week < action.development_spend_until_week
        else (action.development_spend_after_experiment or 0.0)
    )
    targeted_development_active = (
        action.targeted_development_spend_until_week is None
        or state.week < action.targeted_development_spend_until_week
    )
    targeted_development_spend = (
        7.0
        * sum(
            allocation.daily_spend
            for allocation in action.targeted_development_allocations
        )
        if targeted_development_active
        else (
            state.targeted_development_spend
            if action.targeted_development_spend_after_experiment is None
            else action.targeted_development_spend_after_experiment
        )
    )
    selected_lead_promotion = (
        state.lead_promotion_monthly
        if action.lead_promotion_monthly is None
        else action.lead_promotion_monthly
    )
    lead_promotion = (
        selected_lead_promotion
        if action.lead_promotion_until_week is None
        or state.week < action.lead_promotion_until_week
        else (
            state.lead_promotion_monthly
            if action.lead_promotion_after_experiment is None
            else action.lead_promotion_after_experiment
        )
    )
    operations_spend = (
        state.operations_spend
        if action.operations_spend is None
        else action.operations_spend
    )
    current_model_tiers = (state.model_tier_a, state.model_tier_b, state.model_tier_c)
    selected_model_tiers = (
        state.model_tier_a if action.model_tier_a is None else action.model_tier_a,
        state.model_tier_b if action.model_tier_b is None else action.model_tier_b,
        state.model_tier_c if action.model_tier_c is None else action.model_tier_c,
    )
    current_compute_rate = sum(
        MODEL_TIER_COST_PER_USAGE_UNIT[tier] for tier in current_model_tiers
    ) / len(current_model_tiers)
    selected_compute_rate = sum(
        MODEL_TIER_COST_PER_USAGE_UNIT[tier] for tier in selected_model_tiers
    ) / len(selected_model_tiers)
    compute_cost_multiplier = selected_compute_rate / current_compute_rate

    quota_saturation = parameters.get(
        WorldModelParameterName.QUOTA_SATURATION, _DEFAULT_QUOTA_SATURATION
    )
    quota_demand_reference = parameters.get(
        WorldModelParameterName.QUOTA_DEMAND_REFERENCE,
        _DEFAULT_QUOTA_DEMAND_REFERENCE,
    )
    selected_quotas = (
        state.usage_quota_a if action.usage_quota_a is None else action.usage_quota_a,
        state.usage_quota_b if action.usage_quota_b is None else action.usage_quota_b,
        state.usage_quota_c if action.usage_quota_c is None else action.usage_quota_c,
    )
    current_quota = state.average_usage_quota
    selected_quota = (
        None
        if all(value is None for value in selected_quotas)
        else sum(value or 0.0 for value in selected_quotas) / len(selected_quotas)
    )
    # Observed usage is clamped by the active allowance, so it is a lower bound on
    # demand. The larger of that bound and the learned reference is the honest
    # denominator; it stays uncertain until an allowance above demand reveals it.
    # A purchased estimate replaces the wide prior as the demand scale; the
    # censored observation still raises it when it is already higher.
    demand_reference = max(
        state.daily_usage_per_customer,
        state.estimated_usage_demand_per_day
        if state.estimated_usage_demand_per_day is not None
        else quota_demand_reference,
    )
    allowance_is_known = current_quota is not None or selected_quota is not None
    if allowance_is_known:
        current_served_ratio = _served_usage_ratio(current_quota or 0.0, demand_reference)
        selected_served_ratio = _served_usage_ratio(
            selected_quota or 0.0, demand_reference
        )
    else:
        # No allowance information: rationing is not modelled for this state.
        current_served_ratio = 1.0
        selected_served_ratio = 1.0
    # Rationing degrades the value a customer perceives, with a learned exponent.
    current_fulfillment = current_served_ratio**quota_saturation
    selected_fulfillment = selected_served_ratio**quota_saturation
    # An allowance of zero serves nothing, so it converts nothing; above zero the
    # ratio is smoothed the same way the tier ratio is.
    quota_conversion_multiplier = (
        0.0
        if selected_fulfillment <= 0.0
        else (selected_fulfillment + 0.05) / (current_fulfillment + 0.05)
    )

    capacity_tier_step = parameters.get(
        WorldModelParameterName.CAPACITY_TIER_STEP, _DEFAULT_CAPACITY_TIER_STEP
    )
    selected_capacity_tier = (
        state.capacity_tier if action.capacity_tier is None else action.capacity_tier
    )
    capacity_scale = capacity_tier_step ** (selected_capacity_tier - state.capacity_tier)
    # Capacity bought this week serves from the following week onward, so the
    # conversion cap still uses the capacity actually in place.
    capacity = max(1.0, state.capacity * capacity_scale)
    capacity_spend_weekly = state.capacity_spend_weekly * capacity_scale

    recurring_promotion = (
        state.recurring_promotion_monthly
        if action.recurring_promotion_monthly is None
        else action.recurring_promotion_monthly
    )
    ads_strength = (
        state.ads_strength if action.ads_strength is None else action.ads_strength
    )
    targeted_ops_spend = (
        state.targeted_ops_spend
        if action.targeted_ops_spend is None
        else action.targeted_ops_spend
    )
    social_posts = (
        state.social_posts_weekly
        if action.social_posts is None
        else float(action.social_posts)
    )

    quality_transition = assembly.quality.transition(
        state=state,
        action=action,
        # Targeted work is documented as more efficient for the selected cohort and
        # harder for competitors to copy. This is an action-semantics prior, not a
        # benchmark threshold or winning spend level.
        development_spend=development_spend + 5.0 * targeted_development_spend,
        quality_lag_weeks=(1 if targeted_development_spend > 0 else quality_lag_weeks),
        development_quality_response=parameters.get(
            WorldModelParameterName.DEVELOPMENT_QUALITY_RESPONSE,
            _DEFAULT_DEVELOPMENT_QUALITY_RESPONSE,
        ),
        development_quality_exponent=parameters.get(
            WorldModelParameterName.DEVELOPMENT_QUALITY_EXPONENT,
            _DEFAULT_DEVELOPMENT_QUALITY_EXPONENT,
        ),
    )
    # Tiers still maturing tick down one week; a landed tier leaves the queue.
    research_tiers_in_progress = tuple(
        item.model_copy(update={"weeks_remaining": item.weeks_remaining - 1})
        for item in state.research_tiers_in_progress
        if item.weeks_remaining > 1
    )
    research_spend = 0.0
    research_tiers_started = state.research_tiers_started
    if (
        action.research_project_tier
        and action.research_project_tier not in state.research_tiers_started
        and not any(
            item.tier == action.research_project_tier
            for item in state.research_tiers_in_progress
        )
    ):
        # A research programme lands later and larger than weekly development
        # work; it joins the same pending-effect queue. One invocation per tier
        # can mature at a time — the environment's own listing semantics — so a
        # candidate action repeated across a rollout horizon restarts (and pays
        # for) a tier only after the previous invocation has landed.
        lag_per_tier = parameters.get(
            WorldModelParameterName.RESEARCH_LAG_WEEKS_PER_TIER,
            _DEFAULT_RESEARCH_LAG_WEEKS_PER_TIER,
        )
        quality_per_tier = parameters.get(
            WorldModelParameterName.RESEARCH_QUALITY_PER_TIER,
            _DEFAULT_RESEARCH_QUALITY_PER_TIER,
        )
        # Quality lives in [0, 1], so that is the only bound this transition may
        # assert on its own; the tighter 0.25 ceiling it used to apply silently
        # truncated large programmes, which made the one step-change lever
        # unable to ever forecast as one.
        improvement = _bounded(
            quality_per_tier * action.research_project_tier, 1e-6, 1.0
        )
        weeks_remaining = max(
            1, min(52, round(lag_per_tier * action.research_project_tier))
        )
        facts = next(
            (
                entry
                for entry in state.research_catalog
                if entry.tier == action.research_project_tier
            ),
            None,
        )
        if facts is not None:
            # The environment's own listing overrides the learned draw wherever
            # it speaks: the charge is exact, and the listed means centre the
            # improvement and the lag. Observed facts, not priors; the listing
            # publishes means rather than ranges, so the residual sampling
            # variance stays with the rollout's process noise.
            research_spend = facts.cost
            if facts.mean_quality_boost is not None:
                improvement = _bounded(facts.mean_quality_boost, 1e-6, 1.0)
            if facts.mean_weeks is not None:
                weeks_remaining = max(1, min(52, facts.mean_weeks))
        quality_transition = replace(
            quality_transition,
            pending_effects=(
                *quality_transition.pending_effects,
                PendingQualityEffect(
                    weeks_remaining=weeks_remaining,
                    improvement=improvement,
                ),
            ),
        )
        research_tiers_in_progress = (
            *research_tiers_in_progress,
            PendingResearch(
                tier=action.research_project_tier,
                weeks_remaining=weeks_remaining,
            ),
        )
        research_tiers_started = (
            *research_tiers_started,
            action.research_project_tier,
        )

    current_catalog_price = state.effective_catalog_price_per_customer_weekly
    price_change = action.price_per_customer_weekly / current_catalog_price - 1.0
    entry_price_monthly = state.entry_price_monthly or (
        current_catalog_price * 30.0 / 7.0
    )
    current_net_entry_price = max(
        0.01,
        entry_price_monthly
        - state.lead_promotion_monthly
        - state.recurring_promotion_monthly,
    )
    proposed_entry_price = entry_price_monthly * (
        action.price_per_customer_weekly / current_catalog_price
    )
    proposed_net_entry_price = max(
        0.01, proposed_entry_price - lead_promotion - recurring_promotion
    )
    acquisition_price_change = proposed_net_entry_price / current_net_entry_price - 1.0
    price_conversion_multiplier = max(
        0.05,
        1.0 - price_elasticity * acquisition_price_change,
    )

    # What the customer actually receives: rationed by the allowance and reduced by
    # in-product advertising.
    ads_quality_tradeoff = parameters.get(
        WorldModelParameterName.ADS_QUALITY_TRADEOFF, _DEFAULT_ADS_QUALITY_TRADEOFF
    )
    delivered_quality = _bounded(
        quality_transition.delivered_quality
        * selected_fulfillment
        * (1.0 - ads_quality_tradeoff * ads_strength),
        0.0,
        1.0,
    )
    # Support and reliability work directed at existing customers, measured against
    # the observed unit cost of serving them.
    ops_reliability_response = parameters.get(
        WorldModelParameterName.OPS_RELIABILITY_RESPONSE,
        _DEFAULT_OPS_RELIABILITY_RESPONSE,
    )
    ops_spend_per_customer = (
        targeted_ops_spend / state.customers if state.customers > 0 else 0.0
    )
    ops_retention_effect = ops_reliability_response * _saturating(
        ops_spend_per_customer, state.operating_cost_per_customer_weekly
    )
    churn_change = churn_sensitivity * (
        max(price_change, 0.0) * 0.04 + (0.5 - delivered_quality) * 0.02
    ) - ops_retention_effect * state.churn_rate
    if shock is not None:
        churn_change += shock.churn_delta
    churn_rate = _bounded(state.churn_rate + churn_change, 0.0, 1.0)
    retained_customers = state.customers * (1.0 - churn_rate)
    lead_arrival = assembly.lead_arrival.transition(
        state=state,
        action=action,
        marketing_spend=marketing_spend,
        marketing_saturation=marketing_saturation,
        segment_response=segment_response,
    )
    # Owned-channel publishing is a bounded, saturating lift on arriving leads.
    social_lead_response = parameters.get(
        WorldModelParameterName.SOCIAL_LEAD_RESPONSE, _DEFAULT_SOCIAL_LEAD_RESPONSE
    )
    social_lead_multiplier = 1.0 + social_lead_response * _saturating(social_posts, 1.0)
    if social_lead_multiplier != 1.0:
        lead_arrival = replace(
            lead_arrival,
            predicted_leads=lead_arrival.predicted_leads * social_lead_multiplier,
        )
    funnel_transition = assembly.conversion.transition(
        state=state,
        action=action,
        lead_arrival=lead_arrival,
        retained_customers=retained_customers,
        quality_multiplier=(
            quality_transition.conversion_multiplier * quota_conversion_multiplier
        ),
        price_multiplier=price_conversion_multiplier,
        shock=shock,
        # The participation gate compares what a customer actually receives —
        # rationed and ads-degraded — against the highest bar actually measured:
        # the purchased floor is the bar at purchase time, and the run's own
        # mass churn reveals where the bar has moved since. Ignoring the
        # revealed bound made the forecast keep promising conversions from a
        # segment the drift had already closed.
        delivered_quality_effective=delivered_quality,
        participation_floor=effective_participation_floor(state),
        participation_rate=parameters.get(
            WorldModelParameterName.PARTICIPATION_CONVERSION_RATE,
            _DEFAULT_PARTICIPATION_CONVERSION_RATE,
        ),
        participation_softness=parameters.get(
            WorldModelParameterName.PARTICIPATION_SOFTNESS,
            _DEFAULT_PARTICIPATION_SOFTNESS,
        ),
    )
    acquired_customers = funnel_transition.acquired_customers
    customers = retained_customers + acquired_customers

    expected_arpu = state.projected_arpu(
        action.price_per_customer_weekly,
        delivered_quality=quality_transition.delivered_quality,
        premium_quality_scale=parameters.get(
            WorldModelParameterName.PREMIUM_QUALITY_SCALE,
            _DEFAULT_PREMIUM_QUALITY_SCALE,
        ),
    )
    # A recurring discount comes off the listed price every billing period.
    expected_arpu = max(0.0, expected_arpu - recurring_promotion * 7.0 / 30.0)
    realized_arpu = expected_arpu * (
        1.0 if shock is None else shock.revenue_multiplier
    )
    retained_revenue = retained_customers * realized_arpu
    first_period_promotion_weekly = lead_promotion * 7.0 / 30.0
    acquired_revenue = acquired_customers * max(
        0.0,
        realized_arpu - first_period_promotion_weekly,
    )
    ads_revenue_rate = parameters.get(
        WorldModelParameterName.ADS_REVENUE_RATE, _DEFAULT_ADS_REVENUE_RATE
    )
    ads_revenue = ads_revenue_rate * ads_strength * customers

    # Negotiated seat demand. Acceptance needs delivered quality above what such
    # buyers require and a price the reference offer can bear; both are learned.
    accepted_seats = 0.0
    enterprise_revenue = state.enterprise_revenue_weekly
    if action.enterprise_engage and action.enterprise_target_price_per_seat:
        # No invented floor. What these buyers require is bought, not assumed:
        # until a group insight reports it, seat acceptance is unmeasured rather
        # than zero, and the card says so instead of quietly forecasting none.
        # The floor is a purchased measurement carried on the state — it was
        # never a learnable parameter, and as a parameter nothing ever wrote it.
        quality_floor = state.measured_quality_floor_enterprise
        price_sensitivity = parameters.get(
            WorldModelParameterName.ENTERPRISE_PRICE_SENSITIVITY,
            _DEFAULT_ENTERPRISE_PRICE_SENSITIVITY,
        )
        quality_headroom = (
            0.0
            if quality_floor is None
            else _bounded(
                (delivered_quality - quality_floor) / max(quality_floor, 1e-6),
                0.0,
                1.0,
            )
        )
        reference_price = max(entry_price_monthly, 1e-6)
        price_ratio = action.enterprise_target_price_per_seat / reference_price
        price_acceptance = _bounded(
            1.0 - price_sensitivity * max(0.0, price_ratio - 1.0), 0.0, 1.0
        )
        acceptance = quality_headroom * price_acceptance
        available_seats = state.open_enterprise_seats
        if action.enterprise_max_new_seats is not None:
            available_seats = min(available_seats, action.enterprise_max_new_seats)
        accepted_seats = available_seats * acceptance
        enterprise_revenue = (
            state.enterprise_revenue_weekly
            + accepted_seats * action.enterprise_target_price_per_seat * 7.0 / 30.0
        )
    revenue = retained_revenue + acquired_revenue + ads_revenue + enterprise_revenue
    # Compute is metered on usage actually served, so an allowance change moves
    # cost with it. The observed cost is the anchor whenever usage was already
    # being served; the metered floor keeps the forecast honest when it was not.
    scaled_observed_cost = (
        state.operating_cost_per_customer_weekly
        * compute_cost_multiplier
        * (selected_served_ratio / current_served_ratio)
        if current_served_ratio > 0.0
        else 0.0
    )
    metered_cost = (
        selected_served_ratio * demand_reference * selected_compute_rate * 7.0
        if allowance_is_known
        else 0.0
    )
    operating_cost_per_customer = max(scaled_observed_cost, metered_cost)
    operating_cost = customers * operating_cost_per_customer
    cash = (
        state.cash
        + revenue
        - operating_cost
        - operations_spend
        - targeted_ops_spend
        - capacity_spend_weekly
        - marketing_spend
        - development_spend
        - targeted_development_spend
        - research_spend
        + (0.0 if shock is None else shock.cash_flow_delta)
    )
    reputation = _bounded(
        state.reputation
        + (quality_transition.quality - 0.5) * 0.01
        - max(price_change, 0.0) * 0.01,
        0.0,
        1.0,
    )

    return SimulationState(
        week=state.week + 1,
        cash=cash,
        revenue_weekly=revenue,
        customers=customers,
        churn_rate=churn_rate,
        price_per_customer_weekly=realized_arpu,
        catalog_price_per_customer_weekly=action.price_per_customer_weekly,
        # The catalog's range is a property of the company, not of one week, so
        # it survives the transition. Dropping it left the mix model reading
        # None from the second week onward and quietly falling back to a single
        # blended price — the very thing it exists to replace.
        catalog_price_entry_weekly=state.catalog_price_entry_weekly,
        catalog_price_premium_weekly=state.catalog_price_premium_weekly,
        entry_price_monthly=entry_price_monthly
        * (action.price_per_customer_weekly / current_catalog_price),
        lead_promotion_monthly=lead_promotion,
        price_realization_ratio=(
            state.price_realization_ratio
            if state.price_realization_ratio is not None
            else state.price_per_customer_weekly / current_catalog_price
        ),
        weekly_acquisition=acquired_customers,
        weekly_leads=funnel_transition.predicted_leads,
        weekly_conversions=acquired_customers,
        weekly_lost_leads=funnel_transition.lost_leads,
        total_leads=state.total_leads + funnel_transition.predicted_leads,
        total_conversions=state.total_conversions + acquired_customers,
        total_lost_leads=state.total_lost_leads + funnel_transition.lost_leads,
        marketing_spend=marketing_spend,
        marketing_spend_social_media_weekly=channel_spend["social_media"],
        marketing_spend_search_ads_weekly=channel_spend["search_ads"],
        marketing_spend_linkedin_weekly=channel_spend["linkedin"],
        marketing_spend_content_marketing_weekly=channel_spend[
            "content_marketing"
        ],
        marketing_spend_referral_program_weekly=channel_spend[
            "referral_program"
        ],
        development_spend=development_spend,
        targeted_development_spend=targeted_development_spend,
        operations_spend=operations_spend,
        capacity_spend_weekly=capacity_spend_weekly,
        product_quality=quality_transition.quality,
        capacity=capacity,
        reputation=reputation,
        operating_cost_per_customer_weekly=operating_cost_per_customer,
        model_tier_a=selected_model_tiers[0],
        model_tier_b=selected_model_tiers[1],
        model_tier_c=selected_model_tiers[2],
        usage_quota_a=selected_quotas[0],
        usage_quota_b=selected_quotas[1],
        usage_quota_c=selected_quotas[2],
        daily_usage_per_customer=selected_served_ratio * demand_reference,
        capacity_tier=selected_capacity_tier,
        recurring_promotion_monthly=recurring_promotion,
        ads_strength=ads_strength,
        ads_revenue_weekly=ads_revenue,
        targeted_ops_spend=targeted_ops_spend,
        social_posts_weekly=social_posts,
        active_seats=state.active_seats + accepted_seats,
        open_enterprise_threads=state.open_enterprise_threads,
        open_enterprise_seats=max(0.0, state.open_enterprise_seats - accepted_seats),
        enterprise_revenue_weekly=enterprise_revenue,
        pending_quality_effects=quality_transition.pending_effects,
        # Observed facts are properties of the company's knowledge, not of one
        # week, so both survive the transition.
        research_catalog=state.research_catalog,
        research_tiers_in_progress=research_tiers_in_progress,
        research_tiers_started=research_tiers_started,
        research_spend_weekly=research_spend,
        cash_flow_adjustment_weekly=0.0 if shock is None else shock.cash_flow_delta,
    )
