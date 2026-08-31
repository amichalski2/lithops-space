"""Public-table evidence collection for CEO-Bench."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from lithops.domain.evidence import (
    AcquisitionEvidence,
    CohortEvidence,
    ConfigurationEvidence,
    LedgerEvidence,
    QualityEvidence,
    WeeklyEvidencePacket,
)
from lithops.domain.models import ObservationSnapshot
from lithops.domain.public_instruments import MODEL_TIER_QUALITY_MULTIPLIER

ACQUISITION_EVIDENCE_QUERY = """
WITH current_day AS (SELECT COALESCE(MAX(day), 0) AS day FROM service_day)
SELECT
    COALESCE(group_id, 'unknown') AS segment,
    COALESCE(channel_id, 'unknown') AS channel,
    COALESCE(SUM(leads_generated), 0) AS leads,
    COALESCE(SUM(spend), 0) AS spend
FROM ad_channel_leads
WHERE day > (SELECT day FROM current_day) - 7
  AND day <= (SELECT day FROM current_day)
GROUP BY group_id, channel_id
ORDER BY group_id, channel_id
""".strip()

COHORT_EVIDENCE_QUERY = """
WITH current_day AS (SELECT COALESCE(MAX(day), 0) AS day FROM service_day)
SELECT
    COALESCE(c.group_id, 'unknown') AS segment,
    COALESCE(c.acquisition_source, 'unknown') AS channel,
    COUNT(*) AS leads,
    COALESCE(SUM(CASE WHEN s.status IN ('subscribed', 'cancelled') THEN 1 ELSE 0 END), 0)
        AS conversions,
    COALESCE(SUM(CASE WHEN s.status = 'lost' THEN 1 ELSE 0 END), 0) AS losses,
    COALESCE(SUM(CASE WHEN s.status = 'lead' THEN 1 ELSE 0 END), 0) AS pending
FROM subscriptions AS s
JOIN customers AS c ON c.customer_id = s.customer_id
WHERE s.start_day > (SELECT day FROM current_day) - 7
  AND s.start_day <= (SELECT day FROM current_day)
GROUP BY c.group_id, c.acquisition_source
ORDER BY c.group_id, c.acquisition_source
""".strip()

LEDGER_EVIDENCE_QUERY = """
WITH current_day AS (SELECT COALESCE(MAX(day), 0) AS day FROM service_day)
SELECT
    category,
    COALESCE(SUM(CASE
        WHEN day > (SELECT day FROM current_day) - 7
         AND day <= (SELECT day FROM current_day)
        THEN amount ELSE 0 END), 0) AS weekly_amount,
    COALESCE(SUM(amount), 0) AS cumulative_amount
FROM ledger
GROUP BY category
ORDER BY category
""".strip()

CONFIGURATION_EVIDENCE_QUERY = """
SELECT
    price_A, price_B, price_C,
    tier_A, tier_B, tier_C,
    quota_A, quota_B, quota_C,
    ad_spend_social_media, ad_spend_search_ads, ad_spend_linkedin,
    ad_spend_content_marketing, ad_spend_referral_program,
    spend_operations, spend_development, capacity_tier,
    COALESCE((SELECT settings_json FROM config_overrides
        WHERE tool_name = 'set_lead_promotion' ORDER BY day DESC, id DESC LIMIT 1), '{}')
        AS lead_promotion_json,
    COALESCE((SELECT settings_json FROM config_overrides
        WHERE tool_name = 'set_targeted_ad_spend' ORDER BY day DESC, id DESC LIMIT 1), '{}')
        AS targeted_ads_json,
    COALESCE((SELECT settings_json FROM config_overrides
        WHERE tool_name = 'set_targeted_dev_spend' ORDER BY day DESC, id DESC LIMIT 1), '{}')
        AS targeted_development_json,
    COALESCE((SELECT settings_json FROM config_overrides
        WHERE tool_name = 'set_promotion' ORDER BY day DESC, id DESC LIMIT 1), '{}')
        AS recurring_promotion_json,
    COALESCE((SELECT settings_json FROM config_overrides
        WHERE tool_name = 'set_ads_strength' ORDER BY day DESC, id DESC LIMIT 1), '{}')
        AS ads_strength_json,
    COALESCE((SELECT settings_json FROM config_overrides
        WHERE tool_name = 'set_targeted_ops_spend' ORDER BY day DESC, id DESC LIMIT 1), '{}')
        AS targeted_ops_json
FROM config_history
ORDER BY day DESC
LIMIT 1
""".strip()

_TIER_MULTIPLIERS = MODEL_TIER_QUALITY_MULTIPLIER
_CHANNEL_COLUMNS = {
    "social_media": "ad_spend_social_media",
    "search_ads": "ad_spend_search_ads",
    "linkedin": "ad_spend_linkedin",
    "content_marketing": "ad_spend_content_marketing",
    "referral_program": "ad_spend_referral_program",
}


def _number(row: dict[str, Any], name: str, default: float = 0.0) -> float:
    value = row.get(name, default)
    return float(default if value is None else value)


def _targeted_development(value: Any) -> dict[str, float]:
    try:
        decoded = json.loads(str(value or "{}"))
    except (TypeError, ValueError):
        return {}
    if isinstance(decoded, dict) and isinstance(decoded.get("targeted_spend"), dict):
        decoded = decoded["targeted_spend"]
    if not isinstance(decoded, dict):
        return {}
    return {
        str(segment): float(amount)
        for segment, amount in decoded.items()
        if isinstance(amount, int | float) and float(amount) >= 0.0
    }


async def collect_weekly_evidence(
    query: Callable[[str], Awaitable[list[dict[str, Any]]]],
    observation: ObservationSnapshot,
) -> WeeklyEvidencePacket:
    acquisition_rows = await query(ACQUISITION_EVIDENCE_QUERY)
    cohort_rows = await query(COHORT_EVIDENCE_QUERY)
    ledger_rows = await query(LEDGER_EVIDENCE_QUERY)
    config_rows = await query(CONFIGURATION_EVIDENCE_QUERY)
    if len(config_rows) != 1:
        raise ValueError("CEO-Bench configuration evidence must contain exactly one row")
    config = config_rows[0]
    tiers = {plan: int(_number(config, f"tier_{plan}", 1)) for plan in "ABC"}
    prices = {plan: _number(config, f"price_{plan}") for plan in "ABC"}
    targeted = _targeted_development(config.get("targeted_development_json"))
    segments = tuple(
        sorted(
            segment.strip()
            for segment in str(observation.metrics.get("known_segments") or "S1").split(",")
            if segment.strip()
        )
    ) or ("S1",)
    base_quality = float(observation.metrics.get("product_quality", 0.2) or 0.2)
    quality = tuple(
        QualityEvidence(
            segment=segment,
            plan=plan,
            base_quality_proxy=base_quality,
            model_tier=tiers[plan],
            tier_multiplier=_TIER_MULTIPLIERS[tiers[plan]],
            delivered_quality_proxy=min(1.0, base_quality * _TIER_MULTIPLIERS[tiers[plan]]),
            targeted_development_daily=targeted.get(segment, 0.0),
        )
        for segment in segments
        for plan in "ABC"
    )
    return WeeklyEvidencePacket(
        day=observation.day,
        window_start_day_exclusive=observation.day - 7,
        window_end_day_inclusive=observation.day,
        acquisition=tuple(AcquisitionEvidence.model_validate(row) for row in acquisition_rows),
        cohorts=tuple(CohortEvidence.model_validate(row) for row in cohort_rows),
        quality=quality,
        ledger=tuple(LedgerEvidence.model_validate(row) for row in ledger_rows),
        configuration=ConfigurationEvidence(
            prices=prices,
            model_tiers=tiers,
            usage_quotas={plan: _number(config, f"quota_{plan}") for plan in "ABC"},
            daily_channel_spend={
                channel: _number(config, column)
                for channel, column in _CHANNEL_COLUMNS.items()
            },
            daily_operations_spend=_number(config, "spend_operations"),
            daily_development_spend=_number(config, "spend_development"),
            capacity_tier=int(_number(config, "capacity_tier")),
            lead_promotion_json=str(config.get("lead_promotion_json") or "{}"),
            recurring_promotion_json=str(config.get("recurring_promotion_json") or "{}"),
            ads_strength_json=str(config.get("ads_strength_json") or "{}"),
            targeted_ads_json=str(config.get("targeted_ads_json") or "{}"),
            targeted_development_json=str(
                config.get("targeted_development_json") or "{}"
            ),
            targeted_ops_json=str(config.get("targeted_ops_json") or "{}"),
        ),
    )

