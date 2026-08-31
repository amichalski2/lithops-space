from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any
from uuid import UUID

from lithops.benchmark.ceobench.action_mapper import build_action_code
from lithops.benchmark.ceobench.cli import CeobenchCli, parse_json_output
from lithops.benchmark.ceobench.evidence import collect_weekly_evidence
from lithops.benchmark.ceobench.market_signals import (
    MARKET_SIGNAL_PARSER_VERSION,
    announced_quality_releases,
    quality_bar_shift,
    unquantified_release_count,
)
from lithops.domain.errors import BenchmarkContractError
from lithops.domain.experiment_contracts import OBSERVATION_CONTRACT_VERSION
from lithops.domain.models import (
    ActionCommand,
    ActionReceipt,
    CashForecasts,
    ObservationSnapshot,
    ReceiptStatus,
)
from lithops.domain.public_instruments import MODEL_TIER_QUALITY_MULTIPLIER
from lithops.infrastructure.security.sql_guard import validate_readonly_sql

# The environment's public name for the player's own product (it is printed
# throughout the published API docs); used only to exclude our own launches
# from competitor-release counting.
PLAYER_BRAND = "NovaMind"

COMPETITOR_SIGNAL_QUERY = """
SELECT day, content
FROM social_media_posts
WHERE content LIKE '%quality boost%'
   OR content LIKE '%competitor%'
   OR content LIKE '%releas%'
   OR content LIKE '%launch%'
   OR content LIKE '%updat%'
   OR content LIKE '%upgrad%'
   OR content LIKE '%overhaul%'
   OR content LIKE '%breakthrough%'
ORDER BY day
""".strip()

MARKET_FEED_QUERY = """
SELECT day, kind, message FROM (
    SELECT day, 'social_post' AS kind, substr(content, 1, 240) AS message,
           post_id AS ordering
    FROM social_media_posts
    WHERE day >= (SELECT COALESCE(MAX(day), 0) - 13 FROM social_media_posts)
    ORDER BY day DESC, post_id DESC
    LIMIT 25
)
UNION ALL
SELECT day, kind, message FROM (
    SELECT day, 'notification' AS kind,
           substr(COALESCE(type, '') || ': ' || COALESCE(message, ''), 1, 240)
               AS message,
           notification_id AS ordering
    FROM notifications
    WHERE day >= (SELECT COALESCE(MAX(day), 0) - 27 FROM notifications)
    ORDER BY day DESC, notification_id DESC
    LIMIT 10
)
UNION ALL
SELECT day, kind, message FROM (
    SELECT day, 'macro' AS kind, substr(description, 1, 240) AS message,
           day AS ordering
    FROM macroeconomic_conditions
    ORDER BY day DESC
    LIMIT 1
)
ORDER BY day DESC
""".strip()

OBSERVATION_QUERY = """
WITH current_day AS (
    SELECT COALESCE(MAX(day), 0) AS day
    FROM service_day
),
subscription_metrics AS (
    SELECT
        COALESCE(SUM(CASE WHEN status = 'subscribed' THEN 1 ELSE 0 END), 0)
            AS active_customers,
        COALESCE(SUM(CASE WHEN status = 'subscribed' THEN seat_count ELSE 0 END), 0)
            AS active_seats,
        COALESCE(SUM(
            CASE WHEN status = 'subscribed' THEN effective_price * seat_count ELSE 0 END
        ) * 7.0 / 30.0, 0) AS weekly_revenue,
        COALESCE(SUM(CASE
            WHEN start_day > (SELECT day FROM current_day) - 7 THEN 1 ELSE 0
        END), 0) AS weekly_leads,
        COALESCE(SUM(CASE
            WHEN status = 'subscribed'
             AND start_day > (SELECT day FROM current_day) - 7 THEN 1 ELSE 0
        END), 0) AS weekly_conversions,
        COALESCE(SUM(CASE
            WHEN status = 'lost'
             AND start_day > (SELECT day FROM current_day) - 7 THEN 1 ELSE 0
        END), 0) AS weekly_lost_leads,
        COALESCE(SUM(CASE WHEN status = 'lead' THEN 1 ELSE 0 END), 0) AS pending_leads,
        COALESCE(COUNT(*), 0) AS total_leads,
        COALESCE(SUM(CASE
            WHEN status IN ('subscribed', 'cancelled') THEN 1 ELSE 0
        END), 0) AS total_conversions,
        COALESCE(SUM(CASE WHEN status = 'lost' THEN 1 ELSE 0 END), 0)
            AS total_lost_leads,
        COALESCE(SUM(
            CASE
                WHEN status = 'cancelled'
                 AND end_day > (SELECT day FROM current_day) - 28 THEN 1
                ELSE 0
            END
        ), 0) AS cancellations_28d,
        COALESCE(SUM(
            CASE WHEN status = 'subscribed' THEN effective_price * seat_count ELSE 0 END
        ) / NULLIF(SUM(
            CASE WHEN status = 'subscribed' THEN 1 ELSE 0 END
        ), 0) * 7.0 / 30.0, 0) AS price_per_customer_weekly
    FROM subscriptions
),
latest_open_enterprise_turns AS (
    SELECT turn.customer_id, turn.sender, turn.day, turn.seat_count
    FROM enterprise_turns AS turn
    JOIN (
        SELECT customer_id, MAX(message_id) AS latest_message_id
        FROM enterprise_turns
        WHERE COALESCE(closed, 0) = 0
        GROUP BY customer_id
    ) AS latest ON latest.latest_message_id = turn.message_id
    WHERE COALESCE(turn.closed, 0) = 0
),
enterprise_revenue AS (
    -- Seat contracts are ordinary subscriptions held by an enterprise buyer, so
    -- the split is by who holds them, not by how they are billed.
    SELECT COALESCE(SUM(
        CASE WHEN subscription.status = 'subscribed'
            THEN subscription.effective_price * subscription.seat_count
            ELSE 0 END
    ) * 7.0 / 30.0, 0) AS enterprise_revenue_weekly
    FROM subscriptions AS subscription
    JOIN customers AS customer ON customer.customer_id = subscription.customer_id
    WHERE customer.customer_type != 'small'
),
enterprise_metrics AS (
    SELECT
        COALESCE(SUM(CASE WHEN sender = 'customer' THEN 1 ELSE 0 END), 0)
            AS open_enterprise_threads,
        MIN(CASE WHEN sender = 'customer' THEN day END)
            AS oldest_open_enterprise_thread_day,
        COALESCE(GROUP_CONCAT(
            CASE WHEN sender = 'customer'
                THEN customer_id || ':' || MAX(1, seat_count) || ':' || day
            END,
            ','
        ), '') AS enterprise_inbox
    FROM latest_open_enterprise_turns
),
latest_config AS (
    SELECT
        COALESCE((SELECT price_A FROM config_history ORDER BY day DESC LIMIT 1), 0)
            AS price_A,
        COALESCE((SELECT price_B FROM config_history ORDER BY day DESC LIMIT 1), 0)
            AS price_B,
        COALESCE((SELECT price_C FROM config_history ORDER BY day DESC LIMIT 1), 0)
            AS price_C,
        COALESCE((SELECT ad_spend_social_media FROM config_history ORDER BY day DESC LIMIT 1), 0)
            AS ad_spend_social_media,
        COALESCE((SELECT ad_spend_search_ads FROM config_history ORDER BY day DESC LIMIT 1), 0)
            AS ad_spend_search_ads,
        COALESCE((SELECT ad_spend_linkedin FROM config_history ORDER BY day DESC LIMIT 1), 0)
            AS ad_spend_linkedin,
        COALESCE((
            SELECT ad_spend_content_marketing FROM config_history ORDER BY day DESC LIMIT 1
        ), 0)
            AS ad_spend_content_marketing,
        COALESCE((
            SELECT ad_spend_referral_program FROM config_history ORDER BY day DESC LIMIT 1
        ), 0)
            AS ad_spend_referral_program,
        COALESCE((SELECT spend_advertising FROM config_history ORDER BY day DESC LIMIT 1), 0)
            AS spend_advertising,
        COALESCE((SELECT spend_operations FROM config_history ORDER BY day DESC LIMIT 1), 500)
            AS spend_operations,
        COALESCE((SELECT spend_development FROM config_history ORDER BY day DESC LIMIT 1), 250)
            AS spend_development,
        COALESCE((SELECT tier_A FROM config_history ORDER BY day DESC LIMIT 1), 1) AS tier_A,
        COALESCE((SELECT tier_B FROM config_history ORDER BY day DESC LIMIT 1), 2) AS tier_B,
        COALESCE((SELECT tier_C FROM config_history ORDER BY day DESC LIMIT 1), 3) AS tier_C,
        COALESCE((SELECT quota_A FROM config_history ORDER BY day DESC LIMIT 1), 0)
            AS quota_A,
        COALESCE((SELECT quota_B FROM config_history ORDER BY day DESC LIMIT 1), 0)
            AS quota_B,
        COALESCE((SELECT quota_C FROM config_history ORDER BY day DESC LIMIT 1), 0)
            AS quota_C,
        COALESCE((SELECT capacity_tier FROM config_history ORDER BY day DESC LIMIT 1), 0)
            AS capacity_tier
),
recent_usage AS (
    -- Usage is recorded after the configured allowance clamps it, so this is a
    -- censored observation of demand, never demand itself.
    SELECT COALESCE(AVG(usage_units), 0.0) AS daily_usage_per_customer
    FROM daily_usage
    WHERE day > (SELECT day FROM current_day) - 7
),
latest_lead_promotion AS (
    SELECT COALESCE((
        SELECT CAST(json_extract(settings_json, '$.global') AS REAL)
        FROM config_overrides
        WHERE tool_name = 'set_lead_promotion'
        ORDER BY day DESC, id DESC
        LIMIT 1
    ), 0.0) AS global_monthly
),
latest_recurring_promotion AS (
    SELECT COALESCE((
        SELECT CAST(json_extract(settings_json, '$.global') AS REAL)
        FROM config_overrides
        WHERE tool_name = 'set_promotion'
          AND json_extract(settings_json, '$.global') IS NOT NULL
        ORDER BY day DESC, id DESC
        LIMIT 1
    ), 0.0) AS global_monthly
),
latest_ads_strength AS (
    SELECT COALESCE((
        SELECT CAST(json_extract(settings_json, '$.global') AS REAL)
        FROM config_overrides
        WHERE tool_name = 'set_ads_strength'
          AND json_extract(settings_json, '$.global') IS NOT NULL
        ORDER BY day DESC, id DESC
        LIMIT 1
    ), 0.0) AS global_strength
),
latest_targeted_ops AS (
    SELECT COALESCE((
        SELECT settings_json
        FROM config_overrides
        WHERE tool_name = 'set_targeted_ops_spend'
        ORDER BY day DESC, id DESC
        LIMIT 1
    ), '{}') AS settings_json
),
latest_targeted_development AS (
    SELECT COALESCE((
        SELECT settings_json
        FROM config_overrides
        WHERE tool_name = 'set_targeted_dev_spend'
        ORDER BY day DESC, id DESC
        LIMIT 1
    ), '{}') AS settings_json
),
latest_targeted_ads AS (
    SELECT COALESCE((
        SELECT settings_json
        FROM config_overrides
        WHERE tool_name = 'set_targeted_ad_spend'
        ORDER BY day DESC, id DESC
        LIMIT 1
    ), '{}') AS settings_json
),
targeted_ads_by_channel AS (
    -- Acquisition spend set through the targeted tool lives in the override
    -- table, not in the config columns. Reading only the columns reports zero
    -- spend while money is actually being spent.
    SELECT channel.key AS channel, SUM(CAST(grp.value AS REAL)) AS daily_spend
    FROM latest_targeted_ads,
         json_each(
             CASE
                 WHEN json_type(latest_targeted_ads.settings_json, '$.targeted_spend')
                     = 'object'
                 THEN json_extract(latest_targeted_ads.settings_json, '$.targeted_spend')
                 ELSE latest_targeted_ads.settings_json
             END
         ) AS channel,
         json_each(channel.value) AS grp
    WHERE json_type(channel.value) = 'object'
    GROUP BY channel.key
),
latest_service AS (
    SELECT COALESCE(
        (SELECT capacity_units FROM service_day ORDER BY day DESC LIMIT 1),
        1000
    ) AS capacity_units
),
issue_metrics AS (
    SELECT COALESCE(SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END), 0) AS open_issues
    FROM issues
),
known_groups AS (
    SELECT GROUP_CONCAT(group_id) AS known_segments
    FROM group_info_levels
    WHERE info_level > 0
),
recent_cost_metrics AS (
    SELECT
        COALESCE(-SUM(CASE
            WHEN category = 'compute'
             AND day > (SELECT day FROM current_day) - 7
            THEN amount ELSE 0 END
        ), 0) AS weekly_compute_cost,
        CASE
            WHEN (SELECT day FROM current_day) = 0 THEN 595.0
            ELSE COALESCE(-SUM(CASE
                WHEN category = 'capacity'
                 AND day > (SELECT day FROM current_day) - 7
                THEN amount ELSE 0 END
            ), 0)
        END AS capacity_spend_weekly
    FROM ledger
),
research_quality AS (
    -- Completed R&D applies a quality boost the benchmark announces publicly
    -- (list_research_projects, completion notifications) - summing what those
    -- announcements state keeps the proxy moving when spend is a step, not a
    -- daily flow.
    SELECT
        COALESCE(SUM(CASE WHEN status = 'completed'
                          THEN COALESCE(quality_boost_applied, 0.0)
                          ELSE 0.0 END), 0.0)
            AS research_completed_quality_boost_total,
        COALESCE(SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END), 0)
            AS research_in_progress_count
    FROM research_projects
),
quality_metrics AS (
    -- quality-proxy-v2: coefficients are the environment's published dev-spend
    -- documentation (global 0.006, targeted 0.030 per public docs), applied to
    -- the public ledger. Targeted rows are identified by the ledger's own note
    -- text. A proxy with named provenance, not a measurement.
    SELECT MIN(1.0, 0.20
        + COALESCE(SUM(
            CASE
                WHEN category = 'development' AND note = 'Targeted dev spend'
                THEN 0.030 * ln(1.0 + MAX(0.0, -amount) / 5000.0)
                WHEN category = 'development'
                THEN 0.006 * ln(1.0 + MAX(0.0, -amount) / 5000.0)
                ELSE 0.0
            END
        ), 0.0)
        + (SELECT research_completed_quality_boost_total FROM research_quality)
    ) AS product_quality
    FROM ledger
)
SELECT
    (SELECT day FROM current_day) AS day,
    (SELECT COALESCE(SUM(amount), 0) FROM ledger) AS cash,
    subscription_metrics.active_customers,
    subscription_metrics.active_seats,
    subscription_metrics.weekly_revenue,
    subscription_metrics.weekly_conversions AS weekly_acquisition,
    subscription_metrics.weekly_leads,
    subscription_metrics.weekly_conversions,
    subscription_metrics.weekly_lost_leads,
    subscription_metrics.total_leads,
    subscription_metrics.total_conversions,
    subscription_metrics.total_lost_leads,
    subscription_metrics.pending_leads,
    CASE
        WHEN subscription_metrics.weekly_leads = 0 THEN 0
        ELSE subscription_metrics.weekly_conversions * 1.0 /
             subscription_metrics.weekly_leads
    END AS lead_conversion_rate,
    enterprise_metrics.open_enterprise_threads,
    enterprise_metrics.enterprise_inbox,
    enterprise_revenue.enterprise_revenue_weekly,
    CASE
        WHEN subscription_metrics.active_customers + subscription_metrics.cancellations_28d = 0
        THEN 0
        ELSE subscription_metrics.cancellations_28d * 1.0 /
             (subscription_metrics.active_customers + subscription_metrics.cancellations_28d)
    END AS churn_rate,
    subscription_metrics.price_per_customer_weekly,
    CASE
        WHEN subscription_metrics.active_customers = 0 THEN 0
        ELSE recent_cost_metrics.weekly_compute_cost /
             subscription_metrics.active_customers
    END AS operating_cost_per_customer_weekly,
    latest_config.price_A AS price_a,
    latest_config.price_B AS price_b,
    latest_config.price_C AS price_c,
    -- Raw configured values: never substituted by a planning estimate, so an
    -- unset control stays visibly unset to the Executive.
    latest_config.price_A AS configured_price_a,
    latest_config.price_B AS configured_price_b,
    latest_config.price_C AS configured_price_c,
    latest_lead_promotion.global_monthly AS lead_promotion_monthly,
    latest_recurring_promotion.global_monthly AS recurring_promotion_monthly,
    latest_ads_strength.global_strength AS ads_strength,
    COALESCE((
        SELECT SUM(CAST(value AS REAL)) * 7.0
        FROM json_each(
            CASE
                WHEN json_type(latest_targeted_ops.settings_json, '$.targeted_spend')
                    = 'object'
                THEN json_extract(latest_targeted_ops.settings_json, '$.targeted_spend')
                ELSE latest_targeted_ops.settings_json
            END
        )
    ), 0.0) AS targeted_ops_spend,
    latest_targeted_development.settings_json
        AS targeted_development_allocations_json,
    latest_targeted_ads.settings_json AS targeted_ad_allocations_json,
    COALESCE((
        SELECT SUM(CAST(value AS REAL)) * 7.0
        FROM json_each(
            CASE
                WHEN json_type(
                    latest_targeted_development.settings_json,
                    '$.targeted_spend'
                ) = 'object'
                THEN json_extract(
                    latest_targeted_development.settings_json,
                    '$.targeted_spend'
                )
                ELSE latest_targeted_development.settings_json
            END
        )
    ), 0.0) AS targeted_development_spend,
    COALESCE(MAX(
        latest_config.spend_advertising,
        latest_config.ad_spend_social_media
        + latest_config.ad_spend_search_ads
        + latest_config.ad_spend_linkedin
        + latest_config.ad_spend_content_marketing
        + latest_config.ad_spend_referral_program,
        COALESCE((SELECT SUM(daily_spend) FROM targeted_ads_by_channel), 0)
    ) * 7.0, 0) AS marketing_spend,
    MAX(
        COALESCE(latest_config.ad_spend_social_media, 0),
        COALESCE((SELECT daily_spend FROM targeted_ads_by_channel
                  WHERE channel = 'social_media'), 0)
    ) * 7.0 AS marketing_spend_social_media_weekly,
    MAX(
        COALESCE(latest_config.ad_spend_search_ads, 0),
        COALESCE((SELECT daily_spend FROM targeted_ads_by_channel
                  WHERE channel = 'search_ads'), 0)
    ) * 7.0 AS marketing_spend_search_ads_weekly,
    MAX(
        COALESCE(latest_config.ad_spend_linkedin, 0),
        COALESCE((SELECT daily_spend FROM targeted_ads_by_channel
                  WHERE channel = 'linkedin'), 0)
    ) * 7.0 AS marketing_spend_linkedin_weekly,
    MAX(
        COALESCE(latest_config.ad_spend_content_marketing, 0),
        COALESCE((SELECT daily_spend FROM targeted_ads_by_channel
                  WHERE channel = 'content_marketing'), 0)
    ) * 7.0 AS marketing_spend_content_marketing_weekly,
    MAX(
        COALESCE(latest_config.ad_spend_referral_program, 0),
        COALESCE((SELECT daily_spend FROM targeted_ads_by_channel
                  WHERE channel = 'referral_program'), 0)
    ) * 7.0 AS marketing_spend_referral_program_weekly,
    COALESCE(latest_config.spend_operations * 7.0, 0) AS operations_spend,
    COALESCE(latest_config.spend_development * 7.0, 0) AS development_spend,
    recent_cost_metrics.capacity_spend_weekly,
    quality_metrics.product_quality,
    research_quality.research_completed_quality_boost_total,
    research_quality.research_in_progress_count,
    latest_config.tier_A AS model_tier_a,
    latest_config.tier_B AS model_tier_b,
    latest_config.tier_C AS model_tier_c,
    latest_config.quota_A AS usage_quota_a,
    latest_config.quota_B AS usage_quota_b,
    latest_config.quota_C AS usage_quota_c,
    latest_config.capacity_tier,
    recent_usage.daily_usage_per_customer,
    COALESCE(latest_service.capacity_units, 1000) AS capacity,
    CASE
        WHEN subscription_metrics.active_customers = 0 THEN 0.5
        ELSE MAX(0.0, 1.0 - issue_metrics.open_issues * 1.0 /
             subscription_metrics.active_customers)
    END AS reputation,
    issue_metrics.open_issues,
    COALESCE(
        (SELECT day FROM current_day)
            - enterprise_metrics.oldest_open_enterprise_thread_day,
        0
    ) AS enterprise_oldest_thread_age_days,
    COALESCE(known_groups.known_segments, 'S1') AS known_segments
FROM subscription_metrics, enterprise_metrics, enterprise_revenue, latest_config,
     latest_lead_promotion,
     latest_recurring_promotion, latest_ads_strength, latest_targeted_ops,
     latest_targeted_development, latest_targeted_ads, latest_service, issue_metrics,
     known_groups, recent_cost_metrics, quality_metrics, research_quality,
     recent_usage
""".strip()


class CeobenchAdapter:
    """Implements BenchmarkPort through the documented public CEO-Bench CLI."""

    def __init__(self, *, cli: CeobenchCli, seed: int = 42) -> None:
        self.cli = cli
        self.seed = seed
        self._session_by_run: dict[UUID, str] = {}
        self._receipts: dict[tuple[UUID, str], ActionReceipt] = {}

    async def create_session(self, run_id: UUID, *, days: int) -> str:
        existing = self._session_by_run.get(run_id)
        if existing is not None:
            return existing
        if self._session_by_run:
            raise BenchmarkContractError(
                "this CEO-Bench adapter already owns its single allowed session"
            )

        response = await self.cli.new_session(days=days, seed=self.seed)
        session_id = str(response["session_id"])
        self._session_by_run[run_id] = session_id
        return session_id

    async def _market_feed(self, session_id: str) -> str | None:
        """Recent externally authored signals: customer posts, competitor
        announcements, inbox notifications, and the macro climate.

        This is perception, not judgement: the feed is handed to the Executive
        verbatim as untrusted evidence about the world outside the company. A
        benchmark build without these tables degrades to an absent feed rather
        than a failed observation.
        """

        try:
            rows = await self.query_readonly(session_id, MARKET_FEED_QUERY)
        except BenchmarkContractError:
            return None
        feed = [
            {
                "day": int(row["day"]),
                "kind": str(row["kind"]),
                "message": str(row["message"] or ""),
            }
            for row in rows
            if row.get("day") is not None
            and row.get("kind") is not None
            and "message" in row
        ]
        return json.dumps(feed, ensure_ascii=False, separators=(",", ":"))

    async def _quality_bar_signals(
        self, session_id: str, *, day: int
    ) -> dict[str, float] | None:
        """How far competitor releases have moved the quality bar so far.

        A lower bound assembled from public announcements, not a measurement:
        only releases someone quantified are counted. It exists so the bar's
        rate can be compared against the company's own rate of improvement.
        """

        try:
            rows = await self.query_readonly(session_id, COMPETITOR_SIGNAL_QUERY)
        except BenchmarkContractError:
            return None
        releases = announced_quality_releases(rows)
        return {
            "competitor_quality_releases": float(len(releases)),
            "competitor_quality_bar_shift": quality_bar_shift(releases),
            "competitor_quality_bar_shift_28d": quality_bar_shift(
                releases, since_day=max(0, day - 28)
            ),
            # Releases the market discussed without a number: each one moved the
            # bar by an unknown amount, so a zero shift beside a nonzero count
            # is "drift unmeasured", never "no drift".
            "competitor_quality_releases_unquantified": float(
                unquantified_release_count(rows, own_brand=PLAYER_BRAND)
            ),
        }

    async def _research_catalog(self, session_id: str) -> str | None:
        """The environment's own R&D price list, from its read-only listing.

        Observed facts, not priors: the cost is exact at listing time and the
        quality/duration figures are the means the listing itself publishes.
        Normalized to the compact form the planner consumes; an absent or
        unreadable listing degrades to ``None`` rather than a failed
        observation, and ``None`` means unread — never an empty catalog.
        """

        code = "\n".join(
            [
                "import json",
                "from novamind_api import research",
                "print(json.dumps(research.list_research_projects(), default=str))",
            ]
        )
        try:
            output = await self.cli.python_c(session_id, code)
            payload = parse_json_output(output)
        except Exception:
            # Perception degrades to absence; judgement stays with the planner.
            return None
        tiers = payload.get("tiers") if isinstance(payload, dict) else None
        entries: list[dict[str, float | int]] = []
        for item in tiers or ():
            if not isinstance(item, dict):
                continue
            try:
                entry: dict[str, float | int] = {
                    "tier": int(item["tier"]),
                    "cost": float(item["cost"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
            mean_days = item.get("mean_days")
            if isinstance(mean_days, int | float):
                entry["mean_weeks"] = max(0, round(float(mean_days) / 7.0))
            boost = item.get("mean_quality_boost")
            if isinstance(boost, int | float) and float(boost) >= 0.0:
                entry["mean_quality_boost"] = float(boost)
            in_progress = item.get("in_progress")
            if isinstance(in_progress, int | float) and int(in_progress) > 0:
                entry["in_progress"] = int(in_progress)
            entries.append(entry)
        if not entries:
            return None
        return json.dumps(entries, separators=(",", ":"))

    async def observe_status(self, session_id: str) -> ObservationSnapshot:
        rows = await self.query_readonly(session_id, OBSERVATION_QUERY)
        market_feed = await self._market_feed(session_id)
        research_catalog = await self._research_catalog(session_id)
        if len(rows) != 1:
            raise BenchmarkContractError(
                f"CEO-Bench observation query returned {len(rows)} rows instead of one"
            )
        row = rows[0]
        try:
            day = int(row["day"])
            cash = float(row["cash"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BenchmarkContractError(
                "CEO-Bench observation is missing numeric day or cash"
            ) from exc
        numeric_metrics = (
            "active_customers",
            "active_seats",
            "weekly_revenue",
            "weekly_acquisition",
            "weekly_leads",
            "weekly_conversions",
            "weekly_lost_leads",
            "total_leads",
            "total_conversions",
            "total_lost_leads",
            "pending_leads",
            "lead_conversion_rate",
            "open_enterprise_threads",
            "enterprise_revenue_weekly",
            "churn_rate",
            "price_per_customer_weekly",
            "operating_cost_per_customer_weekly",
            "price_a",
            "price_b",
            "price_c",
            "lead_promotion_monthly",
            "marketing_spend",
            "marketing_spend_social_media_weekly",
            "marketing_spend_search_ads_weekly",
            "marketing_spend_linkedin_weekly",
            "marketing_spend_content_marketing_weekly",
            "marketing_spend_referral_program_weekly",
            "operations_spend",
            "development_spend",
            "targeted_development_spend",
            "capacity_spend_weekly",
            "product_quality",
            "research_completed_quality_boost_total",
            "research_in_progress_count",
            "model_tier_a",
            "model_tier_b",
            "model_tier_c",
            "usage_quota_a",
            "usage_quota_b",
            "usage_quota_c",
            "capacity_tier",
            "daily_usage_per_customer",
            "recurring_promotion_monthly",
            "ads_strength",
            "targeted_ops_spend",
            "configured_price_a",
            "configured_price_b",
            "configured_price_c",
            "capacity",
            "reputation",
            "open_issues",
            "enterprise_oldest_thread_age_days",
        )
        initial_estimates = {
            "active_customers": 0.0,
            "active_seats": 0.0,
            "weekly_revenue": 0.0,
            "weekly_acquisition": 0.0,
            "weekly_leads": 0.0,
            "weekly_conversions": 0.0,
            "weekly_lost_leads": 0.0,
            "total_leads": 0.0,
            "total_conversions": 0.0,
            "total_lost_leads": 0.0,
            "pending_leads": 0.0,
            "lead_conversion_rate": 0.0,
            "open_enterprise_threads": 0.0,
            "enterprise_revenue_weekly": 0.0,
            "churn_rate": 0.0,
            "price_per_customer_weekly": 10.0,
            "operating_cost_per_customer_weekly": 0.0,
            "price_a": 25.0,
            "price_b": 69.0,
            "price_c": 179.0,
            "lead_promotion_monthly": 0.0,
            "marketing_spend": 0.0,
            "marketing_spend_social_media_weekly": 0.0,
            "marketing_spend_search_ads_weekly": 0.0,
            "marketing_spend_linkedin_weekly": 0.0,
            "marketing_spend_content_marketing_weekly": 0.0,
            "marketing_spend_referral_program_weekly": 0.0,
            "operations_spend": 3_500.0,
            "development_spend": 1_750.0,
            "targeted_development_spend": 0.0,
            "capacity_spend_weekly": 595.0,
            "product_quality": 0.2,
            "research_completed_quality_boost_total": 0.0,
            "research_in_progress_count": 0.0,
            "model_tier_a": 1.0,
            "model_tier_b": 1.0,
            "model_tier_c": 1.0,
            "usage_quota_a": 0.0,
            "usage_quota_b": 0.0,
            "usage_quota_c": 0.0,
            "capacity_tier": 0.0,
            "daily_usage_per_customer": 0.0,
            "recurring_promotion_monthly": 0.0,
            "ads_strength": 0.0,
            "targeted_ops_spend": 0.0,
            "configured_price_a": 0.0,
            "configured_price_b": 0.0,
            "configured_price_c": 0.0,
            "capacity": 1_000.0,
            "reputation": 0.5,
            "open_issues": 0.0,
            "enterprise_oldest_thread_age_days": 0.0,
        }
        metrics: dict[str, float | int | str | bool | None] = {}
        invalid_metrics: list[str] = []
        estimated_metrics: list[str] = []
        # Metrics introduced by a later observation contract are estimated rather
        # than fatal when a benchmark build does not report them yet.
        tolerated_when_absent = {
            "reputation",
            "open_issues",
            "enterprise_oldest_thread_age_days",
            "research_completed_quality_boost_total",
            "research_in_progress_count",
            "enterprise_revenue_weekly",
            "usage_quota_a",
            "usage_quota_b",
            "usage_quota_c",
            "capacity_tier",
            "daily_usage_per_customer",
            "recurring_promotion_monthly",
            "ads_strength",
            "targeted_ops_spend",
            "configured_price_a",
            "configured_price_b",
            "configured_price_c",
        }
        for name in numeric_metrics:
            if name not in row:
                if name in tolerated_when_absent or name.startswith("marketing_spend_"):
                    metrics[name] = initial_estimates[name]
                    estimated_metrics.append(name)
                else:
                    invalid_metrics.append(name)
                continue
            if row[name] is None:
                metrics[name] = initial_estimates[name]
                estimated_metrics.append(name)
                continue
            try:
                metrics[name] = float(row[name])
            except (TypeError, ValueError):
                if name in tolerated_when_absent:
                    metrics[name] = initial_estimates[name]
                    estimated_metrics.append(name)
                else:
                    invalid_metrics.append(name)
        if day == 0 and not metrics["active_customers"]:
            for name in ("price_a", "price_b", "price_c"):
                if not metrics[name]:
                    metrics[name] = initial_estimates[name]
                    estimated_metrics.append(name)
            if not metrics["price_per_customer_weekly"]:
                metrics["price_per_customer_weekly"] = (
                    sum(initial_estimates[name] for name in ("price_a", "price_b", "price_c"))
                    / 3.0
                    * 7.0
                    / 30.0
                )
                estimated_metrics.append("price_per_customer_weekly")
        if invalid_metrics:
            raise BenchmarkContractError(
                "CEO-Bench observation is missing normalized business metrics: "
                + ", ".join(invalid_metrics)
            )
        # Delivered quality per plan: the number a customer actually judges,
        # composed from the base proxy and the environment's own published
        # tier-multiplier table (get_cost_info / set_model_tiers impact text).
        # The base is a proxy, so these inherit its confidence.
        base_quality = float(metrics["product_quality"])
        for plan in ("a", "b", "c"):
            raw_tier = metrics[f"model_tier_{plan}"]
            tier = int(raw_tier) if isinstance(raw_tier, int | float) else 1
            tier = min(
                max(tier, min(MODEL_TIER_QUALITY_MULTIPLIER)),
                max(MODEL_TIER_QUALITY_MULTIPLIER),
            )
            metrics[f"delivered_quality_plan_{plan}"] = (
                base_quality * MODEL_TIER_QUALITY_MULTIPLIER[tier]
            )
        quality_bar = await self._quality_bar_signals(session_id, day=day)
        if quality_bar is None:
            quality_bar = {
                "competitor_quality_releases": 0.0,
                "competitor_quality_bar_shift": 0.0,
                "competitor_quality_bar_shift_28d": 0.0,
                "competitor_quality_releases_unquantified": 0.0,
            }
            estimated_metrics.append("competitor_quality_bar_shift")
        metrics.update(quality_bar)
        metrics.update(
            {
                "known_segments": str(row.get("known_segments") or "S1"),
                "enterprise_inbox": str(row.get("enterprise_inbox") or ""),
                "targeted_development_allocations_json": str(
                    row.get("targeted_development_allocations_json") or "{}"
                ),
                "targeted_ad_allocations_json": str(
                    row.get("targeted_ad_allocations_json") or "{}"
                ),
                "market_feed": market_feed if market_feed is not None else "[]",
                "research_catalog_json": (
                    research_catalog if research_catalog is not None else "[]"
                ),
                "estimated_metrics": ",".join(
                    estimated_metrics
                    + (["market_feed"] if market_feed is None else [])
                    + (["research_catalog_json"] if research_catalog is None else [])
                ),
                "derived_metrics": (
                    "product_quality,delivered_quality_plan_a,"
                    "delivered_quality_plan_b,delivered_quality_plan_c"
                ),
                "product_quality_provenance": "derived:ledger:quality-proxy-v2",
                "research_catalog_provenance": (
                    "observed:list_research_projects:published-means"
                ),
                "delivered_quality_provenance": (
                    "derived:product_quality*public-tier-multiplier"
                ),
                "competitor_quality_bar_provenance": (
                    "lower_bound:announced_releases:" + MARKET_SIGNAL_PARSER_VERSION
                ),
                "product_quality_confidence": "proxy",
                "source": "ceobench_public_cli",
                "source_query_version": OBSERVATION_CONTRACT_VERSION,
            }
        )
        return ObservationSnapshot(
            day=day,
            cash=cash,
            metrics=metrics,
        )

    async def query_readonly(
        self, session_id: str, sql: str
    ) -> list[dict[str, Any]]:
        statement = validate_readonly_sql(sql)
        payload = await self.cli.query(session_id, statement)
        rows: Any = (
            payload.get("rows", payload.get("data"))
            if isinstance(payload, Mapping)
            else payload
        )
        if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
            raise BenchmarkContractError("CEO-Bench query response has no row list")
        return [dict(row) for row in rows]

    async def collect_weekly_evidence(
        self, session_id: str, observation: ObservationSnapshot
    ):
        async def query(sql: str) -> list[dict[str, Any]]:
            return await self.query_readonly(session_id, sql)

        return await collect_weekly_evidence(query, observation)

    async def execute_action(
        self,
        session_id: str,
        *,
        run_id: UUID,
        decision_id: UUID,
        command: ActionCommand,
    ) -> ActionReceipt:
        key = (run_id, command.idempotency_key)
        existing = self._receipts.get(key)
        if existing is not None:
            if existing.tool != command.tool:
                raise BenchmarkContractError(
                    "idempotency key was reused for a different CEO-Bench tool"
                )
            return existing.model_copy(update={"status": ReceiptStatus.REPLAYED})

        code = build_action_code(command)
        output = await self.cli.python_c(session_id, code)
        receipt = ActionReceipt(
            run_id=run_id,
            decision_id=decision_id,
            idempotency_key=command.idempotency_key,
            tool=command.tool,
            semantic_command_hash=command.semantic_hash,
            status=ReceiptStatus.EXECUTED,
            external_reference=f"{session_id}:{command.idempotency_key}",
            result={"stdout": output.strip()[-8_000:]},
        )
        self._receipts[key] = receipt
        return receipt

    async def advance_week(
        self,
        session_id: str,
        *,
        rationale: str,
        forecasts: CashForecasts,
    ) -> ObservationSnapshot:
        await self.cli.next_week(
            session_id,
            rationale=rationale,
            forecasts=forecasts,
        )
        return await self.observe_status(session_id)
