from __future__ import annotations

import base64
import json
import sqlite3
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import pytest
from lithops.application.step_run import RunManager, StaticDecisionEngine
from lithops.benchmark.ceobench.action_mapper import build_action_code
from lithops.benchmark.ceobench.adapter import OBSERVATION_QUERY, CeobenchAdapter
from lithops.benchmark.ceobench.cli import (
    CeobenchCli,
    CommandResult,
    parse_json_output,
)
from lithops.domain.errors import BenchmarkContractError
from lithops.domain.models import (
    ActionCommand,
    CashForecast,
    CashForecasts,
    ReceiptStatus,
)
from lithops.infrastructure.persistence.repositories import InMemoryRunRepository


def observation(day: int, cash: float) -> str:
    return json.dumps(
        {
            "rows": [
                {
                    "day": day,
                    "cash": cash,
                    "active_customers": 100,
                    "active_seats": 120,
                    "weekly_revenue": 5000,
                    "weekly_acquisition": 5,
                    "weekly_leads": 12,
                    "weekly_conversions": 5,
                    "weekly_lost_leads": 6,
                    "total_leads": 120,
                    "total_conversions": 50,
                    "total_lost_leads": 69,
                    "pending_leads": 1,
                    "lead_conversion_rate": 5 / 12,
                    "open_enterprise_threads": 2,
                    "enterprise_revenue_weekly": 0,
                    "enterprise_inbox": "",
                    "churn_rate": 0.03,
                    "price_per_customer_weekly": 11.5,
                    "operating_cost_per_customer_weekly": 0.25,
                    "price_a": 25,
                    "price_b": 69,
                    "price_c": 179,
                    "lead_promotion_monthly": 0,
                    "marketing_spend": 3500,
                    "marketing_spend_social_media_weekly": 0,
                    "marketing_spend_search_ads_weekly": 3500,
                    "marketing_spend_linkedin_weekly": 0,
                    "marketing_spend_content_marketing_weekly": 0,
                    "marketing_spend_referral_program_weekly": 0,
                    "operations_spend": 2800,
                    "development_spend": 1750,
                    "targeted_development_spend": 0,
                    "targeted_development_allocations_json": "{}",
                    "targeted_ad_allocations_json": '{"search_ads":{"S1":500}}',
                    "capacity_spend_weekly": 595,
                    "product_quality": 0.6,
                    "research_completed_quality_boost_total": 0.0,
                    "research_in_progress_count": 0,
                    "model_tier_a": 1,
                    "model_tier_b": 2,
                    "model_tier_c": 3,
                    "usage_quota_a": 120,
                    "usage_quota_b": 200,
                    "usage_quota_c": 500,
                    "capacity_tier": 1,
                    "daily_usage_per_customer": 90,
                    "recurring_promotion_monthly": 0,
                    "ads_strength": 0,
                    "targeted_ops_spend": 0,
                    "configured_price_a": 25,
                    "configured_price_b": 69,
                    "configured_price_c": 179,
                    "capacity": 1000,
                    "reputation": 0.95,
                    "known_segments": "S1,E1",
                    "open_issues": 0,
                    "enterprise_oldest_thread_age_days": 0,
                }
            ]
        }
    )


def market_feed(day: int) -> str:
    return json.dumps(
        {
            "rows": [
                {
                    "day": day,
                    "kind": "social_post",
                    "message": "A competitor shipped a product update.",
                },
                {
                    "day": day,
                    "kind": "macro",
                    "message": "The economy is expanding modestly.",
                },
            ]
        }
    )

def research_catalog_listing() -> str:
    """The environment's own R&D price list, as python-c relays it."""

    return json.dumps(
        {
            "tiers": [
                {
                    "tier": 1,
                    "cost": 166_667,
                    "mean_days": 12,
                    "mean_quality_boost": 0.04,
                    "in_progress": 0,
                    "completed": 0,
                },
                {
                    "tier": 3,
                    "cost": 500_000,
                    "mean_days": 23,
                    "mean_quality_boost": 0.11,
                    "in_progress": 1,
                    "completed": 0,
                },
            ]
        }
    )


def evidence(day: int) -> list[str]:
    return [
        json.dumps(
            {
                "rows": [
                    {
                        "segment": "S1",
                        "channel": "search_ads",
                        "leads": 12,
                        "spend": 500,
                    }
                ]
            }
        ),
        json.dumps(
            {
                "rows": [
                    {
                        "segment": "S1",
                        "channel": "search_ads",
                        "leads": 12,
                        "conversions": 5,
                        "losses": 6,
                        "pending": 1,
                    }
                ]
            }
        ),
        json.dumps(
            {
                "rows": [
                    {
                        "category": "advertising",
                        "weekly_amount": -500,
                        "cumulative_amount": -500,
                    }
                ]
            }
        ),
        json.dumps(
            {
                "rows": [
                    {
                        "price_A": 25,
                        "price_B": 69,
                        "price_C": 179,
                        "tier_A": 1,
                        "tier_B": 2,
                        "tier_C": 3,
                        "ad_spend_social_media": 0,
                        "ad_spend_search_ads": 500,
                        "ad_spend_linkedin": 0,
                        "ad_spend_content_marketing": 0,
                        "ad_spend_referral_program": 0,
                        "spend_operations": 400,
                        "spend_development": 250,
                        "capacity_tier": 0,
                        "lead_promotion_json": "{}",
                        "targeted_ads_json": '{"search_ads":{"S1":500}}',
                        "targeted_development_json": "{}",
                        "day": day,
                    }
                ]
            }
        ),
    ]


def competitor_signals(*posts: tuple[int, str]) -> str:
    return json.dumps(
        {"rows": [{"day": day, "content": content} for day, content in posts]}
    )

class ScriptedRunner:
    def __init__(self, outputs: Sequence[str]) -> None:
        self.outputs = deque(outputs)
        self.commands: list[tuple[str, ...]] = []

    async def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float,
    ) -> CommandResult:
        del cwd, timeout_seconds
        argv = tuple(command)
        self.commands.append(argv)
        return CommandResult(
            command=argv,
            returncode=0,
            stdout=self.outputs.popleft(),
            stderr="",
        )


def make_cli(runner: ScriptedRunner) -> CeobenchCli:
    return CeobenchCli(
        command=("python3.13", "novamind-operation"),
        working_directory=Path("ceobench"),
        runner=runner,
    )


def test_json_parser_accepts_a_cli_log_prefix() -> None:
    assert parse_json_output('server ready\n{"session_id":"session-1"}') == {
        "session_id": "session-1"
    }


def test_observation_reports_acquisition_spend_set_through_the_targeted_tool() -> None:
    # Targeted ad spend is recorded as a config override, not in the config
    # columns. Reading only the columns reported zero spend for a whole live run,
    # so the reversion baseline collapsed acquisition to nothing.
    database = sqlite3.connect(":memory:")
    database.executescript(
        """
        CREATE TABLE service_day (day INTEGER, capacity_units INTEGER);
        CREATE TABLE ledger (day INTEGER, category TEXT, amount REAL, note TEXT);
        CREATE TABLE research_projects (
            project_id TEXT, tier INTEGER, status TEXT,
            expected_quality_boost REAL, quality_boost_applied REAL
        );
        CREATE TABLE subscriptions (
            customer_id INTEGER, status TEXT, seat_count INTEGER,
            effective_price REAL, start_day INTEGER, end_day INTEGER
        );
        CREATE TABLE config_history (
            day INTEGER, spend_advertising REAL,
            price_A REAL, price_B REAL, price_C REAL,
            ad_spend_social_media REAL, ad_spend_search_ads REAL,
            ad_spend_linkedin REAL, ad_spend_content_marketing REAL,
            ad_spend_referral_program REAL, spend_operations REAL,
            spend_development REAL, tier_A INTEGER, tier_B INTEGER, tier_C INTEGER,
            quota_A INTEGER, quota_B INTEGER, quota_C INTEGER, capacity_tier INTEGER
        );
        CREATE TABLE daily_usage (
            day INTEGER, customer_id INTEGER, usage_units INTEGER
        );
        CREATE TABLE config_overrides (
            id INTEGER PRIMARY KEY, day INTEGER, tool_name TEXT,
            setting_type TEXT, settings_json TEXT
        );
        CREATE TABLE issues (status TEXT);
        CREATE TABLE group_info_levels (group_id TEXT, info_level INTEGER);
        CREATE TABLE customers (customer_id INTEGER, customer_type TEXT);
        CREATE TABLE enterprise_turns (
            message_id INTEGER, customer_id INTEGER, sender TEXT,
            day INTEGER, closed INTEGER, seat_count INTEGER
        );
        INSERT INTO service_day VALUES (7, 50000);
        INSERT INTO ledger VALUES (0, 'initial_funding', 1000000, NULL);
        INSERT INTO config_history VALUES (
            7, 0, 25, 69, 179, 0, 0, 0, 0, 0, 50, 100, 1, 1, 1, 100, 500, 2000, 1
        );
        INSERT INTO config_overrides VALUES (
            1, 7, 'set_targeted_ad_spend', 'targeted_ad_spend',
            '{"targeted_spend": {"search_ads": {"S1": 500.0},
              "linkedin": {"E1": 100.0}}}'
        );
        """
    )
    cursor = database.execute(OBSERVATION_QUERY)
    row = dict(zip([column[0] for column in cursor.description], cursor.fetchone(), strict=True))

    assert row["marketing_spend"] == pytest.approx(4_200.0)
    assert row["marketing_spend_search_ads_weekly"] == pytest.approx(3_500.0)
    assert row["marketing_spend_linkedin_weekly"] == pytest.approx(700.0)
    assert row["marketing_spend_social_media_weekly"] == pytest.approx(0.0)


def test_observation_query_returns_initial_snapshot_before_daily_tables_exist() -> None:
    database = sqlite3.connect(":memory:")
    database.executescript(
        """
        CREATE TABLE service_day (day INTEGER, capacity_units INTEGER);
        CREATE TABLE ledger (day INTEGER, category TEXT, amount REAL, note TEXT);
        CREATE TABLE research_projects (
            project_id TEXT, tier INTEGER, status TEXT,
            expected_quality_boost REAL, quality_boost_applied REAL
        );
        CREATE TABLE subscriptions (
            customer_id INTEGER, status TEXT, seat_count INTEGER,
            effective_price REAL, start_day INTEGER, end_day INTEGER
        );
        CREATE TABLE config_history (
            day INTEGER, spend_advertising REAL,
            price_A REAL, price_B REAL, price_C REAL,
            ad_spend_social_media REAL, ad_spend_search_ads REAL,
            ad_spend_linkedin REAL, ad_spend_content_marketing REAL,
            ad_spend_referral_program REAL, spend_operations REAL,
            spend_development REAL, tier_A INTEGER, tier_B INTEGER, tier_C INTEGER,
            quota_A INTEGER DEFAULT 0, quota_B INTEGER DEFAULT 0,
            quota_C INTEGER DEFAULT 0, capacity_tier INTEGER DEFAULT 0
        );
        CREATE TABLE daily_usage (
            day INTEGER, customer_id INTEGER, usage_units INTEGER
        );
        CREATE TABLE config_overrides (
            id INTEGER PRIMARY KEY, day INTEGER, tool_name TEXT,
            setting_type TEXT, settings_json TEXT
        );
        CREATE TABLE issues (status TEXT);
        CREATE TABLE group_info_levels (group_id TEXT, info_level INTEGER);
        CREATE TABLE customers (customer_id INTEGER, customer_type TEXT);
        CREATE TABLE enterprise_turns (
            message_id INTEGER, customer_id INTEGER, sender TEXT,
            day INTEGER, closed INTEGER, seat_count INTEGER
        );
        INSERT INTO ledger VALUES (0, 'initial_funding', 1000000, NULL);
        """
    )

    row = database.execute(OBSERVATION_QUERY).fetchone()

    assert row is not None
    assert row[0] == 0
    assert row[1] == 1_000_000
    assert row[-1] == "S1"


def test_action_mapper_rejects_unsupported_or_malformed_actions() -> None:
    with pytest.raises(BenchmarkContractError, match="unsupported"):
        build_action_code(
            ActionCommand(tool="shell", idempotency_key="bad-1", arguments={})
        )

    with pytest.raises(BenchmarkContractError, match="invalid arguments"):
        build_action_code(
            ActionCommand(
                tool="set_capacity_tier",
                idempotency_key="bad-2",
                arguments={"tier": 99},
            )
        )

    with pytest.raises(BenchmarkContractError, match="invalid arguments"):
        build_action_code(
            ActionCommand(
                tool="set_targeted_dev_spend",
                idempotency_key="bad-dev",
                arguments={"targeted_spend": {"S1": 10_001}},
            )
        )


def test_action_mapper_encodes_arguments_instead_of_interpolating_code() -> None:
    command = ActionCommand(
        tool="set_prices",
        idempotency_key="prices-1",
        arguments={"A": 25, "B": 69, "C": 179},
    )
    code = build_action_code(command)

    encoded = code.split("urlsafe_b64decode('", 1)[1].split("')", 1)[0]
    decoded = json.loads(base64.urlsafe_b64decode(encoded))
    assert decoded == {"A": 25.0, "B": 69.0, "C": 179.0}
    assert "pricing.set_prices(**kwargs)" in code


def test_action_mapper_supports_bounded_conversion_actions() -> None:
    promotion = build_action_code(
        ActionCommand(
            tool="set_lead_promotion",
            idempotency_key="promotion-1",
            arguments={"global_promotion": 20},
        )
    )
    deal = build_action_code(
        ActionCommand(
            tool="send_enterprise_deal",
            idempotency_key="deal-1",
            arguments={"deals": [[312, [["A", 9], ["B", 19]]]]},
        )
    )

    assert "marketing.set_lead_promotion(**kwargs)" in promotion
    assert "enterprise.send_enterprise_deal(**kwargs)" in deal

    with pytest.raises(BenchmarkContractError, match="invalid arguments"):
        build_action_code(
            ActionCommand(
                tool="send_enterprise_deal",
                idempotency_key="bad-deal",
                arguments={"deals": [[312, [["A", 9], ["A", 8]]]]},
            )
        )


@pytest.mark.asyncio
async def test_conversion_action_replay_does_not_call_ceobench_twice() -> None:
    runner = ScriptedRunner(['{"result":"lead promotion updated"}'])
    adapter = CeobenchAdapter(cli=make_cli(runner))
    run_id = uuid4()
    decision_id = uuid4()
    command = ActionCommand(
        tool="set_lead_promotion",
        idempotency_key="promotion-replay",
        arguments={"global_promotion": 20},
    )

    first = await adapter.execute_action(
        "session-1",
        run_id=run_id,
        decision_id=decision_id,
        command=command,
    )
    replay = await adapter.execute_action(
        "session-1",
        run_id=run_id,
        decision_id=decision_id,
        command=command,
    )

    assert first.status is ReceiptStatus.EXECUTED
    assert replay.status is ReceiptStatus.REPLAYED
    assert len(runner.commands) == 1


@pytest.mark.asyncio
async def test_next_week_uses_the_official_four_horizon_argument_order() -> None:
    runner = ScriptedRunner(["dashboard"])
    cli = make_cli(runner)
    forecasts = CashForecasts(
        items=[
            CashForecast(horizon_days=84, point=300, lower=250, upper=350),
            CashForecast(horizon_days=7, point=100, lower=90, upper=110),
            CashForecast(horizon_days=182, point=400, lower=300, upper=500),
            CashForecast(horizon_days=28, point=200, lower=175, upper=225),
        ]
    )

    await cli.next_week("session-1", rationale="bounded test", forecasts=forecasts)

    assert runner.commands == [
        (
            "python3.13",
            "novamind-operation",
            "next-week",
            "bounded test",
            "100",
            "90",
            "110",
            "200",
            "175",
            "225",
            "300",
            "250",
            "350",
            "400",
            "300",
            "500",
            "--session",
            "session-1",
        )
    ]


@pytest.mark.asyncio
async def test_one_week_flows_through_the_public_cli_contract() -> None:
    runner = ScriptedRunner(
        [
            '{"session_id":"session-1"}',
            observation(0, 1_000_000),
            market_feed(0),
            research_catalog_listing(),
            competitor_signals(),
            *evidence(0),
            '{"result":"prices updated"}',
            '{"result":"model tiers updated"}',
            '{"result":"spend updated"}',
            '{"result":"targeted spend updated"}',
            '{"result":"targeted development updated"}',
            observation(0, 1_000_000),
            market_feed(0),
            research_catalog_listing(),
            competitor_signals(),
            *evidence(0),
            "weekly dashboard",
            observation(7, 1_005_250),
            market_feed(7),
            research_catalog_listing(),
            competitor_signals(),
            *evidence(7),
        ]
    )
    adapter = CeobenchAdapter(cli=make_cli(runner), seed=42)
    manager = RunManager(
        repository=InMemoryRunRepository(),
        benchmark=adapter,
        decision_engine=StaticDecisionEngine(),
    )
    run = await manager.create_run()

    result = await manager.step_run(run.id, request_id="official-contract-week-0")

    assert result.run.current_day == 7
    assert result.decision.actual_outcome is not None
    assert result.decision.actual_outcome.cash == 1_005_250
    assert [receipt.tool for receipt in result.receipts] == [
        "set_prices",
        "set_model_tiers",
        "set_daily_spend",
        "set_targeted_ad_spend",
        "set_targeted_dev_spend",
    ]
    assert runner.commands[0][-5:] == (
        "new-session",
        "--days",
        "500",
        "--seed",
        "42",
    )
    python_commands = [command for command in runner.commands if command[2] == "python-c"]
    next_week_commands = [command for command in runner.commands if command[2] == "next-week"]
    # Five action tools plus one research-catalog listing per observation
    # (initial, post-action, post-advance).
    assert len(python_commands) == 8
    assert len(next_week_commands) == 1
    assert len(next_week_commands[0][4:16]) == 12
    assert result.decision.actual_outcome.evidence is not None
    assert result.decision.actual_outcome.evidence.cohorts[0].conversions == 5


@pytest.mark.asyncio
async def test_observation_normalizes_public_company_metrics() -> None:
    runner = ScriptedRunner(
        [
            observation(14, 925_000),
            market_feed(14),
            research_catalog_listing(),
            competitor_signals(
                (5, "a 0.01 quality boost"),
                (12, "roughly a 0.30 quality boost across the board"),
            ),
        ]
    )
    adapter = CeobenchAdapter(cli=make_cli(runner))

    snapshot = await adapter.observe_status("session-1")

    assert snapshot.day == 14
    assert snapshot.cash == 925_000
    assert snapshot.metrics["active_customers"] == 100
    assert snapshot.metrics["weekly_revenue"] == 5000
    assert snapshot.metrics["weekly_acquisition"] == 5
    assert snapshot.metrics["weekly_leads"] == 12
    assert snapshot.metrics["weekly_conversions"] == 5
    assert snapshot.metrics["weekly_lost_leads"] == 6
    assert snapshot.metrics["open_enterprise_threads"] == 2
    assert snapshot.metrics["known_segments"] == "S1,E1"
    assert snapshot.metrics["enterprise_inbox"] == ""
    assert snapshot.metrics["model_tier_a"] == 1
    assert snapshot.metrics["model_tier_b"] == 2
    assert snapshot.metrics["model_tier_c"] == 3
    assert snapshot.metrics["market_feed"] == (
        '[{"day":14,"kind":"social_post",'
        '"message":"A competitor shipped a product update."},'
        '{"day":14,"kind":"macro","message":"The economy is expanding modestly."}]'
    )
    # The bar the market moved, accumulated from announcements, with the recent
    # window isolated: 0.30 landed inside 28 days of day 14, 0.01 before it.
    assert snapshot.metrics["competitor_quality_releases"] == 2
    assert snapshot.metrics["competitor_quality_bar_shift"] == pytest.approx(0.31)
    assert snapshot.metrics["competitor_quality_bar_shift_28d"] == pytest.approx(0.31)
    assert snapshot.metrics["competitor_quality_bar_provenance"] == (
        "lower_bound:announced_releases:ceobench-market-signal-v2"
    )
    assert snapshot.metrics["source_query_version"] == "normalized-company-state-v13"
    assert snapshot.metrics["derived_metrics"] == (
        "product_quality,delivered_quality_plan_a,"
        "delivered_quality_plan_b,delivered_quality_plan_c"
    )
    assert snapshot.metrics["product_quality_provenance"] == (
        "derived:ledger:quality-proxy-v2"
    )
    assert snapshot.metrics["product_quality_confidence"] == "proxy"
    # Delivered per plan composes the proxy with the published tier multipliers
    # (tiers 1/2/3 in the fixture row).
    assert snapshot.metrics["delivered_quality_plan_a"] == pytest.approx(0.6 * 0.60)
    assert snapshot.metrics["delivered_quality_plan_b"] == pytest.approx(0.6 * 0.75)
    assert snapshot.metrics["delivered_quality_plan_c"] == pytest.approx(0.6 * 0.90)
    assert snapshot.metrics["delivered_quality_provenance"] == (
        "derived:product_quality*public-tier-multiplier"
    )


def test_observation_query_does_not_count_lost_leads_as_acquisition() -> None:
    database = sqlite3.connect(":memory:")
    database.row_factory = sqlite3.Row
    database.executescript(
        """
        CREATE TABLE service_day (day INTEGER, capacity_units INTEGER);
        CREATE TABLE ledger (day INTEGER, category TEXT, amount REAL, note TEXT);
        CREATE TABLE research_projects (
            project_id TEXT, tier INTEGER, status TEXT,
            expected_quality_boost REAL, quality_boost_applied REAL
        );
        CREATE TABLE subscriptions (
            customer_id INTEGER, status TEXT, seat_count INTEGER,
            effective_price REAL, start_day INTEGER, end_day INTEGER
        );
        CREATE TABLE config_history (
            day INTEGER, spend_advertising REAL,
            price_A REAL, price_B REAL, price_C REAL,
            ad_spend_social_media REAL, ad_spend_search_ads REAL,
            ad_spend_linkedin REAL, ad_spend_content_marketing REAL,
            ad_spend_referral_program REAL, spend_operations REAL,
            spend_development REAL, tier_A INTEGER, tier_B INTEGER, tier_C INTEGER,
            quota_A INTEGER DEFAULT 0, quota_B INTEGER DEFAULT 0,
            quota_C INTEGER DEFAULT 0, capacity_tier INTEGER DEFAULT 0
        );
        CREATE TABLE daily_usage (
            day INTEGER, customer_id INTEGER, usage_units INTEGER
        );
        CREATE TABLE config_overrides (
            id INTEGER PRIMARY KEY, day INTEGER, tool_name TEXT,
            setting_type TEXT, settings_json TEXT
        );
        CREATE TABLE issues (status TEXT);
        CREATE TABLE group_info_levels (group_id TEXT, info_level INTEGER);
        CREATE TABLE customers (customer_id INTEGER, customer_type TEXT);
        CREATE TABLE enterprise_turns (
            message_id INTEGER, customer_id INTEGER, sender TEXT,
            day INTEGER, closed INTEGER, seat_count INTEGER
        );
        INSERT INTO service_day VALUES (14, 1000);
        INSERT INTO ledger VALUES (0, 'initial_funding', 1000000, NULL);
        INSERT INTO subscriptions
            (status, seat_count, effective_price, start_day, end_day)
        VALUES ('lost', 1, 0, 10, 10);
        INSERT INTO subscriptions
            (status, seat_count, effective_price, start_day, end_day)
        VALUES ('lost', 1, 0, 11, 11);
        INSERT INTO subscriptions
            (status, seat_count, effective_price, start_day, end_day)
        VALUES ('lead', 1, 0, 12, NULL);
        INSERT INTO config_overrides VALUES (
            1, 13, 'set_lead_promotion', 'lead_promotion', '{"global": 5.0}'
        );
        INSERT INTO config_overrides VALUES (
            2, 13, 'set_targeted_dev_spend', 'targeted_dev_spend',
            '{"S2": 2000.0}'
        );
        INSERT INTO enterprise_turns VALUES (1, 101, 'customer', 13, 0, 20);
        INSERT INTO enterprise_turns VALUES (2, 102, 'customer', 12, 0, 30);
        INSERT INTO enterprise_turns VALUES (3, 102, 'agent', 13, 0, 30);
        """
    )

    row = dict(database.execute(OBSERVATION_QUERY).fetchone())

    assert row["weekly_leads"] == 3
    assert row["weekly_conversions"] == 0
    assert row["weekly_acquisition"] == 0
    assert row["weekly_lost_leads"] == 2
    assert row["pending_leads"] == 1
    assert row["lead_conversion_rate"] == 0
    assert row["lead_promotion_monthly"] == 5.0
    assert row["targeted_development_spend"] == 14_000.0
    assert json.loads(row["targeted_development_allocations_json"]) == {"S2": 2000.0}
    assert row["open_enterprise_threads"] == 1
    assert row["enterprise_inbox"] == "101:20:13"


def test_observation_separates_base_quality_from_model_tier_controls() -> None:
    database = sqlite3.connect(":memory:")
    database.row_factory = sqlite3.Row
    database.executescript(
        """
        CREATE TABLE service_day (day INTEGER, capacity_units INTEGER);
        CREATE TABLE ledger (day INTEGER, category TEXT, amount REAL, note TEXT);
        CREATE TABLE research_projects (
            project_id TEXT, tier INTEGER, status TEXT,
            expected_quality_boost REAL, quality_boost_applied REAL
        );
        CREATE TABLE subscriptions (
            customer_id INTEGER, status TEXT, seat_count INTEGER,
            effective_price REAL, start_day INTEGER, end_day INTEGER
        );
        CREATE TABLE config_history (
            day INTEGER, spend_advertising REAL,
            price_A REAL, price_B REAL, price_C REAL,
            ad_spend_social_media REAL, ad_spend_search_ads REAL,
            ad_spend_linkedin REAL, ad_spend_content_marketing REAL,
            ad_spend_referral_program REAL, spend_operations REAL,
            spend_development REAL, tier_A INTEGER, tier_B INTEGER, tier_C INTEGER,
            quota_A INTEGER DEFAULT 0, quota_B INTEGER DEFAULT 0,
            quota_C INTEGER DEFAULT 0, capacity_tier INTEGER DEFAULT 0
        );
        CREATE TABLE daily_usage (
            day INTEGER, customer_id INTEGER, usage_units INTEGER
        );
        CREATE TABLE config_overrides (
            id INTEGER PRIMARY KEY, day INTEGER, tool_name TEXT,
            setting_type TEXT, settings_json TEXT
        );
        CREATE TABLE issues (status TEXT);
        CREATE TABLE group_info_levels (group_id TEXT, info_level INTEGER);
        CREATE TABLE customers (customer_id INTEGER, customer_type TEXT);
        CREATE TABLE enterprise_turns (
            message_id INTEGER, customer_id INTEGER, sender TEXT,
            day INTEGER, closed INTEGER, seat_count INTEGER
        );
        INSERT INTO service_day VALUES (7, 1000);
        INSERT INTO ledger VALUES (0, 'initial_funding', 1000000, NULL);
        INSERT INTO ledger VALUES (1, 'development', -5000, NULL);
        INSERT INTO config_history VALUES (
            7, 0, 25, 69, 179, 10, 20, 0, 0, 0, 500, 250, 2, 3, 4, 0, 0, 0, 0
        );
        """
    )

    row = dict(database.execute(OBSERVATION_QUERY).fetchone())

    assert row["product_quality"] == pytest.approx(0.20 + 0.006 * 0.69314718056)
    assert (row["model_tier_a"], row["model_tier_b"], row["model_tier_c"]) == (2, 3, 4)
    assert row["marketing_spend_social_media_weekly"] == 70
    assert row["marketing_spend_search_ads_weekly"] == 140
    assert row["marketing_spend"] == 210


@pytest.mark.asyncio
async def test_observation_marks_an_unavailable_proxy_as_estimated() -> None:
    payload = json.loads(observation(0, 1_000_000))
    payload["rows"][0]["reputation"] = None
    runner = ScriptedRunner(
        [json.dumps(payload), market_feed(0), research_catalog_listing(), competitor_signals()]
    )
    adapter = CeobenchAdapter(cli=make_cli(runner))

    snapshot = await adapter.observe_status("session-1")

    assert snapshot.metrics["reputation"] == 0.5
    assert snapshot.metrics["estimated_metrics"] == "reputation"


@pytest.mark.asyncio
async def test_day_zero_marks_empty_prices_as_bootstrap_estimates() -> None:
    payload = json.loads(observation(0, 1_000_000))
    payload["rows"][0].update(
        {
            "active_customers": 0,
            "price_per_customer_weekly": 0,
            "price_a": 0,
            "price_b": 0,
            "price_c": 0,
        }
    )
    adapter = CeobenchAdapter(
        cli=make_cli(
            ScriptedRunner(
                [
                    json.dumps(payload),
                    market_feed(0),
                    research_catalog_listing(),
                    competitor_signals(),
                ]
            )
        )
    )

    snapshot = await adapter.observe_status("session-1")

    assert snapshot.metrics["price_a"] == 25
    assert snapshot.metrics["price_b"] == 69
    assert snapshot.metrics["price_c"] == 179
    assert "price_per_customer_weekly" in snapshot.metrics["estimated_metrics"]


@pytest.mark.asyncio
async def test_readonly_guard_runs_before_the_cli() -> None:
    runner = ScriptedRunner([])
    adapter = CeobenchAdapter(cli=make_cli(runner))

    with pytest.raises(BenchmarkContractError):
        await adapter.query_readonly(str(uuid4()), "DROP TABLE ledger")

    assert runner.commands == []
