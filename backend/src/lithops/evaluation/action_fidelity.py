"""Deterministic assertions that semantic actions changed public configuration.

Every state-changing tool must have a verifier here. A tool with no verifier is
reported as `{tool}.unverified` rather than silently passing, so extending the
executable surface cannot quietly weaken post-action verification.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from lithops.domain.evidence import ConfigurationEvidence
from lithops.domain.models import ActionCommand, ObservationSnapshot

# Read-only tools: they buy information and change no company configuration.
INFORMATION_TOOLS = frozenset(
    {
        "research_market",
        "research_group",
        "get_group_insights",
        "get_market_overview",
        "get_cost_info",
    }
)

# Tools whose effect is a benchmark-side event (a queued project, a negotiation
# turn, a published post) rather than a configuration row. They are verified by
# their own stage against the matching benchmark table, not by this snapshot.
EVENT_VERIFIED_TOOLS = frozenset(
    {
        "post_social_media",
        "send_enterprise_deal",
        "reject_enterprise_deal",
        "start_research_project",
    }
)


def _close(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) <= 1e-6
    except (TypeError, ValueError):
        return left == right


def _json_object(value: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _unwrap_targeted(value: dict[str, Any], *, key: str = "targeted_spend") -> dict[str, Any]:
    """Read the per-group allocation out of a stored settings payload.

    Tools differ in how they record it: some keep the legacy ``targeted_spend``
    wrapper, others normalize it into a scope key such as ``by_group``.
    """

    nested = value.get(key)
    return nested if isinstance(nested, dict) else value


def _verify_prices(
    expected: Mapping[str, Any], config: ConfigurationEvidence
) -> list[str]:
    return [
        f"set_prices.{plan}"
        for plan in "ABC"
        if not _close(config.prices.get(plan), expected.get(plan))
    ]


def _verify_model_tiers(
    expected: Mapping[str, Any], config: ConfigurationEvidence
) -> list[str]:
    return [
        f"set_model_tiers.{plan}"
        for plan in "ABC"
        if config.model_tiers.get(plan) != expected.get(plan)
    ]


def _verify_usage_quotas(
    expected: Mapping[str, Any], config: ConfigurationEvidence
) -> list[str]:
    return [
        f"set_usage_quotas.{plan}"
        for plan in "ABC"
        if not _close(config.usage_quotas.get(plan), expected.get(plan))
    ]


def _verify_daily_spend(
    expected: Mapping[str, Any], config: ConfigurationEvidence
) -> list[str]:
    violations: list[str] = []
    if not _close(config.daily_operations_spend, expected.get("operations")):
        violations.append("set_daily_spend.operations")
    if not _close(config.daily_development_spend, expected.get("development")):
        violations.append("set_daily_spend.development")
    return violations


def _verify_capacity_tier(
    expected: Mapping[str, Any], config: ConfigurationEvidence
) -> list[str]:
    if config.capacity_tier != expected.get("tier"):
        return ["set_capacity_tier.tier"]
    return []


def _verify_targeted_ad_spend(
    expected: Mapping[str, Any], config: ConfigurationEvidence
) -> list[str]:
    actual = _unwrap_targeted(_json_object(config.targeted_ads_json))
    if actual != expected.get("targeted_spend"):
        return ["set_targeted_ad_spend.targeted_spend"]
    return []


def _verify_targeted_dev_spend(
    expected: Mapping[str, Any], config: ConfigurationEvidence
) -> list[str]:
    actual = _unwrap_targeted(_json_object(config.targeted_development_json))
    if actual != expected.get("targeted_spend"):
        return ["set_targeted_dev_spend.targeted_spend"]
    return []


def _verify_lead_promotion(
    expected: Mapping[str, Any], config: ConfigurationEvidence
) -> list[str]:
    actual = _json_object(config.lead_promotion_json)
    active = actual.get("global", actual.get("global_promotion", 0.0))
    if not _close(active, expected.get("global_promotion")):
        return ["set_lead_promotion.global_promotion"]
    return []


def _verify_recurring_promotion(
    expected: Mapping[str, Any], config: ConfigurationEvidence
) -> list[str]:
    actual = _json_object(config.recurring_promotion_json)
    active = actual.get("global", actual.get("global_promotion", 0.0))
    if not _close(active, expected.get("global_promotion")):
        return ["set_promotion.global_promotion"]
    return []


def _verify_ads_strength(
    expected: Mapping[str, Any], config: ConfigurationEvidence
) -> list[str]:
    actual = _json_object(config.ads_strength_json)
    active = actual.get("global", actual.get("global_strength", 0.0))
    if not _close(active, expected.get("global_strength")):
        return ["set_ads_strength.global_strength"]
    return []


def _verify_targeted_ops_spend(
    expected: Mapping[str, Any], config: ConfigurationEvidence
) -> list[str]:
    # This tool normalizes the legacy ``targeted_spend`` argument into ``by_group``
    # before recording it, so that is the scope to compare against.
    actual = _unwrap_targeted(_json_object(config.targeted_ops_json), key="by_group")
    requested = expected.get("targeted_spend")
    if not isinstance(requested, dict):
        return ["set_targeted_ops_spend.targeted_spend"]
    if {group: float(amount) for group, amount in actual.items()} != {
        group: float(amount) for group, amount in requested.items()
    }:
        return ["set_targeted_ops_spend.targeted_spend"]
    return []


Verifier = Callable[[Mapping[str, Any], ConfigurationEvidence], list[str]]

CONFIGURATION_VERIFIERS: dict[str, Verifier] = {
    "set_prices": _verify_prices,
    "set_model_tiers": _verify_model_tiers,
    "set_usage_quotas": _verify_usage_quotas,
    "set_daily_spend": _verify_daily_spend,
    "set_capacity_tier": _verify_capacity_tier,
    "set_targeted_ad_spend": _verify_targeted_ad_spend,
    "set_targeted_dev_spend": _verify_targeted_dev_spend,
    "set_lead_promotion": _verify_lead_promotion,
    "set_promotion": _verify_recurring_promotion,
    "set_ads_strength": _verify_ads_strength,
    "set_targeted_ops_spend": _verify_targeted_ops_spend,
}


def action_fidelity_violations(
    commands: list[ActionCommand], observation: ObservationSnapshot
) -> tuple[str, ...]:
    """Compare intended setters with the public post-action configuration snapshot."""

    if observation.evidence is None:
        return ("post-action evidence packet is missing",)
    config = observation.evidence.configuration
    violations: list[str] = []
    for command in commands:
        verifier = CONFIGURATION_VERIFIERS.get(command.tool)
        if verifier is not None:
            violations.extend(verifier(command.arguments, config))
        elif command.tool in INFORMATION_TOOLS or command.tool in EVENT_VERIFIED_TOOLS:
            continue
        else:
            violations.append(f"{command.tool}.unverified")
    return tuple(violations)
