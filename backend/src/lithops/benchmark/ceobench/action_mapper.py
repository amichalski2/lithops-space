from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from lithops.domain.errors import BenchmarkContractError
from lithops.domain.models import ActionCommand

PositiveMoney = Annotated[float, Field(gt=0)]
NonNegativeMoney = Annotated[float, Field(ge=0)]
ModelTier = Annotated[int, Field(ge=1, le=5)]
UsageQuota = Annotated[int, Field(ge=0, le=100_000)]
CustomerId = Annotated[int, Field(gt=0)]
PlanName = Literal["A", "B", "C"]
BoundedDailySpend = Annotated[float, Field(ge=0, le=10_000)]


class _StrictArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SetPricesArguments(_StrictArguments):
    A: PositiveMoney
    B: PositiveMoney
    C: PositiveMoney


class SetModelTiersArguments(_StrictArguments):
    A: ModelTier
    B: ModelTier
    C: ModelTier


class SetUsageQuotasArguments(_StrictArguments):
    A: UsageQuota
    B: UsageQuota
    C: UsageQuota


class SetCapacityTierArguments(_StrictArguments):
    tier: int = Field(ge=0, le=7)


class SetDailySpendArguments(_StrictArguments):
    operations: NonNegativeMoney
    development: NonNegativeMoney


class SetTargetedAdSpendArguments(_StrictArguments):
    targeted_spend: dict[str, dict[str, NonNegativeMoney]]

    @field_validator("targeted_spend")
    @classmethod
    def validate_targets(
        cls, value: dict[str, dict[str, float]]
    ) -> dict[str, dict[str, float]]:
        channels = {
            "social_media",
            "search_ads",
            "linkedin",
            "content_marketing",
            "referral_program",
        }
        invalid_channels = set(value) - channels
        if invalid_channels:
            raise ValueError(f"unsupported ad channels: {sorted(invalid_channels)}")
        group_pattern = re.compile(r"^(?:S[1-3]|E[1-3]|D_[SE]\d{2})$")
        invalid_groups = sorted(
            group
            for groups in value.values()
            for group in groups
            if not group_pattern.fullmatch(group)
        )
        if invalid_groups:
            raise ValueError(f"unsupported customer groups: {invalid_groups}")
        return value


class SetTargetedDevSpendArguments(_StrictArguments):
    targeted_spend: dict[str, BoundedDailySpend]

    @field_validator("targeted_spend")
    @classmethod
    def validate_targets(cls, value: dict[str, float]) -> dict[str, float]:
        group_pattern = re.compile(r"^(?:S[1-3]|E[1-3]|D_[SE]\d{2})$")
        invalid_groups = sorted(
            group for group in value if not group_pattern.fullmatch(group)
        )
        if invalid_groups:
            raise ValueError(f"unsupported customer groups: {invalid_groups}")
        if sum(value.values()) > 10_000:
            raise ValueError("total targeted development spend exceeds 10,000/day")
        return value


class SetLeadPromotionArguments(_StrictArguments):
    global_promotion: NonNegativeMoney


class SetPromotionArguments(_StrictArguments):
    """Recurring discount on the listed price, unlike the first-invoice promotion."""

    global_promotion: NonNegativeMoney


class SetAdsStrengthArguments(_StrictArguments):
    """In-product advertising: revenue per seat against perceived quality."""

    global_strength: Annotated[float, Field(ge=0.0, le=1.0)]


class SetTargetedOpsSpendArguments(_StrictArguments):
    targeted_spend: dict[str, BoundedDailySpend]

    @field_validator("targeted_spend")
    @classmethod
    def validate_targets(cls, value: dict[str, float]) -> dict[str, float]:
        group_pattern = re.compile(r"^(?:S[1-3]|E[1-3]|D_[SE]\d{2})$")
        invalid_groups = sorted(
            group for group in value if not group_pattern.fullmatch(group)
        )
        if invalid_groups:
            raise ValueError(f"unsupported customer groups: {invalid_groups}")
        return value


class PostSocialMediaArguments(_StrictArguments):
    content: str = Field(min_length=1, max_length=280)


class StartResearchProjectArguments(_StrictArguments):
    tier: int = Field(ge=1, le=20)


class ResearchGroupArguments(_StrictArguments):
    group_id: str = Field(pattern=r"^(?:S[1-3]|E[1-3]|D_[SE]\d{2})$")
    target_level: int | None = Field(default=None, ge=2, le=5)


class GetGroupInsightsArguments(_StrictArguments):
    group_id: str = Field(pattern=r"^(?:S[1-3]|E[1-3]|D_[SE]\d{2})$")


class NoArguments(_StrictArguments):
    """Tools invoked without parameters."""


class RejectEnterpriseDealArguments(_StrictArguments):
    deals: list[CustomerId] = Field(min_length=1, max_length=100)

    @field_validator("deals")
    @classmethod
    def validate_unique(cls, value: list[int]) -> list[int]:
        if len(set(value)) != len(value):
            raise ValueError("enterprise customer IDs must be unique")
        return value


class SendEnterpriseDealArguments(_StrictArguments):
    deals: list[tuple[CustomerId, list[tuple[PlanName, PositiveMoney]]]] = Field(
        min_length=1,
        max_length=100,
    )

    @field_validator("deals")
    @classmethod
    def validate_deals(
        cls,
        value: list[tuple[int, list[tuple[PlanName, float]]]],
    ) -> list[tuple[int, list[tuple[PlanName, float]]]]:
        customer_ids = [customer_id for customer_id, _ in value]
        if len(set(customer_ids)) != len(customer_ids):
            raise ValueError("enterprise customer IDs must be unique")
        for _, offerings in value:
            if not 1 <= len(offerings) <= 3:
                raise ValueError("each enterprise deal requires 1-3 offerings")
            plans = [plan for plan, _ in offerings]
            if len(set(plans)) != len(plans):
                raise ValueError("enterprise offering plans must be unique")
        return value


@dataclass(frozen=True, slots=True)
class ActionSpec:
    module: str
    function: str
    arguments_model: type[BaseModel]


ACTION_SPECS: dict[str, ActionSpec] = {
    "set_prices": ActionSpec("pricing", "set_prices", SetPricesArguments),
    "set_model_tiers": ActionSpec(
        "pricing", "set_model_tiers", SetModelTiersArguments
    ),
    "set_usage_quotas": ActionSpec(
        "pricing", "set_usage_quotas", SetUsageQuotasArguments
    ),
    "set_capacity_tier": ActionSpec(
        "infrastructure", "set_capacity_tier", SetCapacityTierArguments
    ),
    "set_daily_spend": ActionSpec(
        "marketing", "set_daily_spend", SetDailySpendArguments
    ),
    "set_targeted_ad_spend": ActionSpec(
        "marketing", "set_targeted_ad_spend", SetTargetedAdSpendArguments
    ),
    "set_targeted_dev_spend": ActionSpec(
        "analytics", "set_targeted_dev_spend", SetTargetedDevSpendArguments
    ),
    "set_lead_promotion": ActionSpec(
        "marketing", "set_lead_promotion", SetLeadPromotionArguments
    ),
    "set_promotion": ActionSpec("pricing", "set_promotion", SetPromotionArguments),
    "set_ads_strength": ActionSpec(
        "marketing", "set_ads_strength", SetAdsStrengthArguments
    ),
    "set_targeted_ops_spend": ActionSpec(
        "analytics", "set_targeted_ops_spend", SetTargetedOpsSpendArguments
    ),
    "post_social_media": ActionSpec(
        "marketing", "post_social_media", PostSocialMediaArguments
    ),
    "start_research_project": ActionSpec(
        "research", "start_research_project", StartResearchProjectArguments
    ),
    "send_enterprise_deal": ActionSpec(
        "enterprise", "send_enterprise_deal", SendEnterpriseDealArguments
    ),
    "reject_enterprise_deal": ActionSpec(
        "enterprise", "reject_enterprise_deal", RejectEnterpriseDealArguments
    ),
    # Read-only tools. Their value is the payload they return, so they change no
    # configuration and are exempt from configuration fidelity.
    "research_market": ActionSpec("market", "research_market", NoArguments),
    "research_group": ActionSpec("market", "research_group", ResearchGroupArguments),
    "get_group_insights": ActionSpec(
        "market", "get_group_insights", GetGroupInsightsArguments
    ),
    "get_market_overview": ActionSpec("market", "get_market_overview", NoArguments),
    "get_cost_info": ActionSpec("infrastructure", "get_cost_info", NoArguments),
}


def build_action_code(command: ActionCommand) -> str:
    spec = ACTION_SPECS.get(command.tool)
    if spec is None:
        raise BenchmarkContractError(
            f"unsupported CEO-Bench action tool: {command.tool}"
        )
    try:
        arguments = spec.arguments_model.model_validate(command.arguments)
    except ValidationError as exc:
        raise BenchmarkContractError(
            f"invalid arguments for CEO-Bench tool {command.tool}: {exc}"
        ) from exc

    encoded = base64.urlsafe_b64encode(
        json.dumps(arguments.model_dump(mode="json"), separators=(",", ":")).encode()
    ).decode("ascii")
    return "\n".join(
        [
            "import base64, json",
            f"from novamind_api import {spec.module}",
            f"kwargs = json.loads(base64.urlsafe_b64decode('{encoded}'))",
            f"result = {spec.module}.{spec.function}(**kwargs)",
            "print(json.dumps({'result': result}, default=str))",
        ]
    )
