"""Purchased market information as durable, uncertainty-aware evidence.

Information actions are the one sanctioned exception to the rule that an executed
control must first be simulated: they change no company configuration. In return
they carry their own gate — an affordability ceiling and a duplication check —
and their parsed content is recorded with the noise band the source declares, so
a noisy estimate can never masquerade as a measurement.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field

SEGMENT_PATTERN = r"^(?:S[1-3]|E[1-3]|D_[SE]\d{2})$"

# Read-only tools whose value is the payload they return.
INFORMATION_TOOLS = frozenset(
    {
        "research_market",
        "research_group",
        "get_group_insights",
        "get_market_overview",
        "get_cost_info",
    }
)

# Purchased estimates age: markets move, so a measurement is treated as current
# for this many weeks. After that the duplication gate stops standing in the way
# of re-measuring the same thing — whether to re-buy stays the Executive's call.
INSIGHT_FRESHNESS_WEEKS = 8


class InsightParseStatus(StrEnum):
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    PENDING = "pending"


class InformationRequest(BaseModel):
    """One purchase of information, justified by the unknown it resolves."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool: str = Field(min_length=1, max_length=60)
    target_group: str | None = Field(default=None, pattern=SEGMENT_PATTERN)
    target_level: int | None = Field(default=None, ge=2, le=5)
    expected_information_value: str = Field(min_length=1, max_length=1_000)

    @property
    def identity(self) -> str:
        return f"{self.tool}:{self.target_group or '-'}:{self.target_level or '-'}"

    @property
    def price_key(self) -> str:
        """Purchases of the same tool and depth are priced alike."""

        return f"{self.tool}:{self.target_level or '-'}"


class InsightRecord(BaseModel):
    """Parsed content of one purchased information payload.

    Estimates carry the source's own accuracy band. They are priors to reason
    with and to fit against, never observations of realized outcomes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    run_id: UUID
    week: int = Field(ge=0)
    tool: str = Field(min_length=1, max_length=60)
    target_group: str | None = Field(default=None, pattern=SEGMENT_PATTERN)
    # The identity of the request that bought this, kept verbatim: the response's
    # own info level is not what was asked for, so it cannot stand in for it when
    # deciding whether the same question has already been paid for.
    request_identity: str = Field(default="", max_length=120)
    info_level: int | None = Field(default=None, ge=0, le=5)
    noise_band: float | None = Field(default=None, ge=0.0, le=1.0)
    willingness_to_pay_monthly: float | None = Field(default=None, ge=0.0)
    usage_units_per_day: float | None = Field(default=None, ge=0.0)
    quality_floor: float | None = Field(default=None, ge=0.0, le=1.0)
    market_cap_customers: float | None = Field(default=None, ge=0.0)
    discovered_group: str | None = Field(default=None, pattern=SEGMENT_PATTERN)
    parse_status: InsightParseStatus
    parser_version: str = Field(min_length=1, max_length=40)
    raw_excerpt: str = Field(default="", max_length=4_000)
    cost: float = Field(default=0.0, ge=0.0)
    created_at: datetime

    @property
    def price_key(self) -> str:
        """What a purchase of this kind costs is learned per tool and depth."""

        return f"{self.tool}:{self.info_level or '-'}"

    @property
    def has_decision_content(self) -> bool:
        return self.parse_status in {
            InsightParseStatus.SUCCEEDED,
            InsightParseStatus.PARTIAL,
        } and any(
            value is not None
            for value in (
                self.willingness_to_pay_monthly,
                self.usage_units_per_day,
                self.quality_floor,
                self.discovered_group,
            )
        )


def insight_record_id(run_id: UUID, week: int, identity: str) -> UUID:
    """Deterministic identity so a replayed week reduces to the same record."""

    return uuid5(NAMESPACE_URL, f"lithops:{run_id}:insight:{week}:{identity}")


def fresh_insight_identities(
    records: tuple[InsightRecord, ...] | list[InsightRecord],
    *,
    current_week: int,
) -> frozenset[str]:
    """Identities whose purchased measurement still counts as current.

    Only these block a re-purchase: the market moves, so after
    INSIGHT_FRESHNESS_WEEKS the same question may be asked again. The record
    itself stays forever — age is judged here, never by deleting evidence.
    """

    return frozenset(
        record.request_identity
        for record in records
        if record.request_identity
        and current_week - record.week < INSIGHT_FRESHNESS_WEEKS
    )


def measured_quality_floor_metrics(
    records: tuple[InsightRecord, ...] | list[InsightRecord],
) -> dict[str, float]:
    """Purchased participation floors, reduced to the two bars a forecast needs.

    Per group only the latest purchase counts; across groups the *lowest* floor
    is the binding one — the most accessible group is the first place delivered
    quality can start converting. Expectations drift upward between purchases,
    so an old floor is a lower bound, never an overstatement. No default: a
    floor nobody has bought yields no metric at all — unmeasured is not zero.
    """

    latest_by_group: dict[str, InsightRecord] = {}
    for record in records:
        if record.quality_floor is None or not record.has_decision_content:
            continue
        group = record.target_group
        if group is None:
            continue
        current = latest_by_group.get(group)
        if current is None or record.week > current.week:
            latest_by_group[group] = record
    metrics: dict[str, float] = {}
    individual = [
        record.quality_floor
        for group, record in latest_by_group.items()
        if record.quality_floor is not None
        and not group.removeprefix("D_").startswith("E")
    ]
    enterprise = [
        record.quality_floor
        for group, record in latest_by_group.items()
        if record.quality_floor is not None
        and group.removeprefix("D_").startswith("E")
    ]
    if individual:
        metrics["measured_quality_floor_individual"] = min(individual)
    if enterprise:
        metrics["measured_quality_floor_enterprise"] = min(enterprise)
    return metrics
