"""Deterministic, versioned parsing of CEO-Bench information-tool payloads.

The API returns structured estimates, which is what this reads first; the
formatted display text is handled as a fallback. Parsing stays best-effort by
construction: whatever cannot be read stays raw and is marked unparsed, and an
unparsed payload never updates a prior — it is visible to the Executive as text
and nothing more.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from uuid import UUID

from lithops.domain.insights import (
    InformationRequest,
    InsightParseStatus,
    InsightRecord,
    insight_record_id,
)

PARSER_VERSION = "ceobench-insight-parser-v2"

# Structured estimate keys as the API reports them.
_ESTIMATE_KEYS = {
    "willingness_to_pay": "willingness_to_pay_monthly",
    "usage_volume": "usage_units_per_day",
    "quality_floor_q_min": "quality_floor",
    "market_cap": "market_cap_customers",
}

_NUMBER = r"([0-9][0-9,_]*(?:\.[0-9]+)?)"
_INFO_LEVEL = re.compile(r"Info Level:\s*([0-5])", re.IGNORECASE)
_NOISE_BAND = re.compile(r"±\s*([0-9]{1,3})\s*%")
_WILLINGNESS = re.compile(
    rf"Willingness to pay:\s*~?\$?{_NUMBER}", re.IGNORECASE
)
_USAGE = re.compile(rf"Usage volume:\s*~?{_NUMBER}", re.IGNORECASE)
_QUALITY_FLOOR = re.compile(
    rf"Quality floor[^:]*:\s*~?{_NUMBER}", re.IGNORECASE
)
_MARKET_CAP = re.compile(rf"Market cap:\s*~?{_NUMBER}", re.IGNORECASE)
_GROUP_CODE = re.compile(r"\b(S[1-3]|E[1-3]|D_[SE]\d{2})\b")
_DISCOVERED = re.compile(
    r"discovered[^.\n]*?\b(S[1-3]|E[1-3]|D_[SE]\d{2})\b", re.IGNORECASE
)


def _number(match: re.Match[str] | None) -> float | None:
    if match is None:
        return None
    try:
        return float(match.group(1).replace(",", "").replace("_", ""))
    except (TypeError, ValueError):
        return None


def extract_payload_text(result: dict[str, object]) -> str:
    """Pull the tool's printed payload out of an action receipt result."""

    stdout = result.get("stdout")
    if not isinstance(stdout, str):
        return ""
    text = stdout.strip()
    # The generated call prints {'result': ...}; unwrap it when it is valid JSON.
    try:
        decoded = json.loads(text)
    except (TypeError, ValueError):
        return text
    if isinstance(decoded, dict) and "result" in decoded:
        inner = decoded["result"]
        return inner if isinstance(inner, str) else json.dumps(inner, default=str)
    return text


def _structured_fields(text: str) -> dict[str, object] | None:
    """Read the API's structured estimate payload, when that is what came back."""

    try:
        decoded = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(decoded, dict):
        return None
    fields: dict[str, object] = {}
    estimates = decoded.get("estimates")
    if isinstance(estimates, dict):
        for source, target in _ESTIMATE_KEYS.items():
            value = estimates.get(source)
            if isinstance(value, int | float):
                fields[target] = float(value)
    info_level = decoded.get("info_level")
    if isinstance(info_level, int | float):
        fields["info_level"] = int(info_level)
    noise = decoded.get("noise")
    if isinstance(noise, str):
        match = _NOISE_BAND.search(noise)
        if match is not None:
            fields["noise_band"] = min(1.0, max(0.0, float(match.group(1)) / 100.0))
    group = decoded.get("group_id")
    if isinstance(group, str) and _GROUP_CODE.fullmatch(group):
        fields["target_group"] = group
    for key in ("discovered_group", "discovered_group_id", "new_group"):
        candidate = decoded.get(key)
        if isinstance(candidate, str) and _GROUP_CODE.fullmatch(candidate):
            fields["discovered_group"] = candidate
            break
    # A payload that decodes as an object is structured evidence even when it
    # carries no per-group estimates, such as a market overview.
    fields.setdefault("_structured", True)
    return fields


def parse_insight(
    *,
    run_id: UUID,
    week: int,
    request: InformationRequest,
    payload: str,
    cost: float,
    created_at: datetime,
) -> InsightRecord:
    """Reduce one information payload into a typed, uncertainty-aware record."""

    text = payload.strip()
    structured = _structured_fields(text)
    if structured is not None:
        quality_floor = structured.get("quality_floor")
        estimate_count = sum(
            key in structured for key in _ESTIMATE_KEYS.values()
        )
        return InsightRecord(
            id=insight_record_id(run_id, week, request.identity),
            run_id=run_id,
            week=week,
            tool=request.tool,
            target_group=structured.get("target_group") or request.target_group,
            request_identity=request.identity,
            info_level=structured.get("info_level"),
            noise_band=structured.get("noise_band"),
            willingness_to_pay_monthly=structured.get("willingness_to_pay_monthly"),
            usage_units_per_day=structured.get("usage_units_per_day"),
            quality_floor=(
                quality_floor
                if isinstance(quality_floor, float) and 0.0 <= quality_floor <= 1.0
                else None
            ),
            market_cap_customers=structured.get("market_cap_customers"),
            discovered_group=structured.get("discovered_group"),
            parse_status=(
                InsightParseStatus.SUCCEEDED
                if estimate_count >= len(_ESTIMATE_KEYS)
                or request.tool in {"get_market_overview", "get_cost_info", "research_market"}
                else InsightParseStatus.PARTIAL
                if estimate_count
                else InsightParseStatus.SUCCEEDED
            ),
            parser_version=PARSER_VERSION,
            raw_excerpt=text[:4_000],
            cost=cost,
            created_at=created_at,
        )
    info_level = _number(_INFO_LEVEL.search(text))
    noise = _number(_NOISE_BAND.search(text))
    willingness = _number(_WILLINGNESS.search(text))
    usage = _number(_USAGE.search(text))
    quality_floor = _number(_QUALITY_FLOOR.search(text))
    market_cap = _number(_MARKET_CAP.search(text))

    discovered = None
    if request.tool == "research_market":
        match = _DISCOVERED.search(text)
        if match is not None:
            discovered = match.group(1)

    target_group = request.target_group
    if target_group is None:
        code = _GROUP_CODE.search(text)
        target_group = code.group(1) if code is not None else None

    parsed_fields = [willingness, usage, quality_floor, market_cap, discovered]
    known = sum(field is not None for field in parsed_fields)
    if not text:
        status = InsightParseStatus.FAILED
    elif known == 0:
        # Overview and cost payloads carry no per-group estimates; they are still
        # readable evidence for the Executive, just not prior-updating ones.
        status = (
            InsightParseStatus.SUCCEEDED
            if request.tool in {"get_market_overview", "get_cost_info"}
            else InsightParseStatus.FAILED
        )
    elif known == len([f for f in parsed_fields if f is not None]) and known >= 3:
        status = InsightParseStatus.SUCCEEDED
    else:
        status = InsightParseStatus.PARTIAL

    return InsightRecord(
        id=insight_record_id(run_id, week, request.identity),
        run_id=run_id,
        week=week,
        tool=request.tool,
        target_group=target_group,
        request_identity=request.identity,
        info_level=int(info_level) if info_level is not None else None,
        noise_band=(
            min(1.0, max(0.0, noise / 100.0)) if noise is not None else None
        ),
        willingness_to_pay_monthly=willingness,
        usage_units_per_day=usage,
        quality_floor=(
            quality_floor if quality_floor is not None and quality_floor <= 1.0 else None
        ),
        market_cap_customers=market_cap,
        discovered_group=discovered,
        parse_status=status,
        parser_version=PARSER_VERSION,
        raw_excerpt=text[:4_000],
        cost=cost,
        created_at=created_at,
    )
