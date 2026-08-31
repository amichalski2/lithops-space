"""Read the market's moving quality bar out of what the market says publicly.

Customers post about competitor releases and often state how much better the
release felt. Each release is discussed for a few days, so the same event
appears many times; consecutive discussion days are one release, and its
estimate is the median of what people said about it.

The result is a floor, not a measurement: only announced releases that someone
quantified are counted, and every number is one person's impression. Against
two runs' hidden drift accumulators this recovered 100% and 46% of the true
shift — enough to compare the bar's rate against the company's own, which is
the decision this exists to inform. Python states the number and its
provenance; what to do about it is the Executive's call.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import median

MARKET_SIGNAL_PARSER_VERSION = "ceobench-market-signal-v2"

# "about a 0.0982 quality boost", "roughly a 0.3358 quality boost across the board"
QUALITY_BOOST_PATTERN = re.compile(r"(\d*\.\d+)\s*quality boost", re.IGNORECASE)

# Release-shaped talk that carries no number at all. A bar shift of zero next
# to nonzero unquantified releases means "drift unmeasured", never "no drift" —
# rendering those two identically is what let one run probe a dead segment six
# times while the bar walked away invisibly. The market names rivals by brand,
# not as "competitor", so the shape is a release verb about someone who is not
# us.
RELEASE_MENTION_PATTERN = re.compile(
    r"\b(released|releases|release|launch|launched|update[ds]?|upgrade[ds]?|"
    r"overhaul|breakthrough)\b",
    re.IGNORECASE,
)

# A release stays in conversation for a few days; a break in the conversation
# marks the next one.
SAME_RELEASE_MAX_GAP_DAYS = 1


@dataclass(frozen=True, slots=True)
class AnnouncedRelease:
    """One competitor release, as the market described it."""

    first_day: int
    last_day: int
    quality_shift: float
    mention_count: int


def _daily_estimates(rows: Sequence[Mapping[str, object]]) -> dict[int, list[float]]:
    estimates: dict[int, list[float]] = {}
    for row in rows:
        day = row.get("day")
        content = row.get("content")
        if day is None or not isinstance(content, str):
            continue
        try:
            day_number = int(day)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        for match in QUALITY_BOOST_PATTERN.finditer(content):
            try:
                estimates.setdefault(day_number, []).append(float(match.group(1)))
            except ValueError:
                continue
    return estimates


def announced_quality_releases(
    rows: Sequence[Mapping[str, object]],
) -> tuple[AnnouncedRelease, ...]:
    """Group quantified competitor mentions into the releases they describe."""

    estimates = _daily_estimates(rows)
    releases: list[AnnouncedRelease] = []
    current: list[tuple[int, float, int]] = []
    for day in sorted(estimates):
        values = estimates[day]
        if current and day - current[-1][0] > SAME_RELEASE_MAX_GAP_DAYS:
            releases.append(_release(current))
            current = []
        current.append((day, median(values), len(values)))
    if current:
        releases.append(_release(current))
    return tuple(releases)


def _release(days: list[tuple[int, float, int]]) -> AnnouncedRelease:
    return AnnouncedRelease(
        first_day=days[0][0],
        last_day=days[-1][0],
        quality_shift=median([value for _, value, _ in days]),
        mention_count=sum(count for _, _, count in days),
    )


def quality_bar_shift(
    releases: Sequence[AnnouncedRelease],
    *,
    since_day: int | None = None,
) -> float:
    """How far announced releases have moved the bar, optionally since a day."""

    return sum(
        release.quality_shift
        for release in releases
        if since_day is None or release.last_day >= since_day
    )


def unquantified_release_count(
    rows: Sequence[Mapping[str, object]],
    *,
    own_brand: str | None = None,
) -> int:
    """Releases the market discussed without ever putting a number on them.

    Grouped by the same conversation-gap rule as quantified releases. Each one
    is a bar movement of unknown size: evidence that the shift estimate is a
    floor with something above it, not a completed measurement. Posts about
    ``own_brand`` are excluded — our own launches do not move the bar we chase.
    """

    days: set[int] = set()
    for row in rows:
        day = row.get("day")
        content = row.get("content")
        if day is None or not isinstance(content, str):
            continue
        if QUALITY_BOOST_PATTERN.search(content):
            continue
        if own_brand and own_brand.lower() in content.lower():
            continue
        if RELEASE_MENTION_PATTERN.search(content):
            try:
                days.add(int(day))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
    count = 0
    previous: int | None = None
    for day in sorted(days):
        if previous is None or day - previous > SAME_RELEASE_MAX_GAP_DAYS:
            count += 1
        previous = day
    return count
