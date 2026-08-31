"""The moving quality bar, read out of what the market says publicly."""

from __future__ import annotations

import pytest
from lithops.benchmark.ceobench.market_signals import (
    announced_quality_releases,
    quality_bar_shift,
)


def post(day: int, content: str) -> dict[str, object]:
    return {"day": day, "content": content}


class TestAnnouncedReleases:
    def test_one_release_discussed_for_days_is_counted_once(self) -> None:
        # The live shape: a single release is talked about for three days by
        # several people, each quoting a slightly different impression.
        rows = [
            post(62, "Just tested RivalTech's update — about a 0.0145 quality boost."),
            post(62, "QuantumEdge shipped: roughly a 0.0141 quality boost across the board."),
            post(63, "Still impressed by that release, call it a 0.0129 quality boost."),
            post(64, "A 0.0128 quality boost, barely noticeable in daily work."),
        ]

        releases = announced_quality_releases(rows)

        assert len(releases) == 1
        assert releases[0].first_day == 62
        assert releases[0].last_day == 64
        assert releases[0].mention_count == 4
        # Median of the daily medians, so one loud outlier cannot set the bar.
        assert releases[0].quality_shift == 0.0129

    def test_a_break_in_the_conversation_starts_the_next_release(self) -> None:
        rows = [
            post(62, "a 0.0145 quality boost"),
            post(63, "a 0.0129 quality boost"),
            post(81, "a 0.2119 quality boost — this changes expectations"),
            post(82, "a 0.2062 quality boost"),
        ]

        releases = announced_quality_releases(rows)

        assert [r.quality_shift for r in releases] == [0.0137, 0.20905]

    def test_posts_without_a_number_move_nothing(self) -> None:
        rows = [
            post(10, "ngl the onboarding has me 🤦‍♀️ but support was quick"),
            post(11, "switched back after trying that other tool"),
        ]

        assert announced_quality_releases(rows) == ()
        assert quality_bar_shift(()) == 0.0

    def test_malformed_rows_are_skipped_rather_than_fatal(self) -> None:
        rows = [
            {"day": None, "content": "a 0.5 quality boost"},
            {"day": 5, "content": None},
            {"day": "not-a-day", "content": "a 0.5 quality boost"},
            post(6, "a 0.02 quality boost"),
        ]

        releases = announced_quality_releases(rows)

        assert [r.quality_shift for r in releases] == [0.02]


class TestQualityBarShift:
    def test_the_shift_accumulates_across_releases(self) -> None:
        rows = [
            post(10, "a 0.01 quality boost"),
            post(30, "a 0.02 quality boost"),
            post(60, "a 0.30 quality boost"),
        ]

        releases = announced_quality_releases(rows)

        assert quality_bar_shift(releases) == pytest.approx(0.33)

    def test_a_recent_window_isolates_the_rate_that_matters_now(self) -> None:
        # The seed-83 collapse in miniature: the bar moved further in one recent
        # month than in the whole run before it.
        rows = [
            post(10, "a 0.01 quality boost"),
            post(30, "a 0.02 quality boost"),
            post(200, "a 0.34 quality boost"),
        ]

        releases = announced_quality_releases(rows)

        assert quality_bar_shift(releases, since_day=172) == pytest.approx(0.34)
        assert quality_bar_shift(releases) == pytest.approx(0.37)
