"""Pure interval arithmetic shared by the ledger and temporal model evaluation."""

from __future__ import annotations

import pytest
from lithops.evaluation.interval_math import (
    interval_score,
    nearest_rank_quantile,
    weighted_interval_score,
)


def test_nearest_rank_quantile_does_not_interpolate() -> None:
    values = [10.0, 20.0, 30.0, 40.0]

    assert nearest_rank_quantile(values, 0.0) == 10.0
    assert nearest_rank_quantile(values, 1.0) == 40.0
    assert nearest_rank_quantile([5.0, 5.0, 5.0], 0.975) == 5.0


def test_nearest_rank_quantile_rejects_empty_and_out_of_range() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        nearest_rank_quantile([], 0.5)
    with pytest.raises(ValueError, match="between zero and one"):
        nearest_rank_quantile([1.0], 1.5)


def test_covered_actual_scores_only_interval_width() -> None:
    score = interval_score(lower=90.0, upper=110.0, actual=100.0, alpha=0.05)

    assert score == pytest.approx(20.0)


def test_missed_actual_pays_the_alpha_weighted_penalty() -> None:
    below = interval_score(lower=90.0, upper=110.0, actual=80.0, alpha=0.05)
    above = interval_score(lower=90.0, upper=110.0, actual=120.0, alpha=0.05)

    assert below == pytest.approx(20.0 + 2.0 / 0.05 * 10.0)
    assert above == pytest.approx(20.0 + 2.0 / 0.05 * 10.0)


def test_zero_width_interval_is_punished_unless_exactly_right() -> None:
    exact = weighted_interval_score(
        point=100.0, lower=100.0, upper=100.0, actual=100.0, interval_probability=0.95
    )
    degenerate_miss = weighted_interval_score(
        point=100.0, lower=100.0, upper=100.0, actual=101.0, interval_probability=0.95
    )
    honest = weighted_interval_score(
        point=100.0, lower=95.0, upper=105.0, actual=101.0, interval_probability=0.95
    )

    assert exact == pytest.approx(0.0)
    assert degenerate_miss > honest, "a collapsed interval must not beat an honest one"


def test_weighted_interval_score_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        weighted_interval_score(
            point=1.0, lower=0.0, upper=2.0, actual=1.0, interval_probability=1.0
        )
