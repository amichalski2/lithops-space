"""Deterministic interval-quality math shared by the ledger and model evaluation.

The prediction ledger scores one cash forecast against one observed outcome, while
temporal model evaluation scores several channels of a sampled predictive
distribution. Both need the same proper interval arithmetic, so it lives here as
pure functions over floats with no domain types attached.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil


def nearest_rank_quantile(values: Sequence[float], probability: float) -> float:
    """Return the nearest-rank quantile without interpolation.

    Interpolation is deliberately avoided so a degenerate sample set reports the
    exact repeated value rather than a synthesized spread. The rank is the standard
    ``ceil(p * n)``: truncating ``(n - 1) * p`` instead would collapse an upper
    quantile onto a low sample whenever the set is small, so a two-sample 97.5%
    bound would report the minimum.
    """

    if not values:
        raise ValueError("quantile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    ordered = sorted(values)
    rank = ceil(probability * len(ordered))
    index = min(max(rank - 1, 0), len(ordered) - 1)
    return ordered[index]


def interval_score(*, lower: float, upper: float, actual: float, alpha: float) -> float:
    """Score one central interval: width plus alpha-weighted miss penalties."""

    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must be between zero and one exclusive")
    if upper < lower:
        raise ValueError("interval upper bound cannot be below its lower bound")
    below_penalty = 2.0 / alpha * (lower - actual) if actual < lower else 0.0
    above_penalty = 2.0 / alpha * (actual - upper) if actual > upper else 0.0
    return (upper - lower) + below_penalty + above_penalty


def weighted_interval_score(
    *,
    point: float,
    lower: float,
    upper: float,
    actual: float,
    interval_probability: float,
) -> float:
    """Standard WIS for one central interval: point error plus its interval score.

    A zero-width interval earns no width credit but pays the full miss penalty, so
    a degenerate predictive distribution scores strictly worse than an honest one
    unless it is exactly right.
    """

    if not 0.0 < interval_probability < 1.0:
        raise ValueError("interval probability must be between zero and one exclusive")
    alpha = 1.0 - interval_probability
    score = interval_score(lower=lower, upper=upper, actual=actual, alpha=alpha)
    return (0.5 * abs(actual - point) + (alpha / 2.0) * score) / 1.5
