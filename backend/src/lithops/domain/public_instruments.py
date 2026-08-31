"""Instrument tables published by the benchmark's own tool documentation.

Every value in this module is printed to the agent by the environment itself
(`get_cost_info` output, `set_model_tiers` impact text), so composing with it
is reading the vendor's spec sheet, not decoding hidden state. Anything the
environment does NOT publish — hidden thresholds, drift internals, exact
rationing formulas — must never appear here.
"""

from __future__ import annotations

# Published by get_cost_info and the set_model_tiers impact text, verbatim:
# "Model tiers are quality multipliers on product quality (Tier 4 = 1.0×
# reference). delivered_quality = product_quality × tier_multiplier."
MODEL_TIER_QUALITY_MULTIPLIER: dict[int, float] = {
    1: 0.60,
    2: 0.75,
    3: 0.90,
    4: 1.00,
    5: 1.10,
}
