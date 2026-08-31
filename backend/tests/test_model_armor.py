from __future__ import annotations

import pytest

from lithops.infrastructure.observability import span
from lithops.infrastructure.security.model_armor import (
    ModelArmorScreener,
    _matched_filters,
)


def test_screener_rejects_malformed_template() -> None:
    with pytest.raises(ValueError):
        ModelArmorScreener(template="not-a-template", mode="monitor")


def test_screener_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        ModelArmorScreener(
            template="projects/p/locations/eu/templates/t", mode="audit"
        )


def test_screener_builds_regional_endpoint() -> None:
    screener = ModelArmorScreener(
        template="projects/p/locations/europe-west4/templates/lithops",
        mode="enforce",
    )
    assert screener.mode == "enforce"
    assert screener._endpoint == (
        "https://modelarmor.europe-west4.rep.googleapis.com/v1/"
        "projects/p/locations/europe-west4/templates/lithops:sanitizeUserPrompt"
    )


def test_matched_filters_walks_nested_results() -> None:
    sanitization = {
        "filterMatchState": "MATCH_FOUND",
        "filterResults": {
            "pi_and_jailbreak": {
                "piAndJailbreakFilterResult": {"matchState": "MATCH_FOUND"}
            },
            "rai": {"raiFilterResult": {"matchState": "NO_MATCH_FOUND"}},
        },
    }
    assert _matched_filters(sanitization) == {"piAndJailbreakFilterResult"}


def test_span_is_noop_without_tracing() -> None:
    with span("week.observe", week=3):
        pass
