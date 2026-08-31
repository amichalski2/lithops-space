"""Google Cloud Model Armor screening for untrusted environment text.

CEO-Bench hands Lithops free text written by simulated outsiders: enterprise
inbox threads, market announcements, social chatter. All of it ends up in the
executive's weekly brief, which makes it a prompt-injection surface. Before
the brief is assembled, every string field of the observation is screened
through a Model Armor template.

Modes (LITHOPS_MODEL_ARMOR):
  off      - screening disabled, module inert (default)
  monitor  - verdicts recorded on the run's event ledger; text passes through
  enforce  - additionally, flagged fields are replaced with a redaction notice

Screening fails open by design: an unreachable Model Armor API must never
stall a 504-day run. Errors are recorded on the same ledger event, so a
silent outage is still visible in the audit trail rather than reading as
"nothing was flagged".
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

_TEMPLATE_PATTERN = re.compile(
    r"^projects/[^/]+/locations/(?P<location>[^/]+)/templates/[^/]+$"
)
_MAX_TEXT_CHARS = 6000
_REQUEST_TIMEOUT_S = 10.0

REDACTION_NOTICE = (
    "[text withheld: this field matched a Model Armor prompt-injection filter]"
)


@dataclass(slots=True, frozen=True)
class ScreenResult:
    field: str
    flagged: bool
    filters: tuple[str, ...] = ()
    error: str | None = None


class ModelArmorScreener:
    def __init__(self, *, template: str, mode: str) -> None:
        match = _TEMPLATE_PATTERN.match(template)
        if match is None:
            raise ValueError(
                "LITHOPS_MODEL_ARMOR_TEMPLATE must look like "
                "projects/<p>/locations/<l>/templates/<t>"
            )
        if mode not in {"monitor", "enforce"}:
            raise ValueError("Model Armor mode must be monitor or enforce")
        self.template = template
        self.mode = mode
        location = match.group("location")
        self._endpoint = (
            f"https://modelarmor.{location}.rep.googleapis.com/v1/"
            f"{template}:sanitizeUserPrompt"
        )
        self._credentials = None

    def _token(self) -> str:
        import google.auth
        import google.auth.transport.requests

        if self._credentials is None:
            self._credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        if not self._credentials.valid:
            self._credentials.refresh(google.auth.transport.requests.Request())
        return self._credentials.token

    async def screen(self, texts: dict[str, str]) -> list[ScreenResult]:
        if not texts:
            return []
        try:
            token = await asyncio.to_thread(self._token)
        except Exception as exc:
            logger.warning("Model Armor credentials unavailable: %s", exc)
            return [
                ScreenResult(field=name, flagged=False, error=f"auth: {exc}")
                for name in texts
            ]
        results: list[ScreenResult] = []
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_S) as client:
            for name, text in texts.items():
                results.append(await self._screen_one(client, token, name, text))
        return results

    async def _screen_one(
        self, client: httpx.AsyncClient, token: str, name: str, text: str
    ) -> ScreenResult:
        try:
            response = await client.post(
                self._endpoint,
                headers={"Authorization": f"Bearer {token}"},
                json={"userPromptData": {"text": text[:_MAX_TEXT_CHARS]}},
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning("Model Armor screening failed for %s: %s", name, exc)
            return ScreenResult(field=name, flagged=False, error=str(exc))
        sanitization = payload.get("sanitizationResult", {})
        flagged = sanitization.get("filterMatchState") == "MATCH_FOUND"
        return ScreenResult(
            field=name,
            flagged=flagged,
            filters=tuple(sorted(_matched_filters(sanitization))),
        )


def _matched_filters(node: object, path: str = "") -> set[str]:
    """Collect the filter names that reported MATCH_FOUND, tolerating schema growth."""

    matched: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "matchState" and value == "MATCH_FOUND" and path:
                matched.add(path)
            else:
                matched |= _matched_filters(value, key if key != "filterResults" else path)
    elif isinstance(node, list):
        for item in node:
            matched |= _matched_filters(item, path)
    return matched


_screener: ModelArmorScreener | None = None
_screener_loaded = False


def get_screener() -> ModelArmorScreener | None:
    """Process-wide screener built from the environment; None when off."""

    global _screener, _screener_loaded
    if _screener_loaded:
        return _screener
    _screener_loaded = True
    mode = os.getenv("LITHOPS_MODEL_ARMOR", "off").lower()
    if mode in {"off", ""}:
        return None
    template = os.getenv("LITHOPS_MODEL_ARMOR_TEMPLATE", "")
    try:
        _screener = ModelArmorScreener(template=template, mode=mode)
    except ValueError as exc:
        logger.warning("Model Armor disabled: %s", exc)
        _screener = None
    return _screener
