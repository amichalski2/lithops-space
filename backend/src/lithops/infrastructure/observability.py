"""OpenTelemetry tracing exported to Google Cloud Trace.

Every simulated week becomes one trace: observe, learn, decide, execute,
advance, commit. The exporter is installed once per process and only when
LITHOPS_TRACING is enabled, so local runs and tests stay dependency-free.
Setting the global tracer provider here also lets the Google ADK emit its
own spans into the same trace.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}

_tracer: Any | None = None
_initialized = False


def tracing_requested() -> bool:
    return os.getenv("LITHOPS_TRACING", "off").lower() in _TRUTHY


def init_tracing(service_name: str = "lithops") -> bool:
    """Install the Cloud Trace exporter once per process.

    Returns True when spans will be exported. Any failure (missing packages,
    no credentials, no project) downgrades to a logged warning: tracing must
    never take a run down.
    """

    global _tracer, _initialized
    if _initialized:
        return _tracer is not None
    _initialized = True
    if not tracing_requested():
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        logger.warning("LITHOPS_TRACING is on but OpenTelemetry is not installed: %s", exc)
        return False
    try:
        provider = TracerProvider(
            resource=Resource.create({"service.name": service_name})
        )
        provider.add_span_processor(BatchSpanProcessor(CloudTraceSpanExporter()))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("lithops")
    except Exception as exc:
        logger.warning("Cloud Trace exporter could not start: %s", exc)
        return False
    logger.info("Cloud Trace tracing enabled (service.name=%s)", service_name)
    return True


@contextmanager
def span(name: str, **attributes: str | int | float | bool) -> Iterator[None]:
    """Record one reasoning-chain span; a plain no-op while tracing is off."""

    if _tracer is None:
        yield
        return
    with _tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            current.set_attribute(f"lithops.{key}", value)
        yield
