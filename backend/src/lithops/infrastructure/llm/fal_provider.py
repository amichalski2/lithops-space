"""Structured generation through fal.ai's OpenRouter router endpoint.

fal exposes "any LLM, powered by OpenRouter" as a queue API: one prompt in, one
text out, billed per token against fal credits. It differs from OpenRouter's
chat API in three ways this provider absorbs:

- No ``response_format`` and no constrained decoding at all, so the schema
  rides in the system prompt and the reply is leniently parsed — the same
  posture the OpenRouter provider already ends up in when an upstream drops
  the format.
- No message history: each request is a single (system_prompt, prompt) pair,
  so validation retries fold the failed answer and its errors back into the
  prompt instead of appending chat turns.
- Asynchronous by contract: submit returns a queued request id, and the result
  is polled. Every completed response reports token usage and the exact spend
  in dollars, which flows into the same usage snapshot the other providers
  keep.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
from pydantic import ValidationError

from lithops.domain.ports.model_provider import StructuredOutput
from lithops.infrastructure.llm.openrouter_provider import (
    StructuredGenerationError,
    _extract_json,
)


class FalRouterProvider:
    submit_endpoint = "https://queue.fal.run/openrouter/router"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "google/gemini-3.7-flash",
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 240.0,
        poll_interval_seconds: float = 2.0,
        include_reasoning: bool = True,
    ) -> None:
        if not api_key:
            raise ValueError("fal API key is required")
        self.api_key = api_key
        self.model = model
        self.client = client
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        # The router refuses reasoning-mandatory models (Gemini 3.7 Flash among
        # them) unless reasoning is requested, and returns the reasoning in its
        # own field rather than inline, so requesting it costs nothing here.
        self.include_reasoning = include_reasoning
        self._usage = {
            "logical_calls": 0,
            "http_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_usd": 0.0,
            "validation_retries": 0,
        }

    def usage_snapshot(self) -> dict[str, float]:
        return dict(self._usage)

    @property
    def model_id(self) -> str:
        return self.model

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[StructuredOutput],
    ) -> StructuredOutput:
        schema_instruction = (
            "\n\nRespond with a single JSON object that validates against this "
            "JSON Schema — no markdown fences, no prose before or after. Honour "
            "every pattern, length and bound stated in it:\n"
            + json.dumps(output_schema.model_json_schema(), separators=(",", ":"))
        )
        prompt = user_prompt
        last_error: Exception | None = None
        self._usage["logical_calls"] += 1
        for attempt in range(3):
            try:
                content = await self._complete(
                    system_prompt=system_prompt + schema_instruction,
                    prompt=prompt,
                )
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
                raise StructuredGenerationError(
                    f"fal structured generation failed for model {self.model}"
                ) from exc
            try:
                return output_schema.model_validate_json(_extract_json(content))
            except ValidationError as exc:
                last_error = exc
                self._usage["validation_retries"] += 1
                if attempt < 2:
                    # Single-turn API: the correction rides in the next prompt.
                    prompt = (
                        user_prompt
                        + "\n\nYour previous response failed schema validation."
                        + " Previous response:\n"
                        + content[:4_000]
                        + "\n\nValidation errors:\n"
                        + str(exc)[:2_000]
                        + "\n\nReturn a corrected JSON object matching the schema exactly."
                    )
        detail = str(last_error)[:400] if last_error is not None else ""
        raise StructuredGenerationError(
            "fal returned invalid structured output after retries for "
            f"model {self.model}: {detail}"
        ) from last_error

    async def _complete(self, *, system_prompt: str, prompt: str) -> str:
        payload = {
            "model": self.model,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "reasoning": self.include_reasoning,
            "max_tokens": 16_000,
        }
        submitted = await self._request("POST", self.submit_endpoint, json=payload)
        status_url = submitted.get("status_url")
        response_url = submitted.get("response_url")
        if not isinstance(status_url, str) or not isinstance(response_url, str):
            raise TypeError("fal queue submission returned no status/response URLs")
        deadline = time.monotonic() + self.timeout_seconds
        interval = self.poll_interval_seconds
        while True:
            status_body = await self._request("GET", status_url)
            status = str(status_body.get("status") or "")
            if status == "COMPLETED":
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"fal request stayed {status or 'pending'} beyond "
                    f"{self.timeout_seconds:.0f}s"
                )
            await asyncio.sleep(interval)
            interval = min(interval * 1.5, 10.0)
        result = await self._request("GET", response_url)
        self._usage["http_calls"] += 1
        self._accumulate_usage(result.get("usage"))
        error = result.get("error")
        if error:
            raise RuntimeError("fal router error body: " + json.dumps(error)[:300])
        output = result.get("output")
        if not isinstance(output, str) or not output.strip():
            raise TypeError(
                "fal router response output must be a non-empty string; body: "
                + json.dumps(result)[:300]
            )
        return output

    async def _request(self, method: str, url: str, json: Any | None = None) -> dict:
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }
        # Transport failures are weather: back off and retry rather than losing
        # a simulated week to one bad minute.
        last_error: Exception | None = None
        for transport_attempt in range(4):
            if transport_attempt:
                await asyncio.sleep(5.0 * (2 ** (transport_attempt - 1)))
            try:
                if self.client is not None:
                    response = await self.client.request(
                        method, url, headers=headers, json=json
                    )
                else:
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        response = await client.request(
                            method, url, headers=headers, json=json
                        )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in {408, 409, 429} or (
                    exc.response.status_code >= 500
                ):
                    last_error = exc
                    continue
                raise
            except httpx.HTTPError as exc:
                last_error = exc
                continue
            data = response.json()
            if not isinstance(data, dict):
                raise TypeError("fal response must be a JSON object")
            return data
        raise (
            last_error
            if last_error is not None
            else RuntimeError("fal transport retries exhausted")
        )

    def _accumulate_usage(self, usage: object) -> None:
        if not isinstance(usage, dict):
            return
        for key, field in (
            ("prompt_tokens", "prompt_tokens"),
            ("completion_tokens", "completion_tokens"),
            ("cost", "cost_usd"),
        ):
            value = usage.get(key)
            if isinstance(value, int | float):
                self._usage[field] += value
