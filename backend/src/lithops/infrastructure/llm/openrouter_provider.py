from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from lithops.domain.ports.model_provider import StructuredOutput


class StructuredGenerationError(RuntimeError):
    """A provider failed to return a response matching the requested schema."""


# JSON Schema constraint keywords some upstream providers do not support in
# constrained decoding. Anthropic's structured outputs, for example, reject
# numeric bounds, string length/pattern limits and array cardinality — and when
# such a schema arrives via OpenRouter the response_format is silently dropped,
# so the model answers in prose and every parse fails. The keywords are removed
# from the schema sent upstream only; pydantic still enforces every one of them
# when the response is validated client-side.
_UNSUPPORTED_SCHEMA_KEYWORDS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
    }
)


def _provider_safe_schema(schema: Any) -> Any:
    """Strip constraint keywords constrained decoders cannot honour."""

    if isinstance(schema, dict):
        return {
            key: _provider_safe_schema(value)
            for key, value in schema.items()
            if key not in _UNSUPPORTED_SCHEMA_KEYWORDS
        }
    if isinstance(schema, list):
        return [_provider_safe_schema(item) for item in schema]
    return schema


def _extract_json(content: str) -> str:
    """The JSON object inside a response that may carry fences or prose.

    When an upstream drops the response format, capable models still answer
    with the requested JSON — wrapped in ```json fences or led by a sentence.
    Cutting to the outermost braces recovers it; pydantic then decides whether
    what was recovered actually validates.
    """

    text = content.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return content


class OpenRouterProvider:
    endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "qwen/qwen3-32b",
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 60.0,
        provider_sort: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OpenRouter API key is required")
        self.api_key = api_key
        self.model = model
        self.client = client
        self.timeout_seconds = timeout_seconds
        self.provider_sort = provider_sort
        # Reasoning models bill their chain of thought as output tokens and run
        # at that model's generation speed: on one weekly loop a trivial choice
        # cost 103 seconds and 1,309 output tokens at the default effort, and
        # 10 seconds at "low". Left unbounded, long answers also truncate at
        # max_tokens mid-JSON, which then costs a retry on top. None leaves the
        # provider's own default alone for models where this does not apply.
        self.reasoning_effort = reasoning_effort
        # A harness that spends money on its own behalf has to be able to say how
        # much. Counted per HTTP call, so a schema retry is visible as the second
        # request it actually is.
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
        # Constrained decoding is requested via response_format, but not every
        # upstream honours it: Anthropic rejects rich schemas as "too complex"
        # and the format is then silently dropped, so the schema also rides in
        # the system prompt and parsing tolerates fences and surrounding prose.
        # The prompt copy keeps every constraint (patterns, lengths, bounds) —
        # the model must know the real rules; only the response_format copy is
        # sanitized, because constrained decoders reject those keywords.
        schema_instruction = (
            "\n\nRespond with a single JSON object that validates against this "
            "JSON Schema — no markdown fences, no prose before or after. Honour "
            "every pattern, length and bound stated in it:\n"
            + json.dumps(output_schema.model_json_schema(), separators=(",", ":"))
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt + schema_instruction},
            {"role": "user", "content": user_prompt},
        ]
        last_error: Exception | None = None
        content = ""
        self._usage["logical_calls"] += 1
        for attempt in range(3):
            payload = self._payload(messages, output_schema)
            try:
                response = await self._post(payload)
                content = self._extract_content(response)
                return output_schema.model_validate_json(_extract_json(content))
            except ValidationError as exc:
                last_error = exc
                self._usage["validation_retries"] += 1
                if attempt < 2:
                    messages.append({"role": "assistant", "content": content})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The prior response failed schema validation. Return a corrected "
                                "response matching the schema exactly. Validation errors:\n"
                                + str(exc)[:2_000]
                            ),
                        }
                    )
                    continue
                break
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
                raise StructuredGenerationError(
                    f"OpenRouter structured generation failed for model {self.model}"
                ) from exc
        detail = str(last_error)[:400] if last_error is not None else ""
        raise StructuredGenerationError(
            "OpenRouter returned invalid structured output after retries for "
            f"model {self.model}: {detail}"
        ) from last_error

    def _payload(
        self,
        messages: list[dict[str, str]],
        output_schema: type[BaseModel],
    ) -> dict[str, Any]:
        schema_name = re.sub(r"[^a-zA-Z0-9_-]", "_", output_schema.__name__).lower()
        provider: dict[str, Any] = {"require_parameters": True}
        if self.provider_sort is not None:
            provider["sort"] = self.provider_sort
        return {
            "model": self.model,
            "messages": messages,
            "usage": {"include": True},
            # Without an explicit ceiling some routes truncate long structured
            # answers mid-JSON; 16k leaves room for the largest weekly output.
            "max_tokens": 16_000,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": _provider_safe_schema(
                        output_schema.model_json_schema()
                    ),
                },
            },
            "provider": provider,
            **(
                {"reasoning": {"effort": self.reasoning_effort}}
                if self.reasoning_effort
                else {}
            ),
        }

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "Lithops",
        }
        # Rate limits, upstream overload and timeouts are weather, not verdicts:
        # a transient 429/5xx used to fail the whole stage instantly and cost a
        # full simulated week of safe-continuation. Backed-off retries make one
        # bad minute cost one bad minute.
        last_error: Exception | None = None
        for transport_attempt in range(4):
            if transport_attempt:
                await asyncio.sleep(10.0 * (2 ** (transport_attempt - 1)))
            try:
                if self.client is not None:
                    response = await self.client.post(
                        self.endpoint, headers=headers, json=payload
                    )
                else:
                    async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                        response = await client.post(
                            self.endpoint, headers=headers, json=payload
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
                raise TypeError("OpenRouter response must be a JSON object")
            if "choices" not in data:
                # OpenRouter reports provider-side failures as HTTP 200 with an
                # error body. That is weather too — it burned whole simulated
                # weeks by raising KeyError past every retry.
                last_error = RuntimeError(
                    "OpenRouter error body: " + json.dumps(data.get("error", data))[:300]
                )
                continue
            self._usage["http_calls"] += 1
            self._accumulate_usage(data.get("usage"))
            return data
        raise (
            last_error
            if last_error is not None
            else RuntimeError("OpenRouter transport retries exhausted")
        )

    def _accumulate_usage(self, usage: object) -> None:
        """Fold one response's reported usage in, tolerating providers that omit it."""

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

    @staticmethod
    def _extract_content(response: dict[str, Any]) -> str:
        content = response["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise TypeError("OpenRouter response content must be a non-empty string")
        return content
