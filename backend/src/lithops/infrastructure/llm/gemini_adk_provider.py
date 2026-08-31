from __future__ import annotations

from copy import deepcopy
from typing import Any
from uuid import uuid4

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import ValidationError

from lithops.domain.ports.model_provider import StructuredOutput
from lithops.infrastructure.llm.openrouter_provider import StructuredGenerationError


def gemini_developer_schema(output_schema: type[StructuredOutput]) -> dict[str, Any]:
    """Return a Gemini Developer API-compatible copy of a Pydantic schema.

    Pydantic emits ``additionalProperties: false`` for ``extra='forbid'``.
    The generateContent response_schema path used by ADK rejects that keyword,
    so it is removed only from the provider copy. The original Pydantic model
    remains strict and validates the final response after ADK returns it.
    """

    schema = deepcopy(output_schema.model_json_schema())

    def remove_unsupported_keywords(value: Any) -> None:
        if isinstance(value, dict):
            value.pop("additionalProperties", None)
            value.pop("additional_properties", None)
            value.pop("default", None)
            value.pop("pattern", None)
            value.pop("minLength", None)
            value.pop("maxLength", None)
            value.pop("minItems", None)
            value.pop("maxItems", None)
            for nested in value.values():
                remove_unsupported_keywords(nested)
        elif isinstance(value, list):
            for nested in value:
                remove_unsupported_keywords(nested)

    remove_unsupported_keywords(schema)
    definitions = schema.get("$defs")
    if isinstance(definitions, dict):
        for name in tuple(definitions):
            if definitions[name] == {}:
                definitions.pop(name)
    return schema


class GeminiAdkProvider:
    """Gemini structured generation executed through Google ADK."""

    app_name = "lithops"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-3.7-flash",
        agent_name: str = "lithops_executive",
        agent_description: str = "Selects one bounded weekly company action plan.",
    ) -> None:
        if not api_key:
            raise ValueError("Gemini API key is required")
        self.api_key = api_key
        self.model = model
        self.agent_name = agent_name
        self.agent_description = agent_description
        # Same accounting as the OpenRouter path, minus a price: ADK reports tokens
        # but not what they cost.
        self._usage = {
            "logical_calls": 0,
            "http_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "validation_retries": 0,
        }

    def usage_snapshot(self) -> dict[str, float]:
        return dict(self._usage)

    def _accumulate_usage(self, usage_metadata: object) -> None:
        """Fold one event's token report in; events without one are ignored."""

        if usage_metadata is None:
            return
        for attribute, field in (
            ("prompt_token_count", "prompt_tokens"),
            ("candidates_token_count", "completion_tokens"),
        ):
            value = getattr(usage_metadata, attribute, None)
            if isinstance(value, int | float):
                self._usage[field] += value

    @property
    def model_id(self) -> str:
        return self.model

    def build_agent(
        self,
        *,
        system_prompt: str,
        output_schema: type[StructuredOutput],
    ) -> LlmAgent:
        return LlmAgent(
            name=self.agent_name,
            description=self.agent_description,
            model=Gemini(
                model=self.model,
                client_kwargs={"api_key": self.api_key},
            ),
            instruction=system_prompt,
            output_schema=gemini_developer_schema(output_schema),
            output_key="executive_decision",
            tools=[],
            include_contents="none",
        )

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[StructuredOutput],
    ) -> StructuredOutput:
        prompt = user_prompt
        last_error: ValidationError | None = None
        self._usage["logical_calls"] += 1
        for attempt in range(2):
            final_text = await self._generate_text(
                system_prompt=system_prompt,
                user_prompt=prompt,
                output_schema=output_schema,
            )
            try:
                return output_schema.model_validate_json(final_text)
            except ValidationError as exc:
                last_error = exc
                self._usage["validation_retries"] += 1
                if attempt == 0:
                    prompt = (
                        user_prompt + "\n\nThe prior response failed strict local validation. "
                        "Return the entire "
                        "corrected response. Prior invalid response:\n"
                        + final_text[:50_000]
                        + "\nValidation errors:\n"
                        + str(exc)[:4_000]
                    )
                    continue
        raise StructuredGenerationError(
            f"Google ADK returned invalid structured output twice for model {self.model}"
        ) from last_error

    async def _generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[StructuredOutput],
    ) -> str:
        agent = self.build_agent(system_prompt=system_prompt, output_schema=output_schema)
        session_service = InMemorySessionService()
        session_id = uuid4().hex
        user_id = self.agent_name
        await session_service.create_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id,
        )
        runner = Runner(
            app_name=self.app_name,
            agent=agent,
            session_service=session_service,
        )
        final_text: str | None = None
        message = types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])
        self._usage["http_calls"] += 1
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=message,
        ):
            self._accumulate_usage(getattr(event, "usage_metadata", None))
            if event.is_final_response() and event.content and event.content.parts:
                final_text = "".join(
                    part.text for part in event.content.parts if part.text and not part.thought
                )
        if not final_text:
            raise StructuredGenerationError(
                f"Google ADK returned no final structured output for model {self.model}"
            )
        return final_text
