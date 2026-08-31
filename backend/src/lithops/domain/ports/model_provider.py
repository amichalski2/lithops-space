from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)


class StructuredModelProvider(Protocol):
    """Provider-neutral contract for schema-validated model output."""

    @property
    def model_id(self) -> str: ...

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[StructuredOutput],
    ) -> StructuredOutput: ...
