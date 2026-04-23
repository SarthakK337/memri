"""Reflector agent — garbage-collects stale observations."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from .prompts import REFLECTOR_SYSTEM_PROMPT, REFLECTOR_USER_TEMPLATE

if TYPE_CHECKING:
    from ..llm.provider import BaseLLMProvider


class Reflector:

    def __init__(self, provider: BaseLLMProvider):
        self._provider = provider

    async def reflect(
        self,
        observations: str,
        current_date: datetime,
    ) -> tuple[str, int, int]:
        """Garbage-collect observations.

        Returns:
            (cleaned_observations, input_tokens, output_tokens)
        """
        user_message = REFLECTOR_USER_TEMPLATE.format(
            current_date=current_date.strftime("%Y-%m-%d %H:%M"),
            observations=observations,
        )

        response = await self._provider.complete(
            system_prompt=REFLECTOR_SYSTEM_PROMPT,
            user_message=user_message,
        )

        return response.content, response.input_tokens, response.output_tokens
