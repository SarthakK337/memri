"""Observer agent — compresses raw messages into dense observations."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from .prompts import OBSERVER_SYSTEM_PROMPT, OBSERVER_USER_TEMPLATE

if TYPE_CHECKING:
    from ..llm.provider import BaseLLMProvider
    from ..storage.base import Message


def _format_messages(messages: list[Message]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg.role.upper()
        ts = msg.created_at.strftime("%Y-%m-%d %H:%M") if msg.created_at else ""
        header = f"[{role}]" + (f" {ts}" if ts else "")
        parts.append(f"{header}\n{msg.content}")
    return "\n\n---\n\n".join(parts)


class Observer:

    def __init__(self, provider: BaseLLMProvider):
        self._provider = provider

    async def observe(
        self,
        messages: list[Message],
        existing_observations: str,
        current_date: datetime,
    ) -> tuple[str, int, int]:
        """Compress messages into observations.

        Returns:
            (observation_text, input_tokens, output_tokens)
        """
        user_message = OBSERVER_USER_TEMPLATE.format(
            current_date=current_date.strftime("%Y-%m-%d %H:%M"),
            existing_observations=existing_observations or "(none yet)",
            messages=_format_messages(messages),
        )

        response = await self._provider.complete(
            system_prompt=OBSERVER_SYSTEM_PROMPT,
            user_message=user_message,
            max_tokens=16384,
        )

        return response.content, response.input_tokens, response.output_tokens
