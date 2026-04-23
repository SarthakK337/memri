"""Tests for Reflector agent."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from memri.core.reflector import Reflector


def _mock_provider(response_text: str):
    provider = AsyncMock()
    resp = MagicMock()
    resp.content = response_text
    resp.input_tokens = 100
    resp.output_tokens = 40
    resp.model = "test"
    provider.complete = AsyncMock(return_value=resp)
    return provider


@pytest.mark.asyncio
async def test_reflector_returns_cleaned_text():
    cleaned = "Date: 2026-04-23\n- 🔴 14:00 Core decision retained"
    provider = _mock_provider(cleaned)
    reflector = Reflector(provider)

    raw_obs = (
        "Date: 2026-04-22\n"
        "- 🔴 10:00 Old debugging step (resolved)\n"
        "- 🟡 10:05 Explored approach A (abandoned)\n"
        "Date: 2026-04-23\n"
        "- 🔴 14:00 Core decision retained\n"
    )

    result, inp, out = await reflector.reflect(
        observations=raw_obs,
        current_date=datetime(2026, 4, 23, 15, 0),
    )

    assert result == cleaned
    assert inp == 100
    assert out == 40


@pytest.mark.asyncio
async def test_reflector_passes_observations_to_llm():
    provider = _mock_provider("cleaned")
    reflector = Reflector(provider)

    await reflector.reflect(
        observations="Date: 2026-04-23\n- 🔴 important note",
        current_date=datetime(2026, 4, 23),
    )

    call_args = provider.complete.call_args
    user_message = call_args.kwargs.get("user_message") or call_args.args[1]
    assert "important note" in user_message
