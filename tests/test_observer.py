"""Tests for Observer agent."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from memri.core.observer import Observer
from memri.storage.base import Message


def _mock_provider(response_text: str):
    provider = AsyncMock()
    resp = MagicMock()
    resp.content = response_text
    resp.input_tokens = 50
    resp.output_tokens = 20
    resp.model = "test"
    provider.complete = AsyncMock(return_value=resp)
    return provider


@pytest.mark.asyncio
async def test_observer_returns_text():
    expected = "Date: 2026-04-23\n- 🔴 14:00 User building FastAPI app"
    provider = _mock_provider(expected)
    observer = Observer(provider)

    messages = [
        Message(
            id="1", thread_id="t1", role="user",
            content="I'm building a FastAPI app", token_count=10,
            created_at=datetime(2026, 4, 23, 14, 0)
        ),
        Message(
            id="2", thread_id="t1", role="assistant",
            content="Sure, let me help", token_count=5,
            created_at=datetime(2026, 4, 23, 14, 1)
        ),
    ]

    result, inp, out = await observer.observe(
        messages=messages,
        existing_observations="",
        current_date=datetime(2026, 4, 23, 14, 5),
    )

    assert result == expected
    assert inp == 50
    assert out == 20
    assert provider.complete.called


@pytest.mark.asyncio
async def test_observer_includes_existing_observations():
    provider = _mock_provider("new obs")
    observer = Observer(provider)

    await observer.observe(
        messages=[Message("1", "t1", "user", "test", 5)],
        existing_observations="Date: 2026-04-22\n- 🔴 old note",
        current_date=datetime(2026, 4, 23),
    )

    call_args = provider.complete.call_args
    user_message = call_args.kwargs.get("user_message") or call_args.args[1]
    assert "old note" in user_message


@pytest.mark.asyncio
async def test_observer_formats_message_roles():
    provider = _mock_provider("obs")
    observer = Observer(provider)

    messages = [
        Message("1", "t1", "user", "user msg", 5),
        Message("2", "t1", "assistant", "assistant reply", 5),
        Message("3", "t1", "tool", "tool output", 5),
    ]

    await observer.observe(messages=messages, existing_observations="",
                           current_date=datetime(2026, 4, 23))
    call_args = provider.complete.call_args
    user_message = call_args.kwargs.get("user_message") or call_args.args[1]
    assert "USER" in user_message
    assert "ASSISTANT" in user_message
