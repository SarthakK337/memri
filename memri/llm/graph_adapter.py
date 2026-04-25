"""Adapter: wraps Memri's BaseLLMProvider to speak the graph service interface.

The graph services (ingestion, retrieval, layer0, reflection) call:
  await llm.generate(prompt, system="")         -> str
  await llm.generate_json(prompt, system="")    -> dict | list
  await llm.generate_cached(static, dynamic)    -> str   (no real caching; falls through)
  await llm.generate_json_cached(static, dynamic) -> dict | list
  await llm.extract_entities(query)             -> list[str]

This adapter maps those calls to Memri's BaseLLMProvider.complete().
"""

from __future__ import annotations

import json
import re
from typing import Any

from .provider import BaseLLMProvider, LLMResponse


def _parse_json(text: str) -> Any:
    """Robustly extract JSON from LLM response (handles markdown code fences)."""
    text = text.strip()
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find first JSON object or array
    for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return {}


class GraphLLMAdapter:
    """Wraps Memri's BaseLLMProvider to match the interface expected by graph services."""

    def __init__(self, provider: BaseLLMProvider, model: str):
        self._provider = provider
        self.model = model
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "calls": 0,
        }

    async def generate(self, prompt: str, system: str = "") -> str:
        resp: LLMResponse = await self._provider.complete(
            system_prompt=system,
            user_message=prompt,
            max_tokens=8192,
        )
        self.usage["prompt_tokens"] += resp.input_tokens
        self.usage["completion_tokens"] += resp.output_tokens
        self.usage["calls"] += 1
        return resp.content

    async def generate_json(self, prompt: str, system: str = "") -> Any:
        text = await self.generate(prompt, system)
        return _parse_json(text)

    async def generate_cached(self, static: str, dynamic: str) -> str:
        """No server-side caching — concatenate and call normally."""
        return await self.generate(f"{static}\n\n{dynamic}")

    async def generate_json_cached(self, static: str, dynamic: str) -> Any:
        text = await self.generate_cached(static, dynamic)
        return _parse_json(text)

    async def extract_entities(self, query: str) -> list[str]:
        prompt = (
            "Extract named entities (people, places, organizations) from this query. "
            "Respond as a JSON array of strings only.\n\nQuery: " + query
        )
        result = await self.generate_json(prompt)
        if isinstance(result, list):
            return [str(e) for e in result]
        return []

    def reset_usage(self) -> dict:
        snap = dict(self.usage)
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "calls": 0}
        return snap

    def usage_snapshot(self) -> dict:
        return dict(self.usage)
