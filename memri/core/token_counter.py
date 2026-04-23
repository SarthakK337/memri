"""Fast local token estimation using tiktoken."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..storage.base import Message


class TokenCounter:
    """Counts tokens locally without any API calls."""

    def __init__(self, model: str = "gpt-4o"):
        try:
            import tiktoken
            try:
                self._enc = tiktoken.encoding_for_model(model)
            except KeyError:
                self._enc = tiktoken.get_encoding("cl100k_base")
            self._use_tiktoken = True
        except ImportError:
            self._use_tiktoken = False

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        if self._use_tiktoken:
            return len(self._enc.encode(text))
        # Rough fallback: ~4 chars per token
        return max(1, len(text) // 4)

    def count_messages(self, messages: list) -> int:
        total = 0
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content", "")
            elif hasattr(msg, "content"):
                content = msg.content
            else:
                content = str(msg)

            if isinstance(content, str):
                total += self.count_text(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        total += self.count_text(block.get("text", ""))
        return total
