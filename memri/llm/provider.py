"""LLM provider abstraction — Anthropic, Google Gemini, OpenAI-compatible, and passive (no-key) mode."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

# Status codes that are safe to retry (transient server-side issues)
_RETRYABLE_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 8
_RETRY_BASE_DELAY = 2.0  # seconds, doubles each attempt
# 429 quota exhaustion needs a long wait before retrying
_QUOTA_DELAY = 60.0


def _error_code(exc: Exception) -> Optional[int]:
    """Extract HTTP status code from openai or google-genai exceptions."""
    # openai.APIStatusError
    code = getattr(exc, "status_code", None)
    if code:
        return int(code)
    # google-genai wraps the code in the message string e.g. "503 UNAVAILABLE"
    msg = str(exc)
    for c in _RETRYABLE_CODES:
        if str(c) in msg[:10]:
            return c
    return None


async def _with_retry(call):
    """Run an async callable with exponential backoff on transient errors."""
    delay = _RETRY_BASE_DELAY
    for attempt in range(_MAX_RETRIES):
        try:
            return await call()
        except Exception as exc:
            code = _error_code(exc)
            if code not in _RETRYABLE_CODES or attempt == _MAX_RETRIES - 1:
                raise
            # 429 quota exhaustion needs a longer initial wait than transient errors
            wait = _QUOTA_DELAY if code == 429 else delay
            print(f"  [retry {attempt+1}/{_MAX_RETRIES}] HTTP {code}, waiting {wait:.0f}s...")
            await asyncio.sleep(wait)
            delay = min(delay * 2, 120)


@dataclass
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int
    model: str


class BaseLLMProvider(ABC):

    @abstractmethod
    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        model: Optional[str] = None,
        max_tokens: int = 8192,
    ) -> LLMResponse: ...


class AnthropicProvider(BaseLLMProvider):
    """Uses the Anthropic SDK directly."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "claude-haiku-4-5-20251001",
    ):
        import anthropic  # noqa: PLC0415
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.default_model = default_model

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        model: Optional[str] = None,
        max_tokens: int = 8192,
    ) -> LLMResponse:
        async def _call():
            return await self.client.messages.create(
                model=model or self.default_model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
        response = await _with_retry(_call)
        return LLMResponse(
            content=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
        )


class GeminiProvider(BaseLLMProvider):
    """Google Gemini via the native google-genai SDK.

    Install: pip install google-genai
    Set GOOGLE_API_KEY or GEMINI_API_KEY in the environment.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gemini-2.5-flash",
    ):
        try:
            from google import genai  # noqa: PLC0415
            self.client = genai.Client(api_key=api_key)
        except ImportError as e:
            raise ImportError(
                "google-genai not installed. Run: pip install google-genai"
            ) from e
        self.default_model = default_model

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        model: Optional[str] = None,
        max_tokens: int = 8192,
    ) -> LLMResponse:
        from google.genai import types as gtypes  # noqa: PLC0415

        async def _call():
            return await self.client.aio.models.generate_content(
                model=model or self.default_model,
                contents=user_message,
                config=gtypes.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=max_tokens,
                ),
            )

        response = await _with_retry(_call)
        return LLMResponse(
            content=response.text or "",
            input_tokens=response.usage_metadata.prompt_token_count or 0,
            output_tokens=response.usage_metadata.candidates_token_count or 0,
            model=model or self.default_model,
        )


class OpenAICompatibleProvider(BaseLLMProvider):
    """Works with OpenAI, Groq, Together, Mistral, Ollama, and any OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: str = "gpt-4o-mini",
    ):
        from openai import AsyncOpenAI  # noqa: PLC0415
        self.client = AsyncOpenAI(api_key=api_key or "not-needed", base_url=base_url)
        self.default_model = default_model

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        model: Optional[str] = None,
        max_tokens: int = 8192,
    ) -> LLMResponse:
        async def _call():
            return await self.client.chat.completions.create(
                model=model or self.default_model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
        response = await _with_retry(_call)
        return LLMResponse(
            content=response.choices[0].message.content or "",
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=response.model,
        )


class PassiveProvider(BaseLLMProvider):
    """No-API-key provider for users without LLM credentials.

    LLM-dependent features (compression, reflection, strategy extraction)
    are silently skipped. Storage, recall, search, and explicit notes still work.
    """

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        model: Optional[str] = None,
        max_tokens: int = 8192,
    ) -> LLMResponse:
        raise NotImplementedError("passive mode: no LLM configured")
