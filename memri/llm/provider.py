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


class ClaudeCodeSubprocessProvider(BaseLLMProvider):
    """Uses the Claude Code CLI as a subprocess — no API key needed.

    Works for any user with the `claude` CLI installed and authenticated
    (Claude Pro / Max / Team / Free subscription). The CLI is invoked with
    `claude -p <prompt>` which runs non-interactively using the existing
    OAuth session.

    Note: this requires the *CLI* (`claude` in PATH), not just the VS Code
    extension. Install from https://claude.ai/code if not present.
    """

    def __init__(self, default_model: str = "claude-haiku-4-5-20251001"):
        import shutil
        self._claude_bin = shutil.which("claude")
        if not self._claude_bin:
            raise RuntimeError(
                "Claude Code CLI not found in PATH.\n"
                "Install from https://claude.ai/code, run 'claude' once to log in,\n"
                "then restart memri.\n"
                "If you only have the VS Code extension (not the CLI), use the free\n"
                "Gemini API instead: run 'memri auth login' to set it up."
            )
        self.default_model = default_model

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        model: Optional[str] = None,
        max_tokens: int = 8192,
    ) -> LLMResponse:
        import asyncio  # noqa: PLC0415

        full_prompt = f"{system_prompt}\n\n{user_message}" if system_prompt else user_message
        cmd = [self._claude_bin, "-p", full_prompt, "--output-format", "text"]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError("claude CLI timed out after 180s")

        if proc.returncode != 0:
            err = stderr.decode(errors="replace")[:300]
            raise RuntimeError(f"claude CLI exited {proc.returncode}: {err}")

        content = stdout.decode(errors="replace").strip()
        # Rough token estimates (no exact count available from CLI)
        input_tokens = len(full_prompt) // 4
        output_tokens = len(content) // 4

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model or self.default_model,
        )


# Keep the old name as an alias so existing configs with llm_provider="claude-code-auth"
# continue to work — they'll get a clear error message if the CLI isn't installed.
ClaudeCodeAuthProvider = ClaudeCodeSubprocessProvider


class GeminiADCProvider(BaseLLMProvider):
    """Uses Google Application Default Credentials — no API key needed.

    Works for any user with a Google account who has run:
        gcloud auth application-default login

    gcloud is free to install: https://cloud.google.com/sdk/install
    """

    def __init__(self, default_model: str = "gemini-2.0-flash"):
        try:
            import google.auth
            import google.auth.transport.requests
            self.credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/generative-language"]
            )
            # Refresh immediately so we catch auth errors at init, not at first call
            req = google.auth.transport.requests.Request()
            if not self.credentials.valid:
                self.credentials.refresh(req)
        except Exception as e:
            raise RuntimeError(
                "Google Application Default Credentials not found.\n"
                "Run once:  gcloud auth application-default login\n"
                "Install gcloud:  https://cloud.google.com/sdk/install"
            ) from e

        try:
            from google import genai  # noqa: PLC0415
            self.client = genai.Client(credentials=self.credentials)
        except ImportError as e:
            raise ImportError("google-genai not installed. Run: pip install google-genai") from e

        self.default_model = default_model

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        model: Optional[str] = None,
        max_tokens: int = 8192,
    ) -> LLMResponse:
        from google.genai import types as gtypes  # noqa: PLC0415

        # Refresh token if about to expire
        import google.auth.transport.requests
        if not self.credentials.valid:
            self.credentials.refresh(google.auth.transport.requests.Request())

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
