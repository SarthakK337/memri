"""Configuration management for memri."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .llm.provider import BaseLLMProvider

MEMRI_DIR = Path.home() / ".memri"
DEFAULT_DB_PATH = str(MEMRI_DIR / "memri.db")
DEFAULT_CONFIG_PATH = MEMRI_DIR / "config.json"


@dataclass
class MemriConfig:
    # Storage
    db_path: str = DEFAULT_DB_PATH

    # Thresholds (in tokens)
    observe_threshold: int = 30_000
    reflect_threshold: int = 40_000

    # LLM Provider
    llm_provider: str = "anthropic"          # 'anthropic' | 'gemini' | 'openai' | 'openai-compatible'
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None      # For openai-compatible / self-hosted providers

    # Embedding provider for cross-session search
    embedding_provider: str = "local"        # 'local' | 'openai'
    embedding_model: str = "all-MiniLM-L6-v2"

    # Dashboard
    dashboard_port: int = 8050
    dashboard_host: str = "127.0.0.1"

    # Ingestion
    auto_ingest: bool = True
    watch_claude_code: bool = True
    watch_cursor: bool = False
    watch_codex: bool = False

    @classmethod
    def load(cls) -> MemriConfig:
        config = cls()

        # Load ~/.memri/.env before anything else (lowest priority)
        _dotenv = MEMRI_DIR / ".env"
        if _dotenv.exists():
            with open(_dotenv) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        os.environ.setdefault(k.strip(), v.strip())

        if DEFAULT_CONFIG_PATH.exists():
            with open(DEFAULT_CONFIG_PATH) as f:
                data = json.load(f)
            for key, value in data.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Env var overrides (highest priority)
        _env_overrides = {
            "MEMRI_DB_PATH":         "db_path",
            "MEMRI_LLM_PROVIDER":    "llm_provider",
            "MEMRI_LLM_MODEL":       "llm_model",
            "MEMRI_LLM_BASE_URL":    "llm_base_url",
            "MEMRI_DASHBOARD_PORT":  "dashboard_port",
        }
        for env_key, attr in _env_overrides.items():
            if val := os.environ.get(env_key):
                setattr(config, attr, val)

        # API key: check provider-specific vars then generic
        if not config.llm_api_key:
            config.llm_api_key = (
                os.environ.get("MEMRI_LLM_API_KEY")
                or os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )

        # Auto-detect provider from model name if provider is still default
        if config.llm_provider == "anthropic" and config.llm_model.startswith("gemini"):
            config.llm_provider = "gemini"
        elif config.llm_provider == "anthropic" and config.llm_model.startswith(("gpt-", "o1", "o3")):
            config.llm_provider = "openai"

        return config

    def save(self) -> None:
        MEMRI_DIR.mkdir(parents=True, exist_ok=True)
        import dataclasses
        data = dataclasses.asdict(self)
        # Never persist resolved API keys — read them from env each time
        data.pop("llm_api_key", None)
        with open(DEFAULT_CONFIG_PATH, "w") as f:
            json.dump(data, f, indent=2)

    def get_llm_provider(self) -> BaseLLMProvider:
        from .llm.provider import (
            AnthropicProvider, ClaudeCodeAuthProvider, GeminiProvider,
            GeminiADCProvider, OpenAICompatibleProvider, PassiveProvider,
        )
        if self.llm_provider == "passive":
            return PassiveProvider()
        if self.llm_provider == "claude-code-auth":
            return ClaudeCodeAuthProvider(default_model=self.llm_model)
        if self.llm_provider == "gemini-adc":
            return GeminiADCProvider(default_model=self.llm_model)
        if self.llm_provider == "anthropic":
            return AnthropicProvider(
                api_key=self.llm_api_key,
                default_model=self.llm_model,
            )
        if self.llm_provider == "gemini":
            return GeminiProvider(
                api_key=self.llm_api_key,
                default_model=self.llm_model,
            )
        # 'openai' | 'openai-compatible' | any other value
        return OpenAICompatibleProvider(
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
            default_model=self.llm_model,
        )
