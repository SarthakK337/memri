"""Codex (OpenAI) session ingestion (stub)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..core.memory import MemriMemory


async def ingest_all_sessions(
    memory: MemriMemory,
    sessions_dir: Optional[Path] = None,
) -> dict[str, int]:
    """Placeholder for Codex session ingestion."""
    return {}
