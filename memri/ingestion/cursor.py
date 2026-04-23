"""Cursor session ingestion (stub — expand as Cursor storage format is documented)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..core.memory import MemriMemory


async def ingest_all_sessions(
    memory: MemriMemory,
    sessions_dir: Optional[Path] = None,
) -> dict[str, int]:
    """Placeholder for Cursor session ingestion.

    Cursor stores conversation history in SQLite databases inside its extension dir.
    The exact schema varies by Cursor version. Contribute the implementation at:
    https://github.com/your-org/memri
    """
    return {}
