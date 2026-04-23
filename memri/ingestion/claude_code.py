"""Claude Code session ingestion — reads ~/.claude/projects/ and ingest conversations."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..core.memory import MemriMemory


def _find_session_files(projects_dir: Path) -> list[Path]:
    """Return all JSONL session files under the Claude Code projects directory."""
    return list(projects_dir.rglob("*.jsonl"))


def _load_session(path: Path) -> list[dict]:
    """Parse a JSONL conversation file into a list of message dicts."""
    messages: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        messages.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except (OSError, UnicodeDecodeError):
        pass
    return messages


def _extract_text_content(content) -> str:
    """Extract text from a message content field (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    inner = block.get("content", "")
                    parts.append(_extract_text_content(inner))
        return "\n".join(p for p in parts if p)
    return ""


async def ingest_session(
    memory: MemriMemory,
    session_path: Path,
    agent_type: str = "claude-code",
) -> int:
    """Ingest all messages from a single session file into memory.

    Returns the number of messages ingested.
    """
    raw_messages = _load_session(session_path)
    if not raw_messages:
        return 0

    # Use the session file path as a stable thread ID
    thread_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(session_path)))
    project_path = str(session_path.parent)

    memory.ensure_thread(thread_id, agent_type, project_path)

    count = 0
    for raw in raw_messages:
        msg_type = raw.get("type", "")

        # Claude Code VSCode extension JSONL format:
        # {"type":"user","message":{"role":"user","content":[{"type":"text","text":"..."}]}}
        # {"type":"assistant","message":{"role":"assistant","content":[...]}}
        if msg_type == "user":
            role = "user"
            content_raw = raw.get("message", {}).get("content", "")
        elif msg_type == "assistant":
            role = "assistant"
            content_raw = raw.get("message", {}).get("content", "")
        else:
            # Skip queue-operation, attachment, summary, etc.
            continue

        content = _extract_text_content(content_raw)
        if not content.strip():
            continue

        # Parse timestamp from the JSONL entry
        ts_str = raw.get("timestamp")
        created_at = None
        if ts_str:
            try:
                created_at = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).replace(tzinfo=None)
            except ValueError:
                pass

        token_count = memory.token_counter.count_text(content)
        from ..storage.base import Message as _Msg
        msg = _Msg(
            id=raw.get("uuid") or str(uuid.uuid4()),
            thread_id=thread_id,
            role=role,
            content=content,
            token_count=token_count,
            observed=False,
            created_at=created_at or datetime.now(),
        )
        memory.store.save_message(msg)
        count += 1

    return count


async def ingest_all_sessions(
    memory: MemriMemory,
    projects_dir: Optional[Path] = None,
) -> dict[str, int]:
    """Ingest all Claude Code sessions found under the projects directory.

    Returns a dict mapping session path → messages ingested.
    """
    if projects_dir is None:
        projects_dir = Path.home() / ".claude" / "projects"

    if not projects_dir.exists():
        return {}

    results: dict[str, int] = {}
    for session_file in _find_session_files(projects_dir):
        count = await ingest_session(memory, session_file)
        if count > 0:
            results[str(session_file)] = count

    return results
