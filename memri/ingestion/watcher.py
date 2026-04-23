"""File watcher — auto-ingest new Claude Code (and Cursor/Codex) sessions as they happen."""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from watchdog.events import FileCreatedEvent, FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer


class _SessionHandler(FileSystemEventHandler):
    """Watchdog handler that queues changed JSONL files for ingestion."""

    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        self._queue = queue
        self._loop = loop
        self._seen: set[str] = set()

    def _enqueue(self, path: str) -> None:
        if path.endswith(".jsonl"):
            asyncio.run_coroutine_threadsafe(self._queue.put(path), self._loop)

    def on_created(self, event):
        if not event.is_directory:
            self._enqueue(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._enqueue(event.src_path)


class SessionWatcher:
    """Watch coding-agent session directories and ingest new messages in real time."""

    def __init__(
        self,
        memory,
        paths: list[Path],
        on_ingest: Optional[Callable[[str, int], None]] = None,
    ):
        self._memory = memory
        self._paths = [p for p in paths if p.exists()]
        self._on_ingest = on_ingest  # callback(session_path, messages_added)
        self._observer: Optional[Observer] = None

    async def run_forever(self) -> None:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str] = asyncio.Queue()

        handler = _SessionHandler(queue, loop)
        self._observer = Observer()
        for path in self._paths:
            self._observer.schedule(handler, str(path), recursive=True)
        self._observer.start()

        try:
            while True:
                path_str = await queue.get()
                await self._process(path_str)
        finally:
            self._observer.stop()
            self._observer.join()

    async def _process(self, path_str: str) -> None:
        from .claude_code import ingest_session  # noqa: PLC0415

        path = Path(path_str)
        if not path.exists() or path.suffix != ".jsonl":
            return

        # Determine agent type from path
        agent_type = "claude-code"
        if "cursor" in path_str.lower():
            agent_type = "cursor"
        elif "codex" in path_str.lower():
            agent_type = "codex"

        try:
            added = await ingest_session(self._memory, path, agent_type=agent_type)
            if added and self._on_ingest:
                self._on_ingest(path_str, added)
        except Exception:
            pass  # silently skip malformed files


def default_watch_paths() -> list[Path]:
    """Return the default set of paths to watch based on installed agents."""
    candidates = [
        Path.home() / ".claude" / "projects",   # Claude Code
        Path.home() / ".cursor" / "conversations",
        Path.home() / ".codex" / "sessions",
    ]
    return [p for p in candidates if p.exists()]
