"""MCP server — exposes memri tools to Claude Code, Cursor, Codex, etc."""

from __future__ import annotations

import asyncio
from typing import Optional

from mcp.server.fastmcp import FastMCP

from ..config import MemriConfig
from ..core.memory import MemriMemory

_config = MemriConfig.load()
_memory = MemriMemory(_config)

mcp = FastMCP(
    "memri",
    instructions=(
        "Memri gives you persistent, cross-session memory. "
        "Call memri_recall at the start of each conversation to restore context. "
        "Call memri_store when the user shares something important to remember. "
        "Call memri_search to find something from a past session."
    ),
)


# ─────────────────────────── Tool: recall ──────────────────────────────


@mcp.tool()
async def memri_recall(
    thread_id: str,
    agent_type: str = "claude-code",
    project_path: str = "",
) -> str:
    """Retrieve memory context for the current conversation thread.

    Call this at the start of every session to restore continuity from past work.

    Args:
        thread_id: Unique identifier for this conversation thread.
        agent_type: Which coding agent is calling (claude-code, cursor, codex).
        project_path: Absolute path to the current project directory.

    Returns:
        Observation block + recent messages formatted for injection into context.
    """
    _memory.ensure_thread(thread_id, agent_type, project_path)
    context = _memory.get_context(thread_id)

    if not context.strip():
        return (
            "No memories yet for this thread. "
            "As you work, memri will automatically build up observations."
        )

    return context


# ─────────────────────────── Tool: store ───────────────────────────────


@mcp.tool()
def memri_store(
    thread_id: str,
    note: str,
) -> str:
    """Explicitly store an important piece of information in memory.

    Use this when the user shares something critical that should persist
    across sessions: architectural decisions, preferences, deadlines, etc.

    Args:
        thread_id: The current conversation thread ID.
        note: The information to store verbatim.

    Returns:
        Confirmation message.
    """
    _memory.store_note(thread_id, note)
    return f"Stored: {note}"


# ─────────────────────────── Tool: search ──────────────────────────────


@mcp.tool()
def memri_search(
    query: str,
    top_k: int = 5,
) -> str:
    """Search across all past sessions for relevant memories.

    Use this when you need context that might come from a different thread
    or project — e.g. the user's preferred auth pattern, a library they
    always use, or a decision they made in a previous project.

    Args:
        query: Natural language search query.
        top_k: Number of results to return (default 5).

    Returns:
        Relevant memory snippets from all past sessions.
    """
    return _memory.search(query, top_k=top_k)


# ─────────────────────────── Tool: status ──────────────────────────────


@mcp.tool()
def memri_status() -> str:
    """Show memory health stats: sessions, observations, tokens saved, costs.

    Returns:
        Formatted status report.
    """
    stats = _memory.get_stats()
    lines = [
        "## Memri Memory Status",
        f"- Threads: {stats['threads']}",
        f"- Messages: {stats['messages']}",
        f"- Observation blocks: {stats['observations']}",
        f"- Tokens saved: {stats['tokens_saved']:,}",
        f"- Cost saved (est.): ${stats['cost_saved_usd']:.4f}",
        f"- LLM cost (memri): ${stats['llm_cost_usd']:.4f}",
    ]
    if stats["oldest_memory"]:
        lines.append(f"- Oldest memory: {stats['oldest_memory'][:10]}")
    if stats["newest_memory"]:
        lines.append(f"- Newest memory: {stats['newest_memory'][:10]}")
    return "\n".join(lines)


# ─────────────────────────── Tool: forget ──────────────────────────────


@mcp.tool()
def memri_forget(
    thread_id: str,
    confirm: bool = False,
) -> str:
    """Remove all observations for a thread.

    Args:
        thread_id: The thread whose memories to delete.
        confirm: Must be True to actually delete (safety guard).

    Returns:
        Confirmation or safety message.
    """
    if not confirm:
        return (
            f"Safety check: pass confirm=True to actually delete memories "
            f"for thread '{thread_id}'."
        )
    _memory.forget_thread(thread_id)
    return f"Memories deleted for thread: {thread_id}"


# ─────────────────────── Ingest endpoint (internal) ────────────────────


@mcp.tool()
async def memri_ingest(
    thread_id: str,
    role: str,
    content: str,
    agent_type: str = "claude-code",
    project_path: str = "",
) -> str:
    """Ingest a new message into the memory pipeline.

    This is called by ingestion hooks (not typically by the user directly).

    Args:
        thread_id: Conversation thread identifier.
        role: Message role (user, assistant, system, tool).
        content: Message content.
        agent_type: Originating agent.
        project_path: Project directory.

    Returns:
        "observed" if an observation cycle ran, "stored" otherwise.
    """
    ran = await _memory.process_message(
        thread_id=thread_id,
        role=role,
        content=content,
        agent_type=agent_type,
        project_path=project_path,
    )
    return "observed" if ran else "stored"


def run() -> None:
    """Entry point: `memri mcp-server`.

    Starts the MCP server and a background file watcher so new Claude Code
    sessions are auto-ingested while the server is running.
    """
    import threading  # noqa: PLC0415
    from ..ingestion.watcher import SessionWatcher, default_watch_paths  # noqa: PLC0415

    watch_paths = default_watch_paths()
    if watch_paths:
        def _start_watcher():
            import asyncio  # noqa: PLC0415
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            watcher = SessionWatcher(_memory, watch_paths)
            loop.run_until_complete(watcher.run_forever())

        t = threading.Thread(target=_start_watcher, daemon=True)
        t.start()

    mcp.run()
