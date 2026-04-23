"""Auto-detect which coding agents are installed and configure memri for each."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Optional

CLAUDE_CODE_SETTINGS = Path.home() / ".claude" / "settings.json"
CURSOR_SETTINGS_PATHS = [
    Path.home() / ".cursor" / "settings.json",
    Path.home() / "AppData" / "Roaming" / "Cursor" / "User" / "settings.json",
    Path.home() / ".config" / "Cursor" / "User" / "settings.json",
]
CODEX_DIR = Path.home() / ".codex"


def _memri_command() -> tuple[str, list[str]]:
    """Return (command, args) for the MCP server entry.

    Prefers the full path to the running Python's memri script so the MCP
    server works even when no venv is active (e.g. Claude Code VSCode extension
    spawning it as a subprocess).
    """
    # Look for memri next to the current Python executable (same venv/install)
    python_dir = Path(sys.executable).parent
    for name in ("memri", "memri.exe", "memri.cmd"):
        candidate = python_dir / name
        if candidate.exists():
            return str(candidate), ["mcp-server"]

    # Fall back to PATH lookup
    found = shutil.which("memri")
    if found:
        return found, ["mcp-server"]

    # Last resort: run via current Python interpreter
    return sys.executable, ["-c", "from memri.mcp.server import run; run()"]


def detect_installed_agents() -> list[str]:
    """Return list of detected coding agent names."""
    agents: list[str] = []
    if CLAUDE_CODE_SETTINGS.exists():
        agents.append("claude-code")
    if any(p.exists() for p in CURSOR_SETTINGS_PATHS):
        agents.append("cursor")
    if CODEX_DIR.exists():
        agents.append("codex")
    return agents


def configure_claude_code(dry_run: bool = False) -> tuple[bool, str]:
    """Add memri MCP server to Claude Code settings.json.

    Returns (success, message).
    """
    settings_path = CLAUDE_CODE_SETTINGS
    if not settings_path.exists():
        return False, f"Claude Code settings not found at {settings_path}"

    with open(settings_path) as f:
        try:
            settings = json.load(f)
        except json.JSONDecodeError as e:
            return False, f"Could not parse settings.json: {e}"

    mcp_servers = settings.setdefault("mcpServers", {})

    if "memri" in mcp_servers:
        return True, "memri already configured in Claude Code settings."

    command, args = _memri_command()
    mcp_servers["memri"] = {"command": command, "args": args}

    if not dry_run:
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)

    return True, f"Added memri MCP server to {settings_path}\n  command: {command}"


def remove_claude_code_config() -> tuple[bool, str]:
    """Remove memri from Claude Code settings.json."""
    if not CLAUDE_CODE_SETTINGS.exists():
        return False, "Claude Code settings not found."

    with open(CLAUDE_CODE_SETTINGS) as f:
        settings = json.load(f)

    mcp_servers = settings.get("mcpServers", {})
    if "memri" not in mcp_servers:
        return True, "memri was not configured in Claude Code settings."

    del mcp_servers["memri"]

    with open(CLAUDE_CODE_SETTINGS, "w") as f:
        json.dump(settings, f, indent=2)

    return True, "Removed memri from Claude Code settings."


def get_claude_code_projects_dir() -> Optional[Path]:
    """Return the Claude Code projects directory for session ingestion."""
    projects_dir = Path.home() / ".claude" / "projects"
    if projects_dir.exists():
        return projects_dir
    return None
