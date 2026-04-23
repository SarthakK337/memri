# memri

**Mastra-level memory intelligence in a `pip install`.**

Observational memory for coding agents (Claude Code, Cursor, Codex) — a standalone Python package and MCP server that keeps your AI assistant from forgetting context across sessions.

[![PyPI](https://img.shields.io/pypi/v/memri)](https://pypi.org/project/memri/)
[![Python](https://img.shields.io/pypi/pyversions/memri)](https://pypi.org/project/memri/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What it does

Every time your conversation grows large, memri silently compresses it into dense, timestamped observations and injects them at the start of your next session. Your AI coding assistant remembers what you worked on — without burning tokens re-reading the full history.

Two background agents run as you code:

- **Observer** — when your conversation exceeds 30K tokens, compresses it into observations (5–40× compression ratio)
- **Reflector** — when observations exceed 40K tokens, garbage-collects stale or redundant ones

The result: a compact, prompt-cacheable memory block at the top of every session.

---

## Install

```bash
pip install memri
```

For semantic search across past sessions:

```bash
pip install "memri[embeddings]"
```

---

## Quick start

### 1. Initialize

```bash
memri init
```

This creates `~/.memri/config.json` and prompts for your API key. Supports Gemini, Anthropic, and any OpenAI-compatible endpoint.

### 2. Connect to your coding agent

**Claude Code:**
```bash
memri init --claude-code
```
Adds memri as an MCP server to your Claude Code config automatically.

**Cursor / VS Code:**
```bash
memri init --cursor
```

**Manual MCP config:**
```json
{
  "mcpServers": {
    "memri": {
      "command": "memri",
      "args": ["mcp-server"]
    }
  }
}
```

### 3. Start the MCP server

```bash
memri mcp-server
```

---

## Configuration

Config lives at `~/.memri/config.json`:

```json
{
  "llm_provider": "gemini",
  "llm_model": "gemini-2.5-flash",
  "observe_threshold": 30000,
  "reflect_threshold": 40000
}
```

API keys in `~/.memri/.env`:

```
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
```

Override model via env or CLI:

```bash
memri mcp-server --model gemini-2.5-flash
memri mcp-server --model claude-haiku-4-5-20251001
memri mcp-server --model gpt-4o-mini
```

---

## MCP Tools

| Tool | What it does |
|------|-------------|
| `memri_recall` | Restore compressed context at session start |
| `memri_store` | Explicitly save a note or decision |
| `memri_search` | Semantic search across all past sessions |
| `memri_status` | Show token savings, cost savings, session stats |
| `memri_forget` | Delete all memories for a thread |
| `memri_ingest` | Manually ingest a session file |

Add this to your Claude Code system prompt to activate auto-recall:

```
At the start of each session, call memri_recall to restore context.
After significant decisions or discoveries, call memri_store to save them.
```

---

## Dashboard

```bash
memri dashboard
# Open http://localhost:8050
```

Shows: total tokens saved, cost savings per model, observation counts, and session history.

---

## CLI reference

```bash
memri init                  # First-time setup
memri init --claude-code    # Setup + wire into Claude Code
memri status                # Quick stats
memri mcp-server            # Start MCP server
memri dashboard             # Start web dashboard
memri observe <thread_id>   # Manually trigger Observer
memri embed                 # Build/update semantic search index
memri watch                 # Start file watcher (auto-ingest sessions)
memri ingest <file>         # Ingest a session JSONL file
memri config                # Show current config
```

---

## How it compares

| | memri | Mastra OM | Full context |
|---|---|---|---|
| Language | Python | TypeScript | any |
| Install | `pip install` | framework lock-in | — |
| Coding agents | Claude Code, Cursor, Codex | Mastra agents only | any |
| Cross-session search | yes (semantic) | no | no |
| Dashboard | yes | no | no |
| Token compression | 5–40x | 5–40x | 1x |
| LongMemEval-S accuracy | 80%+ | 80%+ | 80%+ |

---

## Architecture

```
Your conversation
      |
      v
  [Observer]  — triggers at 30K tokens
      |          compresses turns → observations
      v
 [SQLiteStore] — stores observations + embeddings
      |
      v
  [Reflector] — triggers at 40K observations tokens
      |          removes stale/redundant observations
      v
 [get_context()] — returns compact memory block
      |
      v
 Injected at top of next session
```

Storage: `~/.memri/memory.db` (SQLite, 5 tables).

---

## Benchmarks

Validated on [LongMemEval-S](https://arxiv.org/abs/2410.10813) — 500 QA pairs across 6 question types testing AI assistant memory:

| Question type | Accuracy |
|---|---|
| single-session-user | 97%+ |
| single-session-assistant | 94%+ |
| knowledge-update | 84%+ |
| multi-session | 50%+ |
| **Overall** | **80%+** |

---

## Development

```bash
git clone https://github.com/your-org/memri
cd memri
pip install -e ".[dev,embeddings]"
pytest
```

---

## License

MIT
