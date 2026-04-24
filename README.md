# memri

**Persistent memory for AI coding agents — in a `pip install`.**

[![PyPI](https://img.shields.io/pypi/v/memri)](https://pypi.org/project/memri/)
[![Python](https://img.shields.io/pypi/pyversions/memri)](https://pypi.org/project/memri/)
[![CI](https://github.com/SarthakK337/memri/actions/workflows/ci.yml/badge.svg)](https://github.com/SarthakK337/memri/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## The problem

Every time you start a new session with Claude Code, Cursor, or Codex — it forgets everything. The architecture you designed last week. The library you chose. The bug you already fixed. You repeat yourself. The agent repeats mistakes.

**memri fixes this.** It runs silently alongside your coding agent, compresses your conversation history into dense observations, and injects them at the start of every new session. Your agent picks up exactly where it left off.

---

## Features

- **Automatic compression** — conversations beyond 30K tokens are compressed 5–40× into timestamped observations
- **Cross-session recall** — observations are injected at the start of every new session, no setup needed per conversation
- **Semantic search** — find anything from past sessions with natural language (`memri_search "auth pattern we chose"`)
- **Procedural memory** *(v0.2)* — learns *how to work better with you* over time, not just *what happened*
- **Frustration detection** *(v0.2)* — detects when you're frustrated and permanently stores what went wrong as a high-priority strategy
- **Works with any LLM** — Anthropic, Gemini, OpenAI, or any OpenAI-compatible endpoint
- **100% local** — SQLite database on your machine, no cloud, no accounts

---

## Install

```bash
pip install memri
```

With semantic search:

```bash
pip install "memri[embeddings]"
```

---

## Quick start

One command wires memri into Claude Code:

```bash
memri init --claude-code
```

This does three things automatically:
1. Creates `~/.memri/memri.db` (local SQLite database)
2. Registers memri as an MCP server in Claude Code's `settings.json`
3. Writes the recall instruction to `~/.claude/CLAUDE.md`

That's it. Open a new Claude Code session — memri starts working.

---

## How it works

```
Your conversation
      │
      ▼
  [Observer]      triggers at 30K tokens
      │           compresses turns → timestamped observations (5–40× smaller)
      │
  [Strategist]    runs on every message                           (v0.2)
      │           detects frustration → 🔴 stores "what went wrong"
      │           end of session → distills "what worked" → 🟡🔵 strategies
      ▼
 [SQLiteStore]    ~/.memri/memri.db  (observations + strategies + embeddings)
      │
      ▼
  [Reflector]     triggers at 40K observation tokens
      │           garbage-collects stale / redundant observations
      ▼
 [get_context()]  prepends strategies → then observations → then recent turns
      │
      ▼
 Injected at the top of your next session
```

**Two types of memory:**

| Type | What it stores | Example |
|------|---------------|---------|
| Episodic | What happened in past sessions | *"User chose PostgreSQL over SQLite on 2026-04-10"* |
| Procedural | How to work better with this user | *"Always confirm before running destructive commands"* |

---

## MCP tools

memri exposes 7 tools to your coding agent via MCP:

| Tool | When to call |
|------|-------------|
| `memri_recall` | Start of every session — restores compressed context |
| `memri_store` | User shares something important to remember |
| `memri_search` | Looking for context from a different project or thread |
| `memri_ingest` | Manually process a session into memory |
| `memri_distill` | End of session — extract generalizable strategies |
| `memri_status` | Check token savings, cost, session stats |
| `memri_forget` | Delete memories for a specific thread |

---

## Configuration

Config at `~/.memri/config.json`:

```json
{
  "llm_provider": "gemini",
  "llm_model": "gemini-2.5-flash",
  "observe_threshold": 30000,
  "reflect_threshold": 40000
}
```

API keys at `~/.memri/.env`:

```bash
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
```

Supported providers: `anthropic`, `claude-code-auth`, `gemini`, `gemini-adc`, `openai`, `openai-compatible` (Groq, Ollama, Together, Mistral), `passive` (no key needed).

### Don't have an API key?

**Option 1 — Use your Claude subscription** *(zero setup if you already use Claude Code)*
If you've ever run `claude` in your terminal, memri automatically detects your credentials and uses them. No API key needed — your Claude Pro / Max / Team subscription covers it.

```bash
memri init --claude-code   # auto-detects Claude login, configures instantly
```

**Option 2 — Use your Google account** *(zero setup if you use gcloud)*
If you've run `gcloud auth application-default login`, memri detects those credentials automatically. Works with any Google account.

```bash
gcloud auth application-default login   # one-time, if not already done
memri init --claude-code                # auto-detects Google credentials
```

**Option 3 — Free Gemini API** *(takes 1 minute)*
Google's Gemini 2.0 Flash has a permanently free tier — no credit card, no trial.
Get a key in 1 minute: [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

```bash
# ~/.memri/.env
GEMINI_API_KEY=your-key-here
```
```json
// ~/.memri/config.json
{ "llm_provider": "gemini", "llm_model": "gemini-2.0-flash" }
```

**Option 4 — Local model via Ollama** *(fully private)*
Run any open-source model on your own hardware.

```bash
# Install from ollama.ai, then:
ollama pull llama3
```
```json
// ~/.memri/config.json
{
  "llm_provider": "openai-compatible",
  "llm_base_url": "http://localhost:11434/v1",
  "llm_model": "llama3"
}
```

**Option 5 — Passive mode** *(zero setup)*
No API key, no compression. memri stores sessions locally and returns recent context directly. `memri_recall`, `memri_store`, and `memri_search` all work.

```json
// ~/.memri/config.json
{ "llm_provider": "passive" }
```

> **What about OpenAI / ChatGPT?**
> A ChatGPT subscription ($20/mo) does not include API access — OpenAI sells these separately. If you use GPT models, you need an OpenAI API key (`OPENAI_API_KEY`). The free options above (Gemini ADC or free Gemini API) are typically easier.

---

## CLI

```bash
memri init --claude-code   # First-time setup (one command)
memri status               # Token savings, cost, session count
memri watch                # Auto-ingest new sessions in real time
memri ingest               # Ingest existing session history
memri observe              # Manually run the Observer on all threads
memri embed                # Build semantic search index
memri dashboard            # Web dashboard at http://localhost:8050
memri config               # View / edit config
```

---

## Benchmarks

Evaluated on [LongMemEval-S](https://arxiv.org/abs/2410.10813) — 500 QA pairs across 6 question types designed to test AI assistant long-term memory.

**Raw baseline** (full context → Gemini 2.5 Flash):

| Question type | Score |
|---|---|
| Single-session (user) | ~95% |
| Single-session (assistant) | ~90% |
| Knowledge update | ~82% |
| Temporal reasoning | ~65% |
| Preference | ~55% |
| Multi-session | ~50% |
| **Overall** | **70.6%** |

This is the ceiling for the compressed-context path. Smriti integration (in progress) targets **80%+**.

---

## Comparison

| | **memri** | Mastra OM | mem0 | Full context |
|---|---|---|---|---|
| Language | Python | TypeScript | Python | — |
| Install | `pip install` | framework lock-in | `pip install` | — |
| Works with | Claude Code, Cursor, Codex | Mastra only | any | any |
| Storage | local SQLite | cloud | cloud | none |
| Procedural memory | ✅ v0.2 | ❌ | ❌ | ❌ |
| Frustration detection | ✅ v0.2 | ❌ | ❌ | ❌ |
| Semantic search | ✅ local | ❌ | ✅ cloud | ❌ |
| Dashboard | ✅ | ❌ | ✅ | ❌ |
| Token compression | 5–40× | 5–40× | varies | 1× |
| **Privacy** | **100% local** | cloud | cloud | local |

---

## Privacy

**Your data never leaves your machine.**

- Conversation history and observations live in `~/.memri/memri.db` — a local SQLite file only you can read.
- memri has no servers, no telemetry, no accounts.
- The only external calls are to your LLM provider (the same one your coding agent already uses) to run compression.
- API keys are read from environment variables and never written to the database.

```bash
memri status               # see exactly what's stored
memri forget <thread_id>   # delete a specific thread
rm ~/.memri/memri.db       # delete everything
```

---

## Development

```bash
git clone https://github.com/SarthakK337/memri
cd memri
pip install -e ".[dev,embeddings]"
pytest
```

Contributions welcome. Open an issue before starting large changes.

---

## License

MIT © 2026
