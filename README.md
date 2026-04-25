# memri

**Persistent, graph-based memory for Claude Code, Cursor, and Codex — in a `pip install`.**

[![PyPI](https://img.shields.io/pypi/v/memri)](https://pypi.org/project/memri/)
[![Python](https://img.shields.io/pypi/pyversions/memri)](https://pypi.org/project/memri/)
[![CI](https://github.com/SarthakK337/memri/actions/workflows/ci.yml/badge.svg)](https://github.com/SarthakK337/memri/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://img.shields.io/pypi/dm/memri)](https://pypi.org/project/memri/)

---

## The problem

Every time you start a new session with Claude Code, Cursor, or Codex — it forgets everything. The architecture you designed last week. The library you chose. The bug you already fixed. You repeat yourself. The agent repeats mistakes.

**memri fixes this.** It gives your AI coding agent a persistent memory that survives across sessions — so it always knows who you are, what you're building, and how you like to work.

---

## Results

Evaluated on [LongMemEval-S](https://arxiv.org/abs/2410.10813) — 500 QA pairs designed to test AI long-term memory:

| | Score |
|---|---|
| Full context (no memory, 115K tokens) | 70.6% |
| **memri v1.0 graph memory** | **83%** |

Better recall. A fraction of the tokens.

---

## Install

```bash
pip install "memri[graph]"
```

One command to wire it into Claude Code:

```bash
memri init --claude-code
```

That's it. Open a new Claude Code session — memri starts working immediately.

---

## What it does

Every conversation you have with your coding agent gets ingested into a 3-layer memory graph:

```
Your conversation
      │
  [Graph Engine]
      │
      ├── Layer 2  raw episode archive  (SQLite, zero data loss)
      │
      ├── Layer 1  fact/entity/reflection graph  (NetworkX)
      │            causal chains · temporal edges · entity linking
      │
      └── Layer 0  always-in-context routing index  (~500 tokens)
                   entity index · topic clusters · user summary
      │
  [Retrieval]   Layer 0 → BFS traversal → RRF ranking
      │          returns only the relevant facts (~500 tokens)
      │
  Injected at the top of your next session
```

When you ask a question, memri doesn't dump your entire history into the prompt. It finds the specific facts, entities, and context that matter for that query — and injects only those.

---

## Features

- **Graph-based memory** — entities, facts, causal chains, and higher-level reflections stored in a queryable graph
- **Entity tracking** — people, projects, and concepts linked across all sessions
- **Three-layer architecture** — always-in-context index (Layer 0), fact graph (Layer 1), raw episode archive (Layer 2)
- **RRF ranking** — Reciprocal Rank Fusion across vector, BM25, importance, and recency signals
- **Automatic compression** — conversations beyond 30K tokens compressed 5–40× into timestamped observations
- **Cross-session recall** — memory injected at the start of every session automatically
- **Semantic search** — find anything from past sessions (`memri search "auth pattern we chose"`)
- **Procedural memory** — learns *how to work with you* over time, not just *what happened*
- **Frustration detection** — detects when you're frustrated, permanently stores what went wrong
- **Works with any LLM** — Anthropic, Gemini, OpenAI, or any OpenAI-compatible endpoint
- **100% local** — all data on your machine, no cloud, no accounts
- **Visual dashboard** — interactive graph visualization, Layer 0 index, episode browser

---

## Quick start

```bash
# Install
pip install "memri[graph]"

# Wire into Claude Code (one command)
memri init --claude-code

# Open a new Claude Code session — memri is already running
```

### Don't have an API key?

**Use your existing Claude subscription** — if you've run `claude` in your terminal, memri detects your credentials automatically. No API key needed.

```bash
memri init --claude-code   # auto-detects Claude login
```

**Use your Google account** — if you've run `gcloud auth application-default login`:

```bash
memri init --claude-code   # auto-detects gcloud credentials
```

**Free Gemini API** — [get a key in 1 minute](https://aistudio.google.com/apikey), no credit card:

```bash
# ~/.memri/.env
GEMINI_API_KEY=your-key-here
```

**Passive mode** — no API key, no compression, still works:

```json
{ "llm_provider": "passive" }
```

---

## How it works

### Three layers of memory

| Layer | What it stores | Size |
|-------|---------------|------|
| Layer 0 | Entity index, topic clusters, user summary — always in context | ~500 tokens |
| Layer 1 | Fact/entity/reflection graph with causal and temporal edges | grows with sessions |
| Layer 2 | Raw episode archive — zero data loss, full session text | cold storage |

### Three types of memory

| Type | What it stores | Example |
|------|---------------|---------|
| Episodic | What happened in past sessions | *"Chose PostgreSQL over SQLite on 2026-04-10"* |
| Procedural | How to work better with this user | *"Always confirm before running destructive commands"* |
| Graph | Entity relationships and causal chains | *"Deadline stress caused repeated tool failures"* |

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
  "memory_engine": "graph",
  "observe_threshold": 30000
}
```

Supported providers: `anthropic`, `claude-code-auth`, `gemini`, `gemini-adc`, `openai`, `openai-compatible` (Groq, Ollama, Together, Mistral), `passive`.

---

## CLI

```bash
memri init --claude-code   # First-time setup
memri status               # Token savings, cost, session count
memri watch                # Auto-ingest new sessions in real time
memri ingest               # Ingest existing session history
memri observe              # Run Observer on all threads
memri embed                # Build semantic search index
memri dashboard            # Web dashboard at http://localhost:8050
memri config               # View / edit config
```

---

## Benchmarks

Evaluated on [LongMemEval-S](https://arxiv.org/abs/2410.10813) — 500 QA pairs across 6 question types designed to test AI assistant long-term memory.

| Question type | Raw baseline | memri v1.0 graph |
|---|---|---|
| Single-session (user) | ~95% | ~97% |
| Single-session (assistant) | ~90% | ~93% |
| Knowledge update | ~82% | ~88% |
| Temporal reasoning | ~65% | ~76% |
| Preference | ~55% | ~72% |
| Multi-session | ~50% | ~74% |
| **Overall** | **70.6%** | **83%** |

**Raw baseline**: full ~115K token conversation passed directly to Gemini 2.5 Flash.
**memri v1.0 graph**: sessions ingested into the 3-layer graph, top-k facts retrieved per query (~500 tokens). Better accuracy, 200× fewer tokens.

---

## Comparison

| | **memri** | Mastra OM | mem0 | Full context |
|---|---|---|---|---|
| Language | Python | TypeScript | Python | — |
| Install | `pip install` | framework lock-in | `pip install` | — |
| Works with | Claude Code, Cursor, Codex | Mastra only | any | any |
| Storage | local SQLite + graph | cloud | cloud | none |
| Graph-based memory | ✅ v1.0 | ❌ | ❌ | ❌ |
| Entity tracking | ✅ v1.0 | ❌ | partial | ❌ |
| Causal chains | ✅ v1.0 | ❌ | ❌ | ❌ |
| Procedural memory | ✅ v0.2 | ❌ | ❌ | ❌ |
| Frustration detection | ✅ v0.2 | ❌ | ❌ | ❌ |
| Semantic search | ✅ local | ❌ | ✅ cloud | ❌ |
| Dashboard | ✅ | ❌ | ✅ | ❌ |
| Token compression | 200× (graph retrieval) | 5–40× | varies | 1× |
| LongMemEval-S accuracy | **83%** | — | — | 70.6% |
| **Privacy** | **100% local** | cloud | cloud | local |

---

## Privacy

**Your data never leaves your machine.**

- Conversation history and memory live in `~/.memri/` — local files only you can read
- No servers, no telemetry, no accounts
- The only external calls are to your LLM provider (the same one your coding agent already uses)
- API keys are read from environment variables and never written to the database

```bash
memri status               # see exactly what's stored
memri forget <thread_id>   # delete a specific thread
rm -rf ~/.memri/           # delete everything
```

---

## Development

```bash
git clone https://github.com/SarthakK337/memri
cd memri
pip install -e ".[dev,graph,embeddings]"
pytest
```

Contributions welcome. Open an issue before starting large changes.

---

## License

MIT © 2026
