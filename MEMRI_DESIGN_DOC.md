# MEMRI — Design Document v1.0
## "Mastra-level memory intelligence in a pip install"

**Date:** April 23, 2026  
**Goal:** Reimplement Mastra's Observational Memory architecture as a standalone Python pip package + MCP server for coding agents (Claude Code, Cursor, Codex, etc.)

---

## 1. WHAT TO STUDY FROM MASTRA'S REPO

### Repository Structure
```
github.com/mastra-ai/mastra
├── packages/memory/                    # ← MAIN TARGET
│   └── src/
│       └── processors/
│           └── observational-memory/
│               └── observational-memory.ts   # Core OM logic
├── packages/core/
│   └── src/
│       ├── agent/agent.ts              # How agent integrates memory
│       └── memory/
│           ├── memory.ts               # Base memory class
│           └── types.ts                # Memory type definitions
├── docs/src/content/en/docs/memory/
│   ├── observational-memory.mdx        # OM documentation
│   └── memory-processors.mdx          # Processor chain docs
└── mastracode/                         # Their CLI coding agent
```

### Key Files to Read (Priority Order)
1. `packages/memory/src/processors/observational-memory/observational-memory.ts` — The core algorithm
2. `docs/src/content/en/docs/memory/observational-memory.mdx` — Architecture explanation
3. `packages/core/src/memory/types.ts` — Data structures
4. `packages/core/src/agent/agent.ts` — How memory hooks into the agent loop
5. `integrations/opencode/src/index.ts` — How they built a standalone plugin (PR #12925)

### Key Constants & Prompts to Extract
- `OBSERVATION_CONTEXT_PROMPT` — The system prompt for the Observer agent
- `OBSERVATION_CONTINUATION_HINT` — Hint injected for continuation
- Observer threshold: 30,000 tokens (default)
- Reflector threshold: 40,000 tokens (default)
- Default model: `google/gemini-2.5-flash`
- Emoji priority system: 🔴 (critical), 🟡 (notable), etc.

---

## 2. MASTRA'S OBSERVATIONAL MEMORY — HOW IT WORKS

### Architecture (from their research + docs)
```
┌────────────────────────────────────────────────────┐
│                   CONTEXT WINDOW                     │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │ System Prompt                                  │   │
│  └──────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────┐   │
│  │ Observations (dense, compressed history)       │   │
│  │                                                │   │
│  │ Date: 2026-01-15                              │   │
│  │ - 🔴 12:10 User building Next.js + Supabase   │   │
│  │ - 🔴 12:15 App name is "Acme Dashboard"       │   │
│  │ - 🟡 12:20 Chose PostgreSQL over MongoDB      │   │
│  │                                                │   │
│  │ Date: 2026-01-16                              │   │
│  │ - 🔴 09:00 Auth migration started             │   │
│  │ - 🔴 09:30 Decided on JWT + refresh tokens    │   │
│  │                                                │   │
│  │ [This block GROWS over time, is CACHEABLE]     │   │
│  └──────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────┐   │
│  │ Recent Messages (raw, uncompressed)            │   │
│  │ [User]: Can you add rate limiting?             │   │
│  │ [Assistant]: Sure, I'll use express-rate-limit  │   │
│  │ [User]: Use Redis instead of in-memory         │   │
│  └──────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────┘
```

### The Two Background Agents

#### OBSERVER (runs when messages exceed 30K tokens)
- Takes: Raw message history
- Produces: Dense text observations with timestamps + emoji priorities
- Format: Plain text, NOT JSON, NOT knowledge graph
- Three-date model:
  - `observation_date` — when the observation was recorded
  - `referenced_date` — date mentioned in conversation  
  - `relative_date` — "1 week from now" resolved to absolute date
- Compression ratio: 5-40x
- Also tracks: current task + suggested next response

#### REFLECTOR (runs when observations exceed 40K tokens)
- Takes: Accumulated observations
- Produces: Garbage-collected observations (removes stale/irrelevant ones)
- Combines related observations
- Discards what no longer matters
- Runs infrequently — only when observations grow too large

### Why This Beats Knowledge Graphs
- Text is the universal interface for LLMs
- No schema mismatch — LLM reads text, not graph queries
- Prompt-cacheable — observation block stays stable across turns
- Easier to debug — just read the text
- 84.23% on LongMemEval vs SuperMemory's 81.6% (which uses knowledge graphs)

---

## 3. MEMRI — PYTHON REIMPLEMENTATION DESIGN

### Package Structure
```
memri/
├── pyproject.toml
├── README.md
├── memri/
│   ├── __init__.py
│   ├── cli.py                  # CLI: memri init, memri status, memri dashboard
│   ├── config.py               # Configuration management
│   ├── core/
│   │   ├── __init__.py
│   │   ├── observer.py         # Observer agent — compresses messages
│   │   ├── reflector.py        # Reflector agent — garbage collects
│   │   ├── memory.py           # Main memory manager
│   │   ├── token_counter.py    # Fast local token estimation
│   │   └── prompts.py          # All LLM prompts (observer, reflector)
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── sqlite_store.py     # SQLite storage (default, zero-config)
│   │   └── base.py             # Abstract storage interface
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── claude_code.py      # Auto-ingest Claude Code sessions
│   │   ├── cursor.py           # Auto-ingest Cursor sessions
│   │   ├── codex.py            # Auto-ingest Codex sessions
│   │   └── auto_detect.py      # Detect which agent is running
│   ├── mcp/
│   │   ├── __init__.py
│   │   └── server.py           # MCP server (the main interface)
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── provider.py         # LLM abstraction (any OpenAI-compatible)
│   │   └── cost_tracker.py     # Track LLM costs for dashboard
│   └── dashboard/
│       ├── __init__.py
│       └── app.py              # Local web dashboard (FastAPI + simple HTML)
└── tests/
    ├── test_observer.py
    ├── test_reflector.py
    ├── test_memory.py
    └── benchmarks/
        ├── run_longmemeval.py  # LongMemEval-S benchmark runner
        └── run_locomo.py       # LoCoMo benchmark runner
```

### Core Algorithm (Python Pseudocode)

```python
# memri/core/memory.py

class MemriMemory:
    def __init__(self, config: MemriConfig):
        self.store = SQLiteStore(config.db_path)  # ~/.memri/memri.db
        self.observer = Observer(config.llm_provider)
        self.reflector = Reflector(config.llm_provider)
        self.token_counter = TokenCounter()  # Local, no API needed
        
        # Thresholds (configurable)
        self.observe_threshold = config.observe_threshold  # 30_000 tokens
        self.reflect_threshold = config.reflect_threshold  # 40_000 tokens
    
    async def process_message(self, thread_id: str, message: Message):
        """Called for every new message in a conversation."""
        # 1. Store raw message
        self.store.save_message(thread_id, message)
        
        # 2. Check if we need to observe (compress)
        raw_messages = self.store.get_messages(thread_id)
        raw_token_count = self.token_counter.count(raw_messages)
        
        if raw_token_count > self.observe_threshold:
            await self._run_observer(thread_id, raw_messages)
    
    async def _run_observer(self, thread_id: str, messages: list[Message]):
        """Compress messages into dense observations."""
        existing_observations = self.store.get_observations(thread_id)
        
        # Observer LLM call (cheap model: GPT-4o-mini, Haiku, Gemini Flash)
        new_observations = await self.observer.observe(
            messages=messages,
            existing_observations=existing_observations,
            current_date=datetime.now()
        )
        
        # Store observations, clear processed messages
        self.store.append_observations(thread_id, new_observations)
        self.store.mark_messages_observed(thread_id)
        
        # Check if reflector needs to run
        obs_token_count = self.token_counter.count(
            self.store.get_observations(thread_id)
        )
        if obs_token_count > self.reflect_threshold:
            await self._run_reflector(thread_id)
    
    async def _run_reflector(self, thread_id: str):
        """Garbage-collect stale observations."""
        observations = self.store.get_observations(thread_id)
        
        cleaned = await self.reflector.reflect(
            observations=observations,
            current_date=datetime.now()
        )
        
        self.store.replace_observations(thread_id, cleaned)
    
    def get_context(self, thread_id: str) -> str:
        """Build context window for a new agent turn."""
        observations = self.store.get_observations(thread_id)
        recent_messages = self.store.get_recent_messages(
            thread_id, 
            max_tokens=self.observe_threshold
        )
        
        return f"""## Memory (Observations from past sessions)
{observations}

## Recent Conversation
{self._format_messages(recent_messages)}"""
    
    # CROSS-SESSION: Search across ALL threads
    def search_across_sessions(self, query: str, top_k: int = 5) -> str:
        """Search observations across all threads.
        This is what Mastra DOESN'T do well — our addition."""
        all_observations = self.store.get_all_observations()
        # Use embeddings for semantic search across observation blocks
        results = self._semantic_search(query, all_observations, top_k)
        return results
```

### Observer Prompt (Inspired by Mastra, rewritten)

```python
# memri/core/prompts.py

OBSERVER_SYSTEM_PROMPT = """You are an Observer — a background process that watches 
conversations between a developer and their AI coding assistant.

Your job: compress the conversation into dense, factual observations.

## Output Format
Write observations grouped by date, with timestamps and priority emojis:
- 🔴 Critical: Decisions, requirements, constraints, user preferences
- 🟡 Notable: Questions asked, approaches discussed, tools mentioned  
- ⚪ Context: Background info that might be useful later

## Date Model
For each observation, track:
- The date/time it was observed
- Any dates referenced ("deploy by Friday" → resolve to actual date)
- Relative dates ("last week" → resolve to actual date range)

## Rules
1. Be factual, not interpretive
2. Preserve technical details (library names, versions, config values)
3. Resolve pronouns ("it" → "the PostgreSQL migration")
4. Track current task and what the user wants next
5. Never discard user preferences or decisions
6. Compress tool output/code blocks to their essence
7. If user corrects earlier info, note the UPDATE

## Example Output
Date: 2026-04-23
- 🔴 14:10 User building auth system for "memri" project using FastAPI + SQLite
- 🔴 14:12 Decision: Use JWT tokens with 24h expiry, refresh tokens in httponly cookies
- 🟡 14:15 Explored bcrypt vs argon2 for password hashing, chose argon2
- 🔴 14:20 UPDATE: Changed from REST to MCP server architecture
- ⚪ 14:25 User mentioned they're applying to YC Startup School with this project

Current task: Implementing MCP server with memory tools
Suggested next: Set up the MCP tool definitions for store/recall/search
"""

REFLECTOR_SYSTEM_PROMPT = """You are a Reflector — you review accumulated observations 
and clean them up.

Your job: Remove stale, redundant, or superseded observations while preserving 
everything that's still relevant.

## Rules
1. If an observation was UPDATED later, keep only the latest version
2. Combine related observations into single dense entries
3. Remove observations about completed tasks that have no future relevance
4. ALWAYS keep: user preferences, architectural decisions, project constraints
5. ALWAYS keep: anything marked 🔴
6. Aggressively remove: debugging steps that were resolved, exploration that was abandoned
7. Output in the same format (date-grouped, emoji-prioritized)
"""
```

### MCP Server (The Main Interface)

```python
# memri/mcp/server.py

"""
MCP Tools exposed to Claude Code / Cursor / any MCP client:

1. memri_recall    — Get relevant context for current conversation
2. memri_store     — Explicitly store a piece of information  
3. memri_search    — Search across all past sessions
4. memri_status    — Show memory stats (tokens saved, sessions, health)
5. memri_forget    — Remove specific memories
"""

# Tool: memri_recall
# Called automatically at start of each session
# Returns: observations from this thread + relevant cross-session context

# Tool: memri_search  
# Called when agent needs to find something from past sessions
# Input: query string
# Returns: relevant observation snippets from across all threads

# Tool: memri_status
# Called when user asks about memory
# Returns: { threads: 42, observations: 1,847, tokens_saved: 2.4M, 
#            cost_saved: "$12.40", oldest_memory: "2026-01-15" }
```

### Auto-Detection & Ingestion

```python
# memri/ingestion/auto_detect.py

"""
Claude Code stores sessions in: ~/.claude/conversations/
Cursor stores in: varies by OS
Codex stores in: ~/.codex/sessions/

On `memri init`:
1. Detect which coding agents are installed
2. Set up file watchers / hooks for auto-ingestion
3. Configure MCP server in the agent's config

On `memri init --claude-code`:
1. Read ~/.claude/settings.json
2. Add memri MCP server to mcpServers config
3. Set up post-session hook to ingest conversation
"""

CLAUDE_CODE_MCP_CONFIG = {
    "mcpServers": {
        "memri": {
            "command": "memri",
            "args": ["mcp-server"],
            # No URL, no API key, no auth — local process
        }
    }
}
```

### Dashboard

```python
# memri/dashboard/app.py

"""
Local dashboard at http://localhost:8050

Pages:
1. Overview — Total sessions, observations, tokens saved, cost saved
2. Timeline — Chronological view of all observations
3. Sessions — List of all threads with observation summaries  
4. Search — Full-text + semantic search across all memory
5. Settings — Configure thresholds, LLM provider, auto-detect

Key metrics shown:
- Tokens saved (observations vs raw history)
- Cost saved ($)
- Memory health (observation count, reflection frequency)
- Top topics/projects detected
"""
```

---

## 4. STORAGE SCHEMA (SQLite)

```sql
-- Main tables

CREATE TABLE threads (
    id TEXT PRIMARY KEY,
    agent_type TEXT,          -- 'claude-code', 'cursor', 'codex'
    project_path TEXT,        -- Project directory
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    metadata JSON
);

CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    thread_id TEXT REFERENCES threads(id),
    role TEXT,                -- 'user', 'assistant', 'system', 'tool'
    content TEXT,
    token_count INTEGER,
    observed BOOLEAN DEFAULT FALSE,  -- Has this been compressed?
    created_at TIMESTAMP
);

CREATE TABLE observations (
    id TEXT PRIMARY KEY,
    thread_id TEXT REFERENCES threads(id),
    content TEXT,             -- Dense text observations
    token_count INTEGER,
    version INTEGER DEFAULT 1,  -- Incremented on reflection
    created_at TIMESTAMP,
    reflected_at TIMESTAMP    -- Last time reflector processed this
);

-- For cross-session search
CREATE TABLE observation_embeddings (
    observation_id TEXT REFERENCES observations(id),
    embedding BLOB,           -- Vector embedding for semantic search
    chunk_text TEXT,           -- The text chunk that was embedded
    created_at TIMESTAMP
);

-- Cost tracking for dashboard
CREATE TABLE llm_calls (
    id TEXT PRIMARY KEY,
    call_type TEXT,           -- 'observe', 'reflect', 'search'
    model TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    created_at TIMESTAMP
);

-- Token savings tracking
CREATE TABLE token_savings (
    thread_id TEXT REFERENCES threads(id),
    raw_tokens INTEGER,       -- What it would have cost without memri
    compressed_tokens INTEGER, -- What it actually cost with observations
    saved_tokens INTEGER,     -- raw - compressed
    saved_usd REAL,
    recorded_at TIMESTAMP
);
```

---

## 5. WHAT MEMRI ADDS BEYOND MASTRA

| Feature | Mastra OM | memri |
|---------|-----------|-------|
| Language | TypeScript | Python |
| Distribution | npm (framework-locked) | pip (standalone) |
| Works with | Mastra agents only | Any MCP client |
| Cross-session search | No (per-thread only) | Yes (semantic search across all threads) |
| Auto-ingestion | No (manual) | Yes (auto-detect Claude Code/Cursor) |
| Dashboard | No | Yes (tokens saved, costs, timeline) |
| Cost tracking | No | Yes (shows $ saved vs full context) |
| Benchmarks | LongMemEval only | LongMemEval + LoCoMo + MemoryStress |
| Team memory | No | Future v2 (git-based sharing) |

---

## 6. IMPLEMENTATION PLAN

### Phase 1: Core (Week 1-2)
- [ ] Set up Python package structure
- [ ] Implement SQLite storage
- [ ] Implement Observer agent (prompts + LLM call)
- [ ] Implement Reflector agent
- [ ] Implement MemriMemory core class
- [ ] Basic token counting (tiktoken)
- [ ] Run LongMemEval-S benchmark → get baseline score

### Phase 2: MCP + CLI (Week 3)
- [ ] Implement MCP server with 5 tools
- [ ] CLI: `memri init`, `memri status`, `memri serve`
- [ ] Auto-detect Claude Code installation
- [ ] Auto-configure MCP in Claude Code settings
- [ ] Test end-to-end: pip install → memri init → use in Claude Code

### Phase 3: Cross-Session Search (Week 4)
- [ ] Implement local embeddings (sentence-transformers or ONNX)
- [ ] Semantic search across observation blocks
- [ ] `memri_search` MCP tool
- [ ] Re-run LongMemEval-S → compare improvement

### Phase 4: Dashboard + Polish (Week 5)
- [ ] FastAPI dashboard at localhost:8050
- [ ] Token savings calculator
- [ ] Cost tracking
- [ ] Timeline view
- [ ] Settings page

### Phase 5: Benchmark + Launch (Week 6)
- [ ] Run full LongMemEval-S (target: 80%+ with GPT-4o)
- [ ] Run LoCoMo (beat current 66.5%)
- [ ] Write research page with results
- [ ] Publish to PyPI as `memri`
- [ ] Write README with benchmark comparison table
- [ ] Post on X, HN, Reddit

---

## 7. CLAUDE CODE INSTRUCTIONS

When implementing this with Claude Code, clone Mastra's repo first:

```bash
git clone https://github.com/mastra-ai/mastra.git ~/mastra-reference
```

Then read these files in order:
```bash
# 1. Understand the core OM algorithm
cat ~/mastra-reference/packages/memory/src/processors/observational-memory/observational-memory.ts

# 2. Understand the prompts
grep -r "OBSERVATION" ~/mastra-reference/packages/memory/src/ --include="*.ts" -l

# 3. Understand the types
cat ~/mastra-reference/packages/core/src/memory/types.ts

# 4. Understand the OpenCode plugin (standalone integration pattern)
cat ~/mastra-reference/integrations/opencode/src/index.ts

# 5. Understand token counting
grep -r "TokenCounter\|tokenCount\|meetsObservationThreshold" ~/mastra-reference/packages/memory/src/ --include="*.ts" -l
```

Key things to port from TypeScript to Python:
- `doSynchronousObservation()` → `observer.observe()`
- `maybeReflect()` → `reflector.reflect()`
- `meetsObservationThreshold()` → `memory.should_observe()`
- `optimizeObservationsForContext()` → `memory.get_context()`
- `OBSERVATION_CONTEXT_PROMPT` → `prompts.OBSERVER_SYSTEM_PROMPT`
- `OBSERVATION_CONTINUATION_HINT` → `prompts.CONTINUATION_HINT`

---

## 8. SUCCESS CRITERIA

1. **Benchmark:** ≥80% on LongMemEval-S with GPT-4o
2. **Install:** `pip install memri && memri init` works in <60 seconds  
3. **Zero config:** Works with Claude Code after `memri init` — no manual MCP config
4. **Cost proof:** Dashboard shows measurable token/cost savings
5. **PyPI:** Package name `memri` secured and published
6. **Adoption signal:** 100+ GitHub stars in first week

---

## 9. COMPETITIVE POSITIONING

```
"SuperMemory's benchmark scores. OMEGA's simplicity. 
 Mastra's architecture. Your pip install."
```

Target: Solo developers and small teams using Claude Code / Cursor daily.
Not targeting: Enterprise, B2B API, framework-locked developers.

Differentiator: The ONLY pip-installable, MCP-native memory system that 
achieves 80%+ on LongMemEval-S while showing you exactly how much money 
it saves through a built-in dashboard.
