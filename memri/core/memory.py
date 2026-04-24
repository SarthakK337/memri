"""MemriMemory — the main memory manager."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from ..config import MemriConfig
from ..llm.cost_tracker import calculate_cost, estimate_savings
from ..storage.base import Message, Thread
from ..storage.sqlite_store import SQLiteStore
from .embedder import Embedder
from .observer import Observer
from .reflector import Reflector
from .strategist import StrategistAgent
from .token_counter import TokenCounter


class MemriMemory:
    """
    Manages observational memory for a coding agent conversation.

    Usage:
        memory = MemriMemory(config)
        await memory.process_message(thread_id, message)
        context = memory.get_context(thread_id)
    """

    def __init__(self, config: Optional[MemriConfig] = None):
        self.config = config or MemriConfig.load()
        self.store = SQLiteStore(self.config.db_path)
        self.token_counter = TokenCounter()
        self.embedder = Embedder()
        self._provider = None  # lazy — only created when LLM calls needed
        self._observer = None
        self._reflector = None
        self._strategist = None

    def _get_provider(self):
        if self._provider is None:
            self._provider = self.config.get_llm_provider()
        return self._provider

    @property
    def observer(self) -> Observer:
        if self._observer is None:
            self._observer = Observer(self._get_provider())
        return self._observer

    @observer.setter
    def observer(self, value):
        self._observer = value

    @property
    def reflector(self) -> Reflector:
        if self._reflector is None:
            self._reflector = Reflector(self._get_provider())
        return self._reflector

    @reflector.setter
    def reflector(self, value):
        self._reflector = value

    @property
    def strategist(self) -> StrategistAgent:
        if self._strategist is None:
            self._strategist = StrategistAgent(self.store, self._get_provider())
        return self._strategist

    @strategist.setter
    def strategist(self, value):
        self._strategist = value

    # ──────────────────────── Thread management ────────────────────────

    def ensure_thread(
        self,
        thread_id: str,
        agent_type: str = "unknown",
        project_path: str = "",
    ) -> None:
        if not self.store.get_thread(thread_id):
            self.store.save_thread(
                Thread(
                    id=thread_id,
                    agent_type=agent_type,
                    project_path=project_path,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
            )

    # ────────────────────── Message processing ─────────────────────────

    async def process_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        agent_type: str = "unknown",
        project_path: str = "",
    ) -> bool:
        """Store a message and run Observer/Reflector if thresholds are exceeded.

        Returns True if an observation cycle ran.
        """
        self.ensure_thread(thread_id, agent_type, project_path)

        token_count = self.token_counter.count_text(content)
        message = Message(
            id=str(uuid.uuid4()),
            thread_id=thread_id,
            role=role,
            content=content,
            token_count=token_count,
            observed=False,
            created_at=datetime.now(),
        )
        self.store.save_message(message)

        # v0.2: check for frustration signals and extract strategies immediately
        await self.strategist.process_message(thread_id, role, content)

        # Check whether we should run the observer
        unobserved = self.store.get_messages(thread_id, unobserved_only=True)
        unobserved_tokens = sum(m.token_count for m in unobserved)

        if unobserved_tokens >= self.config.observe_threshold:
            try:
                await self._run_observer(thread_id, unobserved)
                return True
            except NotImplementedError:
                pass  # passive mode — skip compression

        return False

    # Process at most this many tokens per observer call (5-40x compression = 1.25K-10K output)
    _OBSERVER_BATCH_TOKENS = 50_000

    async def _run_observer(self, thread_id: str, messages: list[Message]) -> None:
        # Slice to a batch so the LLM isn't given 400K tokens at once.
        # Callers loop until all unobserved tokens are processed.
        batch: list[Message] = []
        batch_tokens = 0
        for msg in messages:
            if batch_tokens + msg.token_count > self._OBSERVER_BATCH_TOKENS and batch:
                break
            batch.append(msg)
            batch_tokens += msg.token_count

        if not batch:
            batch = messages[:1]  # always process at least one message
            batch_tokens = messages[0].token_count

        existing_obs = self.store.get_observation(thread_id)
        existing_content = existing_obs.content if existing_obs else ""

        new_content, inp_tok, out_tok = await self.observer.observe(
            messages=batch,
            existing_observations=existing_content,
            current_date=datetime.now(),
        )

        compressed_tokens = self.token_counter.count_text(new_content)
        saved_usd = estimate_savings(batch_tokens, compressed_tokens)
        model = self.config.llm_model
        cost = calculate_cost(model, inp_tok, out_tok)

        self.store.append_observations(thread_id, new_content, compressed_tokens)
        self.store.mark_messages_observed(thread_id, [m.id for m in batch])
        self.store.log_llm_call("observe", model, inp_tok, out_tok, cost)
        self.store.log_token_savings(thread_id, batch_tokens, compressed_tokens, saved_usd)

        # Reflector: run if accumulated observations are too large
        updated_obs = self.store.get_observation(thread_id)
        if updated_obs and updated_obs.token_count >= self.config.reflect_threshold:
            await self._run_reflector(thread_id)

    async def _run_reflector(self, thread_id: str) -> None:
        obs = self.store.get_observation(thread_id)
        if not obs:
            return

        cleaned, inp_tok, out_tok = await self.reflector.reflect(
            observations=obs.content,
            current_date=datetime.now(),
        )

        cleaned_tokens = self.token_counter.count_text(cleaned)
        model = self.config.llm_model
        cost = calculate_cost(model, inp_tok, out_tok)

        self.store.replace_observations(thread_id, cleaned, cleaned_tokens)
        self.store.log_llm_call("reflect", model, inp_tok, out_tok, cost)

    # ───────────────────────── Context building ────────────────────────

    def get_context(self, thread_id: str) -> str:
        """Build the memory context block to inject at the top of a new agent turn.

        v0.2: Strategies (procedural memory) are prepended before episodic observations.
        Critical strategies (from frustration) appear first.
        """
        parts: list[str] = []

        # 1. Strategies first — procedural "how to act" memory (v0.2)
        strategies = self.store.get_strategies(thread_id, limit=20)
        if strategies:
            lines = "\n".join(s["content"] for s in strategies)
            parts.append("## Strategies (How to act with this user)\n" + lines)

        # 2. Episodic observations — factual "what happened" memory
        obs = self.store.get_observation(thread_id)
        if obs and obs.content.strip():
            # Filter out strategy rows from the observations block (they're shown above)
            episodic_lines = [
                l for l in obs.content.splitlines()
                if "[STRATEGY/" not in l
            ]
            episodic = "\n".join(episodic_lines).strip()
            if episodic:
                parts.append("## Memory (Observations from past sessions)\n" + episodic)

        # 3. Recent messages — in passive mode show more since there's no compression
        max_tok = self.config.observe_threshold
        recent = self.store.get_recent_messages(thread_id, max_tokens=max_tok)
        if recent:
            formatted = "\n\n".join(
                f"[{m.role.upper()}]\n{m.content}" for m in recent
            )
            label = "## Recent Conversation" if parts else "## Conversation History"
            parts.append(f"{label}\n" + formatted)

        return "\n\n".join(parts)

    # ──────────────────────── Explicit store ───────────────────────────

    def store_note(self, thread_id: str, note: str) -> None:
        """Explicitly add a note to the observation block for a thread."""
        self.ensure_thread(thread_id)
        obs = self.store.get_observation(thread_id)
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M")
        new_line = f"\n- 🔴 {timestamp} [USER NOTE] {note}"

        if obs:
            self.store.replace_observations(
                thread_id,
                obs.content + new_line,
                self.token_counter.count_text(obs.content + new_line),
            )
        else:
            date_line = f"Date: {now.strftime('%Y-%m-%d')}"
            content = f"{date_line}{new_line}"
            self.store.append_observations(
                thread_id,
                content,
                self.token_counter.count_text(content),
            )

    # ──────────────────────── Cross-session search ─────────────────────

    def search(self, query: str, top_k: int = 5) -> str:
        """Search across all observation blocks.

        Uses semantic (vector) search when sentence-transformers is installed,
        falls back to keyword search otherwise.
        """
        all_obs = self.store.get_all_observations()
        if not all_obs:
            return "No memories found."

        if self.embedder.available:
            return self._semantic_search(query, all_obs, top_k)
        return self._keyword_search(query, all_obs, top_k)

    def _keyword_search(self, query: str, all_obs, top_k: int) -> str:
        q = query.lower()
        hits: list[str] = []
        for obs in all_obs:
            for line in obs.content.splitlines():
                if q in line.lower() and line.strip():
                    hits.append(line.strip())
                    if len(hits) >= top_k:
                        return "\n".join(hits)
        return "\n".join(hits) if hits else "No relevant memories found."

    def _semantic_search(self, query: str, all_obs, top_k: int) -> str:
        import numpy as np  # noqa: PLC0415

        # Build embeddings for any observation that doesn't have one yet
        for obs in all_obs:
            if not self.store.has_embedding(obs.id):
                vec = self.embedder.embed_one(obs.content)
                self.store.save_embedding(obs.id, obs.content, self.embedder.to_blob(vec))

        rows = self.store.get_all_embeddings()
        if not rows:
            return "No memories found."

        q_vec = self.embedder.embed_one(query)
        scored: list[tuple[float, str, str]] = []
        for row in rows:
            vec = self.embedder.from_blob(row["embedding"])
            sim = self.embedder.cosine_similarity(q_vec, vec)
            # Extract most relevant lines from the chunk
            best_lines = self._top_lines(row["chunk_text"], query, n=3)
            scored.append((sim, best_lines, row["thread_id"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[str] = []
        for sim, lines, tid in scored[:top_k]:
            results.append(f"[thread:{tid[:8]}] (score:{sim:.2f})\n{lines}")
        return "\n\n".join(results) if results else "No relevant memories found."

    @staticmethod
    def _top_lines(text: str, query: str, n: int) -> str:
        """Return the n lines from text most relevant to query (keyword proximity)."""
        q = query.lower()
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        scored = sorted(lines, key=lambda l: -sum(w in l.lower() for w in q.split()))
        return "\n".join(scored[:n])

    # ────────────────────── Batch observe ──────────────────────────────

    async def observe_all(self, progress_cb=None) -> dict[str, int]:
        """Run observer on every thread that has enough unobserved tokens.

        Returns {thread_id: cycles_run}.
        """
        results: dict[str, int] = {}
        threads = self.store.list_threads()

        for thread in threads:
            cycles = 0
            while True:
                unobserved = self.store.get_messages(thread.id, unobserved_only=True)
                tokens = sum(m.token_count for m in unobserved)
                if tokens < self.config.observe_threshold:
                    break
                await self._run_observer(thread.id, unobserved)
                cycles += 1
                if progress_cb:
                    progress_cb(thread.id, cycles, tokens)

            if cycles:
                results[thread.id] = cycles

        return results

    # ─────────────────── Embedding index (build/refresh) ───────────────

    def build_embeddings(self, progress_cb=None) -> int:
        """Embed all observation blocks that don't have an embedding yet.

        Returns number of observations embedded.
        """
        if not self.embedder.available:
            raise RuntimeError(
                "sentence-transformers not installed.\n"
                "Run: pip install 'memri[embeddings]'"
            )
        all_obs = self.store.get_all_observations()
        count = 0
        for obs in all_obs:
            if not self.store.has_embedding(obs.id):
                vec = self.embedder.embed_one(obs.content)
                self.store.save_embedding(obs.id, obs.content, self.embedder.to_blob(vec))
                count += 1
                if progress_cb:
                    progress_cb(obs.thread_id, count)
        return count

    # ──────────────────────── Forget / delete ──────────────────────────

    def forget_thread(self, thread_id: str) -> None:
        """Remove all memory for a thread (observations + messages)."""
        self.store.delete_thread(thread_id)

    # ──────────────────────────── Stats ────────────────────────────────

    def get_stats(self) -> dict:
        stats = self.store.get_stats()
        return {
            "threads": stats.total_threads,
            "messages": stats.total_messages,
            "observations": stats.total_observations,
            "tokens_saved": stats.total_tokens_saved,
            "cost_saved_usd": round(stats.total_cost_saved_usd, 4),
            "llm_cost_usd": round(stats.total_llm_cost_usd, 4),
            "oldest_memory": stats.oldest_memory.isoformat() if stats.oldest_memory else None,
            "newest_memory": stats.newest_memory.isoformat() if stats.newest_memory else None,
        }
