"""SQLite storage backend — zero-config, single-file database."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from .base import (
    BaseStore,
    LLMCallRecord,
    MemoryStats,
    Message,
    Observation,
    Thread,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    id           TEXT PRIMARY KEY,
    agent_type   TEXT DEFAULT 'unknown',
    project_path TEXT DEFAULT '',
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata     TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS messages (
    id          TEXT PRIMARY KEY,
    thread_id   TEXT NOT NULL REFERENCES threads(id),
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    token_count INTEGER DEFAULT 0,
    observed    INTEGER DEFAULT 0,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS observations (
    id               TEXT PRIMARY KEY,
    thread_id        TEXT NOT NULL REFERENCES threads(id),
    content          TEXT NOT NULL,
    token_count      INTEGER DEFAULT 0,
    version          INTEGER DEFAULT 1,
    observation_type TEXT DEFAULT 'observation',
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reflected_at     TIMESTAMP
);

-- v0.2: strategies table for procedural memory (ReasoningBank-style)
CREATE TABLE IF NOT EXISTS strategies (
    id         TEXT PRIMARY KEY,
    thread_id  TEXT NOT NULL REFERENCES threads(id),
    content    TEXT NOT NULL,
    priority   TEXT DEFAULT 'medium',
    source     TEXT DEFAULT 'unknown',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_strategies_thread ON strategies(thread_id, priority);

CREATE TABLE IF NOT EXISTS observation_embeddings (
    id             TEXT PRIMARY KEY,
    observation_id TEXT NOT NULL REFERENCES observations(id),
    embedding      BLOB,
    chunk_text     TEXT,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id           TEXT PRIMARY KEY,
    call_type    TEXT NOT NULL,
    model        TEXT NOT NULL,
    input_tokens  INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cost_usd      REAL DEFAULT 0.0,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS token_savings (
    id                TEXT PRIMARY KEY,
    thread_id         TEXT NOT NULL REFERENCES threads(id),
    raw_tokens        INTEGER DEFAULT 0,
    compressed_tokens INTEGER DEFAULT 0,
    saved_tokens      INTEGER DEFAULT 0,
    saved_usd         REAL DEFAULT 0.0,
    recorded_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_messages_thread    ON messages(thread_id, observed);
CREATE INDEX IF NOT EXISTS idx_observations_thread ON observations(thread_id);
CREATE INDEX IF NOT EXISTS idx_llm_calls_type     ON llm_calls(call_type);
"""


def _parse_dt(val) -> Optional[datetime]:
    if not val:
        return None
    if isinstance(val, datetime):
        return val
    try:
        return datetime.fromisoformat(str(val))
    except (ValueError, TypeError):
        return None


def _dt(val: Optional[datetime]) -> str:
    """Convert datetime to ISO string for sqlite3 storage."""
    return (val or datetime.now()).isoformat()


class SQLiteStore(BaseStore):

    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            # v0.2 migration: add observation_type column if not present
            cols = [r[1] for r in conn.execute("PRAGMA table_info(observations)").fetchall()]
            if "observation_type" not in cols:
                conn.execute("ALTER TABLE observations ADD COLUMN observation_type TEXT DEFAULT 'observation'")

    # ──────────────────────────── Threads ────────────────────────────

    def save_thread(self, thread: Thread) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO threads (id, agent_type, project_path, created_at, updated_at, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     agent_type   = excluded.agent_type,
                     project_path = excluded.project_path,
                     updated_at   = excluded.updated_at,
                     metadata     = excluded.metadata""",
                (
                    thread.id,
                    thread.agent_type,
                    thread.project_path,
                    _dt(thread.created_at),
                    _dt(thread.updated_at),
                    json.dumps(thread.metadata),
                ),
            )

    def get_thread(self, thread_id: str) -> Optional[Thread]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM threads WHERE id = ?", (thread_id,)
            ).fetchone()
        if not row:
            return None
        return Thread(
            id=row["id"],
            agent_type=row["agent_type"],
            project_path=row["project_path"],
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def list_threads(self) -> list[Thread]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM threads ORDER BY updated_at DESC"
            ).fetchall()
        return [
            Thread(
                id=r["id"],
                agent_type=r["agent_type"],
                project_path=r["project_path"],
                created_at=_parse_dt(r["created_at"]),
                updated_at=_parse_dt(r["updated_at"]),
                metadata=json.loads(r["metadata"] or "{}"),
            )
            for r in rows
        ]

    # ──────────────────────────── Messages ────────────────────────────

    def save_message(self, message: Message) -> None:
        with self._conn() as conn:
            # Ensure thread exists
            conn.execute(
                "INSERT OR IGNORE INTO threads (id) VALUES (?)", (message.thread_id,)
            )
            conn.execute(
                """INSERT OR IGNORE INTO messages
                   (id, thread_id, role, content, token_count, observed, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    message.id,
                    message.thread_id,
                    message.role,
                    message.content,
                    message.token_count,
                    int(message.observed),
                    _dt(message.created_at),
                ),
            )
            conn.execute(
                "UPDATE threads SET updated_at = ? WHERE id = ?",
                (_dt(None), message.thread_id),
            )

    def get_messages(self, thread_id: str, unobserved_only: bool = False, limit: Optional[int] = None) -> list[Message]:
        query = "SELECT * FROM messages WHERE thread_id = ?"
        params: list = [thread_id]
        if unobserved_only:
            query += " AND observed = 0"
        query += " ORDER BY created_at ASC"
        if limit:
            query = f"SELECT * FROM ({query}) ORDER BY created_at DESC LIMIT {limit}"
            query = f"SELECT * FROM ({query}) ORDER BY created_at ASC"
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            Message(
                id=r["id"],
                thread_id=r["thread_id"],
                role=r["role"],
                content=r["content"],
                token_count=r["token_count"],
                observed=bool(r["observed"]),
                created_at=_parse_dt(r["created_at"]),
            )
            for r in rows
        ]

    def get_recent_messages(self, thread_id: str, max_tokens: int) -> list[Message]:
        """Return the most recent messages that fit within max_tokens."""
        all_msgs = self.get_messages(thread_id)
        result: list[Message] = []
        total = 0
        for msg in reversed(all_msgs):
            if total + msg.token_count > max_tokens:
                break
            result.insert(0, msg)
            total += msg.token_count
        return result

    def mark_messages_observed(self, thread_id: str, message_ids: list[str] | None = None) -> None:
        with self._conn() as conn:
            if message_ids:
                placeholders = ",".join("?" * len(message_ids))
                conn.execute(
                    f"UPDATE messages SET observed = 1 WHERE thread_id = ? AND id IN ({placeholders})",
                    [thread_id, *message_ids],
                )
            else:
                conn.execute(
                    "UPDATE messages SET observed = 1 WHERE thread_id = ? AND observed = 0",
                    (thread_id,),
                )

    # ─────────────────────────── Observations ──────────────────────────

    def get_observation(self, thread_id: str) -> Optional[Observation]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM observations WHERE thread_id = ? ORDER BY version DESC LIMIT 1",
                (thread_id,),
            ).fetchone()
        if not row:
            return None
        return Observation(
            id=row["id"],
            thread_id=row["thread_id"],
            content=row["content"],
            token_count=row["token_count"],
            version=row["version"],
            created_at=_parse_dt(row["created_at"]),
            reflected_at=_parse_dt(row["reflected_at"]),
        )

    def append_observations(self, thread_id: str, content: str, token_count: int) -> None:
        existing = self.get_observation(thread_id)
        new_version = (existing.version + 1) if existing else 1
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO observations (id, thread_id, content, token_count, version)
                   VALUES (?, ?, ?, ?, ?)""",
                (str(uuid.uuid4()), thread_id, content, token_count, new_version),
            )

    def replace_observations(self, thread_id: str, content: str, token_count: int) -> None:
        existing = self.get_observation(thread_id)
        new_version = (existing.version + 1) if existing else 1
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM observations WHERE thread_id = ?", (thread_id,)
            )
            conn.execute(
                """INSERT INTO observations (id, thread_id, content, token_count, version, reflected_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (str(uuid.uuid4()), thread_id, content, token_count, new_version, _dt(None)),
            )

    def get_all_observations(self) -> list[Observation]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT o.* FROM observations o
                   INNER JOIN (
                       SELECT thread_id, MAX(version) AS max_ver FROM observations GROUP BY thread_id
                   ) latest ON o.thread_id = latest.thread_id AND o.version = latest.max_ver
                   ORDER BY o.created_at DESC"""
            ).fetchall()
        return [
            Observation(
                id=r["id"],
                thread_id=r["thread_id"],
                content=r["content"],
                token_count=r["token_count"],
                version=r["version"],
                created_at=_parse_dt(r["created_at"]),
                reflected_at=_parse_dt(r["reflected_at"]),
            )
            for r in rows
        ]

    def delete_observation(self, thread_id: str) -> None:
        with self._conn() as conn:
            # Delete embeddings first (FK references observations.id)
            conn.execute(
                """DELETE FROM observation_embeddings WHERE observation_id IN
                   (SELECT id FROM observations WHERE thread_id = ?)""",
                (thread_id,),
            )
            conn.execute("DELETE FROM observations WHERE thread_id = ?", (thread_id,))

    def delete_thread(self, thread_id: str) -> None:
        """Delete everything for a thread: messages, observations, embeddings, savings."""
        with self._conn() as conn:
            conn.execute(
                """DELETE FROM observation_embeddings WHERE observation_id IN
                   (SELECT id FROM observations WHERE thread_id = ?)""",
                (thread_id,),
            )
            conn.execute("DELETE FROM token_savings   WHERE thread_id = ?", (thread_id,))
            conn.execute("DELETE FROM observations    WHERE thread_id = ?", (thread_id,))
            conn.execute("DELETE FROM messages        WHERE thread_id = ?", (thread_id,))
            conn.execute("DELETE FROM threads         WHERE id = ?",        (thread_id,))

    # ──────────────── v0.2: Strategy / Procedural Memory ─────────────

    def add_observation(
        self,
        thread_id: str,
        content: str,
        observation_type: str = "observation",
        token_count: int = 0,
    ) -> None:
        """Insert a single observation row (used by StrategistAgent for strategies)."""
        with self._conn() as conn:
            conn.execute("INSERT OR IGNORE INTO threads (id) VALUES (?)", (thread_id,))
            conn.execute(
                """INSERT INTO observations (id, thread_id, content, token_count, observation_type)
                   VALUES (?, ?, ?, ?, ?)""",
                (str(uuid.uuid4()), thread_id, content, token_count or len(content.split()), observation_type),
            )

    def get_strategies(self, thread_id: str, limit: int = 50) -> list[dict]:
        """Return strategy observations ordered by priority (critical first)."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT id, content, created_at FROM observations
                   WHERE thread_id = ? AND observation_type = 'strategy'
                   ORDER BY
                     CASE WHEN content LIKE '🔴%' THEN 0
                          WHEN content LIKE '🟡%' THEN 1
                          ELSE 2 END,
                     created_at DESC
                   LIMIT ?""",
                (thread_id, limit),
            ).fetchall()
        return [{"id": r["id"], "content": r["content"]} for r in rows]

    def get_all_strategies(self, limit: int = 100) -> list[dict]:
        """Return all strategies across all threads (for global injection)."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT id, thread_id, content, created_at FROM observations
                   WHERE observation_type = 'strategy'
                   ORDER BY
                     CASE WHEN content LIKE '🔴%' THEN 0
                          WHEN content LIKE '🟡%' THEN 1
                          ELSE 2 END,
                     created_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [{"id": r["id"], "thread_id": r["thread_id"], "content": r["content"]} for r in rows]

    # ──────────────────────── Embeddings ──────────────────────────────

    def save_embedding(self, observation_id: str, chunk_text: str, embedding: bytes) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM observation_embeddings WHERE observation_id = ?", (observation_id,))
            conn.execute(
                """INSERT INTO observation_embeddings (id, observation_id, chunk_text, embedding, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (str(uuid.uuid4()), observation_id, chunk_text, embedding, _dt(None)),
            )

    def get_all_embeddings(self) -> list[dict]:
        """Return all stored embeddings with their observation metadata."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT e.observation_id, e.chunk_text, e.embedding, o.thread_id
                   FROM observation_embeddings e
                   JOIN observations o ON e.observation_id = o.id"""
            ).fetchall()
        return [
            {
                "observation_id": r["observation_id"],
                "thread_id": r["thread_id"],
                "chunk_text": r["chunk_text"],
                "embedding": r["embedding"],
            }
            for r in rows
        ]

    def has_embedding(self, observation_id: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM observation_embeddings WHERE observation_id = ?",
                (observation_id,),
            ).fetchone()
        return row is not None

    # ─────────────────────────── LLM Calls ────────────────────────────

    def log_llm_call(
        self,
        call_type: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO llm_calls (id, call_type, model, input_tokens, output_tokens, cost_usd)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (str(uuid.uuid4()), call_type, model, input_tokens, output_tokens, cost_usd),
            )

    def log_token_savings(
        self,
        thread_id: str,
        raw_tokens: int,
        compressed_tokens: int,
        saved_usd: float,
    ) -> None:
        saved = max(0, raw_tokens - compressed_tokens)
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO token_savings
                   (id, thread_id, raw_tokens, compressed_tokens, saved_tokens, saved_usd)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (str(uuid.uuid4()), thread_id, raw_tokens, compressed_tokens, saved, saved_usd),
            )

    # ──────────────────────────── Stats ─────────────────────────────

    def get_stats(self) -> MemoryStats:
        with self._conn() as conn:
            threads = conn.execute("SELECT COUNT(*) FROM threads").fetchone()[0]
            messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            observations = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
            savings = conn.execute(
                "SELECT COALESCE(SUM(saved_tokens),0), COALESCE(SUM(saved_usd),0) FROM token_savings"
            ).fetchone()
            llm_cost = conn.execute(
                "SELECT COALESCE(SUM(cost_usd),0) FROM llm_calls"
            ).fetchone()[0]
            oldest = conn.execute(
                "SELECT MIN(created_at) FROM messages"
            ).fetchone()[0]
            newest = conn.execute(
                "SELECT MAX(created_at) FROM messages"
            ).fetchone()[0]

        return MemoryStats(
            total_threads=threads,
            total_messages=messages,
            total_observations=observations,
            total_tokens_saved=int(savings[0]),
            total_cost_saved_usd=float(savings[1]),
            total_llm_cost_usd=float(llm_cost),
            oldest_memory=_parse_dt(oldest),
            newest_memory=_parse_dt(newest),
        )
