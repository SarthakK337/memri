"""Layer 2 — cold archive. Stores raw conversation segments. Zero data loss."""

import sqlite3
from datetime import datetime
from typing import Dict, List, Optional


CREATE_EPISODES = """
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    session_index INTEGER,
    session_date TEXT,
    raw_text TEXT NOT NULL,
    created_at TEXT NOT NULL
)"""

# Additive migration — safe to run on existing DBs
MIGRATE_SUMMARY = """
ALTER TABLE episodes ADD COLUMN summary TEXT
"""
MIGRATE_SUMMARY_EMBED = """
ALTER TABLE episodes ADD COLUMN summary_embedding_id TEXT
"""


class Layer2Store:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute(CREATE_EPISODES)
        conn.commit()
        # Additive migrations — ignore if columns already exist
        for stmt in (MIGRATE_SUMMARY, MIGRATE_SUMMARY_EMBED):
            try:
                conn.execute(stmt)
                conn.commit()
            except sqlite3.OperationalError:
                pass  # column already exists
        conn.close()

    def store_episode(
        self,
        episode_id: str,
        raw_text: str,
        session_index: int = 0,
        session_date: str = None,
    ):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO episodes (id, session_index, session_date, raw_text, created_at) VALUES (?, ?, ?, ?, ?)",
            (episode_id, session_index, session_date, raw_text, datetime.utcnow().isoformat()),
        )
        conn.commit()
        conn.close()

    def store_summary(self, episode_id: str, summary: str, summary_embedding_id: str = None):
        """Store or update the session summary and its embedding ID."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE episodes SET summary = ?, summary_embedding_id = ? WHERE id = ?",
            (summary, summary_embedding_id, episode_id),
        )
        conn.commit()
        conn.close()

    def get_episode(self, episode_id: str) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT id, session_index, session_date, raw_text, summary FROM episodes WHERE id = ?",
            (episode_id,),
        )
        row = cur.fetchone()
        conn.close()
        if row:
            return {
                "id": row[0], "session_index": row[1], "session_date": row[2],
                "raw_text": row[3], "summary": row[4],
            }
        return None

    def get_episodes_for_facts(self, fact_nodes: list) -> Dict[str, str]:
        """Given fact MemoryNode objects, return {fact_node_id: raw_episode_text}."""
        episode_to_facts: Dict[str, List[str]] = {}
        for node in fact_nodes:
            ep_id = node.source_episode_id
            if ep_id:
                episode_to_facts.setdefault(ep_id, []).append(node.id)

        if not episode_to_facts:
            return {}

        placeholders = ",".join("?" * len(episode_to_facts))
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            f"SELECT id, raw_text FROM episodes WHERE id IN ({placeholders})",
            list(episode_to_facts.keys()),
        )
        ep_text_map = {row[0]: row[1] for row in cur.fetchall()}
        conn.close()

        result = {}
        for ep_id, fact_ids in episode_to_facts.items():
            raw = ep_text_map.get(ep_id)
            if raw:
                for fid in fact_ids:
                    result[fid] = raw
        return result

    def get_all_episodes(self) -> List[Dict]:
        """Return all episodes sorted by session_index."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT id, session_index, session_date, raw_text FROM episodes ORDER BY session_index")
        rows = cur.fetchall()
        conn.close()
        return [{"id": r[0], "session_index": r[1], "session_date": r[2], "raw_text": r[3]} for r in rows]

    def count(self) -> int:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM episodes")
        n = cur.fetchone()[0]
        conn.close()
        return n
