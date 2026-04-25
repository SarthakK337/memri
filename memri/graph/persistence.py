"""SQLite persistence for GraphStore + Layer 0."""

import sqlite3
import json
from typing import Tuple
from datetime import datetime

from .types import MemoryNode, Layer0, EdgeType
from .store import GraphStore

CREATE_NODES = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    data_json TEXT NOT NULL
)"""

CREATE_EDGES = """
CREATE TABLE IF NOT EXISTS edges (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (source_id, target_id, edge_type)
)"""

CREATE_LAYER0 = """
CREATE TABLE IF NOT EXISTS layer0 (
    id INTEGER PRIMARY KEY,
    data_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
)"""


class Persistence:
    def save(self, graph_store: GraphStore, layer0: Layer0, path: str) -> None:
        """Persist graph + Layer 0 to SQLite (full overwrite)."""
        conn = sqlite3.connect(path)
        cur = conn.cursor()

        for stmt in (CREATE_NODES, CREATE_EDGES, CREATE_LAYER0):
            cur.execute(stmt)

        cur.execute("DELETE FROM nodes")
        cur.execute("DELETE FROM edges")
        cur.execute("DELETE FROM layer0")

        # Save nodes
        for node in graph_store.nodes.values():
            cur.execute(
                "INSERT INTO nodes (id, data_json) VALUES (?, ?)",
                (node.id, node.model_dump_json()),
            )

        # Save edges
        for source, target, data in graph_store.graph.edges(data=True):
            cur.execute(
                "INSERT OR REPLACE INTO edges (source_id, target_id, edge_type, weight) VALUES (?, ?, ?, ?)",
                (source, target, data.get("edge_type", ""), data.get("weight", 1.0)),
            )

        # Save Layer 0
        cur.execute(
            "INSERT INTO layer0 (id, data_json, updated_at) VALUES (1, ?, ?)",
            (layer0.model_dump_json(), datetime.utcnow().isoformat()),
        )

        conn.commit()
        conn.close()

    def load(self, path: str) -> Tuple[GraphStore, Layer0]:
        """Load graph + Layer 0 from SQLite."""
        conn = sqlite3.connect(path)
        cur = conn.cursor()

        # Ensure tables exist (in case db was created externally)
        for stmt in (CREATE_NODES, CREATE_EDGES, CREATE_LAYER0):
            cur.execute(stmt)

        graph = GraphStore()

        # Load nodes
        cur.execute("SELECT data_json FROM nodes")
        for (data_json,) in cur.fetchall():
            node = MemoryNode.model_validate_json(data_json)
            graph.nodes[node.id] = node
            graph.graph.add_node(node.id, node_type=node.node_type.value)

        # Load edges
        cur.execute("SELECT source_id, target_id, edge_type, weight FROM edges")
        for source, target, edge_type_str, weight in cur.fetchall():
            graph.graph.add_edge(
                source, target, edge_type=edge_type_str, weight=weight
            )

        # Load Layer 0
        cur.execute("SELECT data_json FROM layer0 WHERE id = 1")
        row = cur.fetchone()
        try:
            layer0 = Layer0.model_validate_json(row[0]) if row else Layer0()
        except Exception as e:
            print(f"[persistence] Layer 0 load failed (using empty): {e}")
            layer0 = Layer0()

        conn.close()
        return graph, layer0
