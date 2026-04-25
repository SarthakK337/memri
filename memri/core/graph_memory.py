"""Graph-based memory engine — the primary memory backend for memri v1.0.

Wraps the graph layer (types, store, persistence, embeddings) and services
(ingestion, retrieval, layer0, reflection) into a single engine class.

This is a direct adaptation of the Smriti.__init__ class, using Memri's
LLM provider abstraction instead of Smriti's own.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from ..graph.types import Layer0, MemoryNode, NodeType, SearchResult, ScoringWeights
from ..graph.store import GraphStore
from ..graph.persistence import Persistence
from ..graph.embeddings import EmbeddingStore
from ..storage.layer2 import Layer2Store
from ..services.ingestion import IngestionService
from ..services.retrieval import RetrievalService
from ..services.reflection import ReflectionService
from ..services.layer0 import Layer0Service
from ..services.query_classifier import classify_query
from ..llm.graph_adapter import GraphLLMAdapter

REFLECT_EVERY_N_SESSIONS = 10


class GraphMemoryEngine:
    """
    Graph-based layered memory engine.

    Three layers:
      Layer 0 — always-in-context entity/topic routing index (~500 tokens)
      Layer 1 — NetworkX graph of fact/entity/reflection nodes
      Layer 2 — raw episode archive (SQLite, zero data loss)
    """

    def __init__(self, storage_path: Path, llm_adapter: GraphLLMAdapter):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.llm = llm_adapter
        self.embeddings = EmbeddingStore(str(self.storage_path / "embeddings"))
        self.graph = GraphStore()
        self.layer0 = Layer0()
        self.weights = ScoringWeights()
        self.persistence = Persistence()
        self.layer2 = Layer2Store(str(self.storage_path / "layer2.db"))

        db_path = str(self.storage_path / "memory.db")
        if Path(db_path).exists():
            self.graph, self.layer0 = self.persistence.load(db_path)

        self._init_services()
        self.retrieval.ranker.build_bm25_index()

    def _init_services(self) -> None:
        self.ingestion = IngestionService(self.graph, self.embeddings, self.llm, self.layer2)
        self.retrieval = RetrievalService(
            self.graph, self.embeddings, self.llm, self.layer0, self.weights, self.layer2
        )
        self.reflection_svc = ReflectionService(self.graph, self.llm)

    async def add(self, conversation: str, session_index: int = 0, session_date: str = None) -> dict:
        """Ingest a conversation. Returns {episode_id, facts_extracted, entities_linked}."""
        result = await self.ingestion.ingest_conversation(conversation, session_index, session_date)
        self.layer0 = await Layer0Service(self.graph, self.llm).generate()
        self.retrieval.layer0 = self.layer0
        self.retrieval.ranker.build_bm25_index()
        self._save()

        episode_count = len(self.graph.get_nodes_by_type(NodeType.EPISODE))
        if episode_count > 0 and episode_count % REFLECT_EVERY_N_SESSIONS == 0:
            try:
                await self.reflect()
                self.retrieval.ranker.build_bm25_index()
            except Exception as e:
                print(f"[memory] Auto-reflection failed: {e}")

        return result

    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Graph search: Layer 0 routing → BFS traversal → RRF ranking."""
        intent = classify_query(query)
        effective_top_k = 30 if intent.is_aggregate else top_k
        max_sessions = 5 if intent.is_aggregate else 3
        output = await self.retrieval.search(query, top_k=effective_top_k, max_sessions=max_sessions)
        return output["results"]

    def context(self) -> str:
        """Layer 0 as a string for injection into agent context."""
        return self.layer0.to_context_string()

    async def reflect(self) -> List[MemoryNode]:
        """Generate higher-level pattern reflections from entity clusters."""
        reflections = await self.reflection_svc.generate_reflections(self.layer0)
        self.layer0 = await Layer0Service(self.graph, self.llm).generate()
        self.retrieval.layer0 = self.layer0
        self._save()
        return reflections

    def _save(self) -> None:
        self.persistence.save(self.graph, self.layer0, str(self.storage_path / "memory.db"))

    def stats(self) -> dict:
        return {
            "facts": len(self.graph.get_nodes_by_type(NodeType.FACT)),
            "entities": len(self.graph.get_nodes_by_type(NodeType.ENTITY)),
            "reflections": len(self.graph.get_nodes_by_type(NodeType.REFLECTION)),
            "episodes": len(self.graph.get_nodes_by_type(NodeType.EPISODE)),
            "edges": self.graph.graph.number_of_edges(),
            "layer0_entities": len(self.layer0.entity_index),
        }

    def get_layer0(self) -> Layer0:
        return self.layer0

    def get_graph_data(self) -> dict:
        """Serializable graph data for dashboard visualization."""
        nodes = []
        for node in self.graph.nodes.values():
            nodes.append({
                "id": node.id,
                "type": node.node_type.value,
                "content": node.content[:200],
                "importance": node.importance,
                "emotional_weight": node.emotional_weight,
                "session_index": node.session_index,
                "session_date": node.session_date,
                "is_superseded": node.is_superseded,
                "created_at": node.created_at.isoformat() if node.created_at else None,
            })
        edges = []
        for src, tgt, data in self.graph.graph.edges(data=True):
            edges.append({
                "source": src,
                "target": tgt,
                "type": data.get("edge_type", ""),
                "weight": data.get("weight", 1.0),
            })
        return {"nodes": nodes, "edges": edges}

    def get_episodes(self) -> list:
        """All Layer 2 episodes for dashboard."""
        return self.layer2.get_all_episodes()
