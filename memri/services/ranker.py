"""RRF + BM25 ranker — replaces weighted sum scoring in retrieval."""

import math
from datetime import datetime
from typing import List, Dict, Set, Optional

from ..graph.types import MemoryNode, NodeType, SearchResult
from ..graph.store import GraphStore
from ..graph.embeddings import EmbeddingStore

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

RRF_K = 60  # standard constant


class RRFRanker:
    """Reciprocal Rank Fusion across 4 signals: vector, BM25, importance, recency."""

    def __init__(self, graph: GraphStore, embeddings: EmbeddingStore):
        self.graph = graph
        self.embeddings = embeddings
        self.bm25_index = None
        self.bm25_node_ids: List[str] = []

    def build_bm25_index(self):
        """Build BM25 index from all fact/reflection nodes. Call after add() or load."""
        if not HAS_BM25:
            return
        nodes = [
            n for n in self.graph.nodes.values()
            if n.node_type in (NodeType.FACT, NodeType.REFLECTION)
        ]
        self.bm25_node_ids = [n.id for n in nodes]
        tokenized = [n.content.lower().split() for n in nodes]
        if tokenized:
            self.bm25_index = BM25Okapi(tokenized)

    def rank(
        self,
        query: str,
        candidate_ids: Set[str],
        anchor_ids: List[str],
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Rank candidates using RRF across 4 signals."""
        candidates = [
            self.graph.get_node(nid) for nid in candidate_ids
            if self.graph.get_node(nid) and
            self.graph.get_node(nid).node_type != NodeType.EPISODE
        ]
        candidates = [c for c in candidates if c is not None]

        if not candidates:
            return []

        candidate_set = {c.id for c in candidates}

        # Signal 1: Vector similarity
        n_request = min(len(candidates) + 20, max(self.embeddings.collection.count(), 1))
        vector_results = self.embeddings.search(query, n_results=n_request)
        relevance_scores = {nid: sim for nid, sim in vector_results if nid in candidate_set}

        # Signal 2: BM25 keyword relevance
        bm25_scores: Dict[str, float] = {}
        if HAS_BM25 and self.bm25_index:
            query_tokens = query.lower().split()
            raw_scores = self.bm25_index.get_scores(query_tokens)
            for idx, score in enumerate(raw_scores):
                nid = self.bm25_node_ids[idx]
                if nid in candidate_set:
                    bm25_scores[nid] = float(score)

        # Signal 3: Importance
        importance_scores = {c.id: c.importance for c in candidates}

        # Signal 4: Recency
        now = datetime.utcnow()
        recency_scores = {
            c.id: math.exp(-0.01 * (now - c.last_accessed).total_seconds() / 3600)
            for c in candidates
        }

        # Signal 5: Emotion boost — emotional facts surface higher
        emotion_scores = {c.id: c.emotional_weight for c in candidates}

        def rank_dict(scores: Dict[str, float]) -> Dict[str, int]:
            sorted_ids = sorted(scores.keys(), key=lambda x: scores.get(x, 0), reverse=True)
            return {nid: rank + 1 for rank, nid in enumerate(sorted_ids)}

        relevance_ranks = rank_dict(relevance_scores)
        bm25_ranks = rank_dict(bm25_scores)
        importance_ranks = rank_dict(importance_scores)
        recency_ranks = rank_dict(recency_scores)
        emotion_ranks = rank_dict(emotion_scores)

        # RRF fusion — query signals (vector + BM25) get 2× weight vs metadata signals
        # This prevents high-importance hub facts from drowning out specific matches
        rrf_scores: Dict[str, float] = {}
        for c in candidates:
            nid = c.id
            rrf_scores[nid] = (
                2.0 / (RRF_K + relevance_ranks.get(nid, 999))
                + 2.0 / (RRF_K + bm25_ranks.get(nid, 999))
                + 0.5 / (RRF_K + importance_ranks.get(nid, 999))
                + 0.5 / (RRF_K + recency_ranks.get(nid, 999))
                + 0.3 / (RRF_K + emotion_ranks.get(nid, 999))
            )

        sorted_candidates = sorted(candidates, key=lambda c: rrf_scores.get(c.id, 0), reverse=True)

        # Confidence check: if top result ranks poorly on both vector and BM25
        low_confidence = False
        if sorted_candidates:
            top_id = sorted_candidates[0].id
            if relevance_ranks.get(top_id, 999) > 50 and bm25_ranks.get(top_id, 999) > 50:
                low_confidence = True

        results = []
        for c in sorted_candidates[:top_k]:
            results.append(SearchResult(
                node=c,
                score=rrf_scores[c.id],
                scoring_breakdown={
                    "relevance_rank": float(relevance_ranks.get(c.id, 999)),
                    "bm25_rank": float(bm25_ranks.get(c.id, 999)),
                    "importance_rank": float(importance_ranks.get(c.id, 999)),
                    "recency_rank": float(recency_ranks.get(c.id, 999)),
                    "emotion_rank": float(emotion_ranks.get(c.id, 999)),
                },
                low_confidence=low_confidence,
            ))

        return results

    def get_top_bm25_ids(self, query: str, top_n: int = 30) -> List[str]:
        """Return top-N fact IDs by BM25 score across the full index (no candidate filter)."""
        if not HAS_BM25 or not self.bm25_index:
            return []
        query_tokens = query.lower().split()
        raw_scores = self.bm25_index.get_scores(query_tokens)
        scored = sorted(
            ((self.bm25_node_ids[i], raw_scores[i]) for i in range(len(raw_scores))),
            key=lambda x: x[1], reverse=True,
        )
        return [nid for nid, score in scored[:top_n] if score > 0]
