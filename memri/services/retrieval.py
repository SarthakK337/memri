"""Retrieval pipeline: query → Layer 0 routing → graph traversal → RRF ranking → top-k."""

from typing import List, Optional, Set

from ..graph.types import EdgeType, Layer0, NodeType, SearchResult, ScoringWeights
from ..graph.store import GraphStore
from ..graph.embeddings import EmbeddingStore
from ..llm.graph_adapter import GraphLLMAdapter as LLMProvider
from ..storage.layer2 import Layer2Store
from .ranker import RRFRanker
from .query_processor import extract_entities, detect_temporal_intent, HAS_SPACY


class RetrievalService:
    def __init__(
        self,
        graph: GraphStore,
        embeddings: EmbeddingStore,
        llm: LLMProvider,
        layer0: Layer0,
        weights: ScoringWeights = None,
        layer2: Layer2Store = None,
        use_cluster_routing: bool = True,
    ):
        self.graph = graph
        self.embeddings = embeddings
        self.llm = llm
        self.layer0 = layer0
        self.weights = weights
        self.layer2 = layer2
        self.use_cluster_routing = use_cluster_routing  # A/B flag: toggle cluster-first routing
        self.ranker = RRFRanker(graph, embeddings)
        self.ranker.build_bm25_index()

    async def search(self, query: str, top_k: int = 5, max_sessions: int = 3) -> dict:
        """Main search: Layer 0 routing → candidates → RRF ranking → top-k.

        Returns dict:
          results: List[SearchResult]  — top-k ranked fact nodes
          source_sessions: list        — raw session texts from Layer 2
        """

        # Step 1: Get anchor nodes via topic + entity routing
        anchor_node_ids = await self._get_anchors(query)

        # Fallback: pure vector search if no anchors
        if not anchor_node_ids:
            results = await self._vector_search_fallback(query, top_k)
            source_sessions = self._get_source_sessions(results, max_sessions) if self.layer2 else []
            return {"results": results, "source_sessions": source_sessions}

        # Step 2: Collect candidates with intersection-based hub fix
        candidate_ids = self._get_candidates_from_anchors(anchor_node_ids)
        candidate_ids.update(anchor_node_ids)

        # Always augment with top BM25 hits from full index
        bm25_boost = self.ranker.get_top_bm25_ids(query, top_n=30)
        candidate_ids.update(bm25_boost)

        # Always augment with top vector hits from full index
        # (catches semantically relevant facts not in BM25 or anchor neighborhood)
        vec_boost = self.embeddings.search(query, n_results=30)
        candidate_ids.update(nid for nid, _ in vec_boost)

        # Temporal chain augmentation for when/date/duration queries
        temporal_info = detect_temporal_intent(query)
        if temporal_info["has_temporal"]:
            for f in self.get_temporal_chain_for_query(query)[:20]:
                candidate_ids.add(f.id)

        # Entity neighborhood augmentation for multi-hop entity queries
        for f in self.get_entity_neighborhood_for_query(query)[:30]:
            candidate_ids.add(f.id)

        # Filter superseded facts — prefer current state in results
        candidate_ids = {
            cid for cid in candidate_ids
            if not getattr(self.graph.get_node(cid), "is_superseded", False)
        }

        # Step 3: RRF ranking
        results = self.ranker.rank(query, candidate_ids, anchor_node_ids, top_k)

        # Step 4: Update access metadata
        from datetime import datetime
        for r in results:
            r.node.last_accessed = datetime.utcnow()
            r.node.access_count += 1

        # Step 5: Fetch source sessions from Layer 2 (max-score selection)
        source_sessions = self._get_source_sessions(results, max_sessions, query) if self.layer2 else []

        return {"results": results, "source_sessions": source_sessions}

    def _get_source_sessions(self, results: List[SearchResult], max_sessions: int = 3, query: str = "") -> list:
        """Select the most relevant sessions from Layer 2.

        Ranks episodes by the BEST (highest) RRF score fact pointing to each,
        not by raw hit count. Also adds sessions containing the top BM25-matched
        facts so keyword-specific answers (guinea pig, Sara Bareilles, etc.) are
        never missed even when importance/recency hubs dominate the RRF ranking.
        """
        episode_best: dict = {}  # episode_id → best RRF score
        for r in results:
            eid = r.node.source_episode_id
            if eid:
                if eid not in episode_best or r.score > episode_best[eid]:
                    episode_best[eid] = r.score

        # BM25 session boost: inject session of top BM25 hit if not already selected
        if query and self.ranker:
            top_bm25_ids = self.ranker.get_top_bm25_ids(query, top_n=10)
            for nid in top_bm25_ids:
                node = self.graph.get_node(nid)
                if node and node.source_episode_id and node.source_episode_id not in episode_best:
                    episode_best[node.source_episode_id] = -1.0  # low priority sentinel
                    break  # add at most one BM25-boosted session

        if not episode_best:
            return []

        # Blend summary embedding similarity into episode ranking
        if query and episode_best:
            for episode_id in list(episode_best.keys()):
                summary_embed_id = f"summary_{episode_id}"
                sim = self.embeddings.similarity(query, summary_embed_id)
                if sim > 0:
                    # Weighted blend: 50% RRF score + 50% summary similarity
                    rrf_score = episode_best[episode_id]
                    if rrf_score > 0:
                        episode_best[episode_id] = 0.5 * rrf_score + 0.5 * sim
                    # For BM25-sentinel episodes (score -1), use summary sim as score
                    else:
                        episode_best[episode_id] = 0.5 * sim

        top_episodes = sorted(episode_best.items(), key=lambda x: x[1], reverse=True)[:max_sessions]

        sessions = []
        for episode_id, best_score in top_episodes:
            episode = self.layer2.get_episode(episode_id)
            if episode:
                sessions.append({
                    "episode_id": episode_id,
                    "session_index": episode.get("session_index", 0),
                    "session_date": episode.get("session_date", ""),
                    "raw_text": episode.get("raw_text", ""),
                    "summary": episode.get("summary", ""),
                    "best_score": best_score,
                })

        sessions.sort(key=lambda x: x["session_index"])
        return sessions

    async def _get_anchors(self, query: str) -> List[str]:
        """Route query through topic clusters first (if enabled), then entity index."""

        # 1. Topic-level routing with synonym expansion (A/B controlled by use_cluster_routing)
        if self.use_cluster_routing and self.layer0.topic_clusters:
            # Expand query with synonyms so "song" matches "music", "book" matches "reading" etc.
            SYNONYMS = {
                "song": ["music", "song", "tune", "track"],
                "music": ["music", "song", "tune"],
                "book": ["book", "reading", "novel", "childhood"],
                "read": ["book", "reading", "novel"],
                "paint": ["painting", "art", "artistic"],
                "painting": ["painting", "art", "artistic"],
                "art": ["art", "painting", "artistic"],
                "adopt": ["adoption", "family"],
                "adoption": ["adoption", "family"],
                "school": ["education", "career", "counseling"],
                "career": ["career", "counseling", "education"],
                "lgbt": ["lgbtq", "transgender", "identity"],
                "trans": ["transgender", "identity", "lgbtq"],
                "yoga": ["yoga", "fitness", "health"],
                "camp": ["camping", "nature", "outdoors"],
                "travel": ["travel", "trip", "road"],
                "trip": ["travel", "trip", "road"],
            }
            query_tokens = set(query.lower().split())
            expanded_tokens = set(query_tokens)
            for token in list(query_tokens):
                token_clean = token.strip("?.,!'\"")
                for base, expansions in SYNONYMS.items():
                    if token_clean.startswith(base):
                        expanded_tokens.update(expansions)

            topic_matches = []
            for topic, node_ids in self.layer0.topic_clusters.items():
                topic_tokens = set(topic.lower().replace("_", " ").split())
                overlap = len(expanded_tokens & topic_tokens)
                if overlap > 0:
                    topic_matches.append((overlap, node_ids))

            if topic_matches:
                topic_matches.sort(key=lambda x: x[0], reverse=True)
                anchor_ids = []
                for _, nids in topic_matches[:3]:
                    anchor_ids.extend(nids)
                anchor_ids = [nid for nid in anchor_ids if self.graph.get_node(nid)]
                if anchor_ids:
                    return list(set(anchor_ids))

        # 2. Entity routing — use spaCy if available, else LLM
        if HAS_SPACY:
            query_entities = extract_entities(query)
        else:
            query_entities = await self.llm.extract_entities(query)

        return self.layer0.get_anchor_nodes(query_entities)

    def _get_candidates_from_anchors(self, anchor_ids: List[str], max_hops: int = 2) -> Set[str]:
        """Intersection-based candidate selection to reduce hub noise."""
        if len(anchor_ids) == 1:
            return self.graph.get_neighborhood(anchor_ids[0], max_hops)

        neighborhoods = [self.graph.get_neighborhood(aid, max_hops) for aid in anchor_ids]
        intersection = set.intersection(*neighborhoods) if neighborhoods else set()

        if len(intersection) >= 5:
            return intersection

        return set.union(*neighborhoods) if neighborhoods else set()

    def get_entity_scoped_facts(
        self,
        query: str,
        top_k: int = None,
        sort_by_session: bool = False,
    ) -> List["MemoryNode"]:
        """
        Entity-scoped fact retrieval using the entity_index.

        Finds all entity names from the query that match known entities,
        pulls ALL their facts from the graph, then either:
          - ranks by BM25+vector relevance and returns top_k, or
          - returns all facts sorted chronologically (sort_by_session=True)

        This bypasses topic routing entirely and uses the graph directly,
        which is more reliable for multi-hop, temporal, and inference questions.
        """
        query_lower = query.lower()
        entity_fact_ids: set = set()

        for entity_name, fact_ids in self.layer0.entity_index.items():
            # Only match entity names of 3+ characters to avoid noise
            if len(entity_name) >= 3 and entity_name.lower() in query_lower:
                entity_fact_ids.update(fact_ids)

        if not entity_fact_ids:
            return []

        facts = [self.graph.get_node(fid) for fid in entity_fact_ids]
        facts = [
            f for f in facts
            if f and f.node_type == NodeType.FACT
            and not getattr(f, "is_superseded", False)
        ]
        filtered_ids = {f.id for f in facts}

        if sort_by_session:
            facts.sort(key=lambda f: (f.session_index or 0, f.created_at))
            return facts  # return all, caller decides how many to use

        if top_k:
            results = self.ranker.rank(query, filtered_ids, list(filtered_ids), top_k)
            return [r.node for r in results]

        facts.sort(key=lambda f: (f.session_index or 0))
        return facts

    async def _vector_search_fallback(self, query: str, top_k: int) -> List[SearchResult]:
        """Pure vector search when Layer 0 routing finds no anchors."""
        all_ids = {
            nid for nid, node in self.graph.nodes.items()
            if node.node_type != NodeType.EPISODE
        }
        results = self.ranker.rank(query, all_ids, [], top_k)

        from datetime import datetime
        for r in results:
            r.node.last_accessed = datetime.utcnow()
            r.node.access_count += 1

        return results

    def get_sessions_by_summary(self, query: str, top_n: int = 4) -> list:
        """Find sessions whose summary embeddings are most similar to query.

        This is the Layer 2 fallback for when fact retrieval misses an event:
        search session summaries directly, return raw session texts.
        Used by Cat2 to find events not captured as facts.
        """
        if not self.layer2:
            return []

        # Search ALL embeddings for summary_ prefix matches
        # We search broadly then filter to summary IDs
        try:
            all_results = self.embeddings.search(query, n_results=min(200, self.embeddings.collection.count()))
        except Exception:
            return []

        summary_hits = [
            (nid, score) for nid, score in all_results
            if nid.startswith("summary_")
        ]
        summary_hits.sort(key=lambda x: x[1], reverse=True)

        sessions = []
        seen_episodes = set()
        for summary_id, score in summary_hits[:top_n]:
            episode_id = summary_id[len("summary_"):]
            if episode_id in seen_episodes:
                continue
            seen_episodes.add(episode_id)
            episode = self.layer2.get_episode(episode_id)
            if episode:
                sessions.append({
                    "episode_id": episode_id,
                    "session_index": episode.get("session_index", 0),
                    "session_date": episode.get("session_date", ""),
                    "raw_text": episode.get("raw_text", ""),
                    "summary": episode.get("summary", ""),
                    "summary_score": score,
                })

        sessions.sort(key=lambda x: x["session_index"])
        return sessions

    # ── v1 graph-traversal helpers ────────────────────────────────────────────

    def get_temporal_chain_for_query(self, query: str, window_days: Optional[int] = None) -> List["MemoryNode"]:
        """Find seed facts via entity index, then walk temporal chain."""
        entity_facts = self.get_entity_scoped_facts(query, top_k=5)
        if not entity_facts:
            return []
        # Walk temporal chain from each seed, merge results
        seen: Set[str] = set()
        chain: List["MemoryNode"] = []
        for seed in entity_facts[:3]:
            for fact in self.graph.get_temporal_chain(seed.id, window_days):
                if fact.id not in seen:
                    seen.add(fact.id)
                    chain.append(fact)
        chain.sort(key=lambda f: (f.temporal_date or "", f.session_index or 0))
        return chain

    def get_causal_chain_for_query(self, query: str, max_depth: int = 3) -> List["MemoryNode"]:
        """Find seed facts via entity index, then walk causal chain."""
        entity_facts = self.get_entity_scoped_facts(query, top_k=5)
        if not entity_facts:
            return []
        seen: Set[str] = set()
        chain: List["MemoryNode"] = []
        for seed in entity_facts[:3]:
            for fact in self.graph.get_causal_chain(seed.id, max_depth):
                if fact.id not in seen:
                    seen.add(fact.id)
                    chain.append(fact)
        return chain

    def get_entity_neighborhood_for_query(self, query: str, max_hops: int = 2) -> List["MemoryNode"]:
        """Find entity nodes from query, BFS via BELONGS_TO/RELATED_TO, return fact nodes."""
        query_lower = query.lower()
        entity_nodes = self.graph.get_nodes_by_type(NodeType.ENTITY)
        anchor_entity_ids = [
            e.id for e in entity_nodes
            if len(e.content) >= 3 and e.content.lower() in query_lower
        ]
        if not anchor_entity_ids:
            return []
        seen: Set[str] = set()
        facts = []
        for eid in anchor_entity_ids[:5]:
            fact_ids = self.graph.get_entity_neighborhood(eid, max_hops)
            for fid in fact_ids:
                if fid not in seen:
                    seen.add(fid)
                    node = self.graph.get_node(fid)
                    if node:
                        facts.append(node)
        facts.sort(key=lambda f: (f.session_index or 0, f.created_at))
        return facts

    def expand_with_graph(self, seed_facts: List["MemoryNode"], edge_types: List[EdgeType], max_hops: int = 1) -> List["MemoryNode"]:
        """Expand seed facts via typed edges, returning merged + deduplicated list."""
        seed_ids = [f.id for f in seed_facts]
        expanded = self.graph.get_connected_facts(seed_ids, edge_types, max_hops)
        seen = {f.id for f in seed_facts}
        merged = list(seed_facts)
        for f in expanded:
            if f.id not in seen:
                seen.add(f.id)
                merged.append(f)
        merged.sort(key=lambda f: (f.session_index or 0, f.created_at))
        return merged
