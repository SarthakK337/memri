"""Core memory types — graph nodes, edges, Layer 0 index."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Union
from enum import Enum
from datetime import datetime
import uuid


# ── Node Types ──

class NodeType(str, Enum):
    FACT = "fact"             # atomic factual unit
    ENTITY = "entity"         # person/place/thing
    REFLECTION = "reflection" # higher-level pattern
    EPISODE = "episode"       # raw conversation segment


# ── Edge Types ──

class EdgeType(str, Enum):
    # Semantic edges
    RELATED_TO = "related_to"
    BELONGS_TO = "belongs_to"
    DERIVED_FROM = "derived_from"

    # Temporal edges
    HAPPENED_BEFORE = "happened_before"
    HAPPENED_AFTER = "happened_after"
    HAPPENED_DURING = "happened_during"

    # Causal edges
    CAUSED = "caused"
    COPING_WITH = "coping_with"

    # Evolution edges
    SUPERSEDES = "supersedes"
    CONTRADICTS = "contradicts"


# ── Memory Node ──

class MemoryNode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    node_type: NodeType
    content: str
    content_summary: Optional[str] = None
    content_abstract: Optional[Dict] = None

    importance: float = 0.5
    emotional_weight: float = 0.0
    confidence: float = 1.0

    # Temporal (v0.2)
    temporal_date: Optional[str] = None       # resolved date YYYY-MM-DD
    temporal_reference: Optional[str] = None   # original text ("May 8, 2023")
    session_index: Optional[int] = None        # which session number
    session_date: Optional[str] = None         # session timestamp string

    is_superseded: bool = False  # True when a newer fact replaces this one (same entity+predicate)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = 0
    source_episode_id: Optional[str] = None
    raw_content_hash: Optional[str] = None
    embedding_id: Optional[str] = None


# ── Edge ──

class Edge(BaseModel):
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict] = None


# ── Layer 0 ──

class Layer0(BaseModel):
    """Always-in-context routing index. Injected into LLM system prompt."""

    user_summary: str = ""
    active_topics: List[str] = Field(default_factory=list)
    emotional_state: str = ""

    entity_index: Dict[str, List[str]] = Field(default_factory=dict)
    topic_clusters: Dict[str, List[str]] = Field(default_factory=dict)
    aliases: Dict[str, str] = Field(default_factory=dict)  # alias → canonical entity name

    fact_count: int = 0
    reflection_count: int = 0
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    def to_context_string(self) -> str:
        """Format Layer 0 as a string to inject into LLM context."""
        parts = []
        if self.user_summary:
            parts.append(f"User Summary: {self.user_summary}")
        if self.active_topics:
            parts.append(f"Active Topics: {', '.join(self.active_topics)}")
        if self.emotional_state:
            parts.append(f"Emotional State: {self.emotional_state}")
        if self.entity_index:
            entities = ', '.join(self.entity_index.keys())
            parts.append(f"Known Entities: {entities}")
        parts.append(f"Memory Stats: {self.fact_count} facts, {self.reflection_count} reflections")
        return "\n".join(parts)

    def resolve_alias(self, name: str) -> str:
        """Resolve alias to canonical entity name. Returns canonical or original."""
        return self.aliases.get(name.lower().strip(), name)

    def get_anchor_nodes(self, query_entities: List[str]) -> List[str]:
        """Route a query to relevant node IDs via entity matching (alias-aware)."""
        anchors = []
        for entity in query_entities:
            # Resolve alias before lookup
            canonical = self.resolve_alias(entity)
            for lookup in {entity, canonical}:
                lookup_lower = lookup.lower().strip()
                for key, node_ids in self.entity_index.items():
                    key_lower = key.lower().strip()
                    if (lookup_lower == key_lower
                            or lookup_lower.startswith(key_lower + " ")
                            or key_lower.startswith(lookup_lower + " ")):
                        anchors.extend(node_ids)
        return list(set(anchors))


# ── Scoring Weights ──

class ScoringWeights(BaseModel):
    recency: float = 0.1
    importance: float = 0.2
    relevance: float = 0.55
    proximity: float = 0.15
    decay_lambda: float = 0.01

    @classmethod
    def companion(cls) -> "ScoringWeights":
        return cls(relevance=0.3, importance=0.3, recency=0.1, proximity=0.3)

    @classmethod
    def enterprise(cls) -> "ScoringWeights":
        return cls(relevance=0.4, importance=0.2, recency=0.3, proximity=0.1)


# ── Search Result ──

class SearchResult(BaseModel):
    node: MemoryNode
    score: float
    scoring_breakdown: Dict[str, float] = Field(default_factory=dict)
    path_from_anchor: Optional[List[str]] = None
    source_text: Optional[str] = None   # raw episode text from Layer 2
    low_confidence: bool = False        # flag for uncertain retrieval
