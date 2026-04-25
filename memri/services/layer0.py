"""Layer 0 generation and management — the always-in-context routing index."""

import re
from datetime import datetime
from typing import List, Dict

from ..graph.types import Layer0, NodeType, EdgeType
from ..graph.store import GraphStore
from ..llm.graph_adapter import GraphLLMAdapter as LLMProvider

SUMMARY_PROMPT = """Summarize this person's conversation history in 2-3 sentences.

Facts:
{facts_text}

Respond as JSON (no other text):
{{
  "user_summary": "2-3 sentence summary",
  "active_topics": ["topic1", "topic2", "topic3"],
  "emotional_state": "brief assessment"
}}"""

CLUSTER_LABEL_PROMPT = """Give each group of facts a short topic label (2-4 words max).
The label will be used to match search queries — use common words a person would use.

Groups:
{groups_text}

Respond as JSON mapping group_id to label:
{{"0": "music preferences", "1": "childhood books", "2": "painting", "3": "yoga practice"}}"""


def _build_aliases(entity_names: List[str], facts_text: str) -> Dict[str, str]:
    """Build alias → canonical mapping from entity names + fact content.

    Detects:
    - First name of a full name ("Mel" → "Melanie" when Melanie is known)
    - Relationship terms ("mom", "husband") → person from fact content
    - All-caps abbreviations mapped to longer form
    """
    aliases: Dict[str, str] = {}
    name_list = [n for n in entity_names if n and len(n) > 1]

    for canonical in name_list:
        parts = canonical.split()
        if len(parts) >= 2:
            # First name as alias
            first = parts[0].lower()
            if first not in aliases and len(first) >= 3:
                aliases[first] = canonical
            # Last name as alias
            last = parts[-1].lower()
            if last not in aliases and len(last) >= 3:
                aliases[last] = canonical

    # Relationship alias extraction from facts
    # Pattern: "X's <relation>" or "<relation> named X" or "my <relation> X"
    rel_patterns = [
        (r"(?:my|her|his)\s+(mom|mother|dad|father|sister|brother|husband|wife|boyfriend|girlfriend|son|daughter)\s+(\w+)", 2),
        (r"(\w+)'s?\s+(?:mom|mother|dad|father|sister|brother|husband|wife|son|daughter)", 1),
    ]
    for line in facts_text.split("\n"):
        for pattern, name_group in rel_patterns:
            m = re.search(pattern, line, re.IGNORECASE)
            if m:
                person_name = m.group(name_group).strip().capitalize()
                if person_name in entity_names:
                    rel_match = re.search(r"(mom|mother|dad|father|sister|brother|husband|wife|boyfriend|girlfriend|son|daughter)", line, re.IGNORECASE)
                    if rel_match:
                        aliases[rel_match.group(1).lower()] = person_name

    return aliases


def _build_topic_clusters_from_graph(graph: GraphStore) -> Dict[str, List[str]]:
    """
    Session-based clustering: group facts by session_index.
    Each conversation session is a natural topic cluster.
    Returns {session_key: [fact_id, ...]}
    """
    facts = graph.get_nodes_by_type(NodeType.FACT)

    session_clusters: Dict[int, List[str]] = {}
    for fact in facts:
        idx = fact.session_index if fact.session_index is not None else -1
        session_clusters.setdefault(idx, []).append(fact.id)

    return {f"session_{k}": v for k, v in sorted(session_clusters.items())}


class Layer0Service:
    def __init__(self, graph: GraphStore, llm: LLMProvider):
        self.graph = graph
        self.llm = llm

    async def generate(self) -> Layer0:
        """Regenerate Layer 0 from current graph state."""
        facts = self.graph.get_nodes_by_type(NodeType.FACT)
        entities = self.graph.get_nodes_by_type(NodeType.ENTITY)
        reflections = self.graph.get_nodes_by_type(NodeType.REFLECTION)

        layer0 = Layer0()

        # ── Build entity_index (programmatic, no LLM) ──
        entity_index = {}
        for entity_node in entities:
            fact_ids = [
                nid for nid in self.graph.get_neighbors(entity_node.id, edge_type=EdgeType.BELONGS_TO)
                if (n := self.graph.get_node(nid)) and n.node_type == NodeType.FACT
            ]
            if fact_ids:
                entity_index[entity_node.content] = fact_ids
        layer0.entity_index = entity_index

        # ── Build topic clusters (graph-based, no LLM) ──
        raw_clusters = _build_topic_clusters_from_graph(self.graph)

        # One cheap LLM call to label the clusters
        if raw_clusters and facts:
            labeled_clusters = await self._label_clusters(raw_clusters)
            layer0.topic_clusters = labeled_clusters
        else:
            layer0.topic_clusters = {}

        # ── LLM summary (one call for user_summary + active_topics) ──
        if facts:
            top_facts = sorted(facts, key=lambda n: n.importance, reverse=True)[:25]
            facts_text = "\n".join(f"- {f.content}" for f in top_facts)
            try:
                result = await self.llm.generate_json(
                    SUMMARY_PROMPT.format(facts_text=facts_text)
                )
                if isinstance(result, dict):
                    layer0.user_summary = result.get("user_summary", "")
                    layer0.active_topics = result.get("active_topics", [])[:5]
                    layer0.emotional_state = result.get("emotional_state", "")
            except Exception as e:
                print(f"[layer0] Summary failed: {e}")
                layer0.user_summary = f"{len(facts)} memories across {len(entities)} entities."

        # ── Build aliases (programmatic, no LLM) ──
        all_entity_names = [e.content for e in entities]
        all_facts_text = "\n".join(f.content for f in facts[:200])
        layer0.aliases = _build_aliases(all_entity_names, all_facts_text)

        layer0.fact_count = len(facts)
        layer0.reflection_count = len(reflections)
        layer0.last_updated = datetime.utcnow()
        return layer0

    async def _label_clusters(self, clusters: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """One LLM call to label all clusters with human-readable topic names."""
        groups_lines = []
        cluster_items = list(clusters.items())[:30]  # up to 30 session clusters
        for idx, (cluster_key, fact_ids) in enumerate(cluster_items):
            sample_facts = []
            session_date = ""
            # Sort by importance descending, sample top 5
            nodes = [self.graph.get_node(fid) for fid in fact_ids if self.graph.get_node(fid)]
            nodes.sort(key=lambda n: n.importance, reverse=True)
            for node in nodes[:5]:
                sample_facts.append(node.content)
                if not session_date and node.session_date:
                    session_date = node.session_date
            if sample_facts:
                date_str = f" [{session_date}]" if session_date else ""
                groups_lines.append(f'Group {idx}{date_str}: {" | ".join(sample_facts[:3])}')

        if not groups_lines:
            return {}

        try:
            result = await self.llm.generate_json(
                CLUSTER_LABEL_PROMPT.format(groups_text="\n".join(groups_lines))
            )
            if not isinstance(result, dict):
                return {}

            labeled = {}
            for idx, (cluster_key, fact_ids) in enumerate(cluster_items):
                label = result.get(str(idx), cluster_key)
                # Normalize label: lowercase, spaces (for BM25 token matching)
                label = str(label).lower().strip()
                labeled[label] = fact_ids
            return labeled
        except Exception as e:
            print(f"[layer0] Cluster labeling failed: {e}")
            return {k: v for k, v in cluster_items}
