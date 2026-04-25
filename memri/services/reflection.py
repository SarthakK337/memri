"""Reflection service: synthesizes higher-level patterns from fact clusters."""

from typing import List, Dict
from datetime import datetime

from ..graph.types import MemoryNode, Edge, NodeType, EdgeType, Layer0
from ..graph.store import GraphStore
from ..llm.graph_adapter import GraphLLMAdapter as LLMProvider

REFLECTION_PROMPT = """You are analyzing a cluster of facts about "{cluster_topic}" from a user's conversation history.

Facts:
{facts_text}

Generate 1-2 higher-level reflections that capture patterns, habits, or recurring themes.
Each reflection should be something you'd only realize by looking across multiple facts.

Respond as JSON array (no other text):
[
  {{
    "content": "User frequently turns to their mother for emotional support during work stress.",
    "importance": 0.9,
    "source_fact_ids": ["fact_id_1", "fact_id_2"]
  }}
]"""

MIN_FACTS_FOR_REFLECTION = 3  # lowered from 5 so early testing works


class ReflectionService:
    def __init__(self, graph: GraphStore, llm: LLMProvider):
        self.graph = graph
        self.llm = llm

    async def generate_reflections(self, layer0: Layer0) -> List[MemoryNode]:
        """Generate reflections from fact clusters in Layer 0 topic groups."""
        facts = self.graph.get_nodes_by_type(NodeType.FACT)
        if not facts:
            return []

        # Build clusters from Layer 0 entity_index
        clusters: Dict[str, List[MemoryNode]] = {}

        for cluster_name, fact_ids in layer0.entity_index.items():
            cluster_facts = [
                self.graph.get_node(fid) for fid in fact_ids
                if self.graph.get_node(fid) is not None
            ]
            cluster_facts = [n for n in cluster_facts if n.node_type == NodeType.FACT]
            if len(cluster_facts) >= MIN_FACTS_FOR_REFLECTION:
                clusters[cluster_name] = cluster_facts

        # Also add a catch-all cluster for all facts if small dataset
        if not clusters and len(facts) >= MIN_FACTS_FOR_REFLECTION:
            clusters["general"] = facts

        new_reflections: List[MemoryNode] = []

        for cluster_topic, cluster_facts in clusters.items():
            facts_text = "\n".join(
                f"[{f.id}] {f.content}" for f in cluster_facts[:20]
            )
            try:
                result = await self.llm.generate_json(
                    REFLECTION_PROMPT.format(
                        cluster_topic=cluster_topic,
                        facts_text=facts_text,
                    )
                )
                if not isinstance(result, list):
                    continue

                for item in result:
                    content = item.get("content", "").strip()
                    if not content:
                        continue

                    reflection = MemoryNode(
                        node_type=NodeType.REFLECTION,
                        content=content,
                        importance=float(item.get("importance", 0.8)),
                        source_episode_id=None,
                    )
                    self.graph.add_node(reflection)
                    new_reflections.append(reflection)

                    # Link to source facts
                    for fact_id in item.get("source_fact_ids", []):
                        if self.graph.get_node(fact_id):
                            self.graph.add_edge(Edge(
                                source_id=reflection.id,
                                target_id=fact_id,
                                edge_type=EdgeType.DERIVED_FROM,
                            ))

            except Exception as e:
                print(f"[reflection] Cluster '{cluster_topic}' failed: {e}")

        return new_reflections
