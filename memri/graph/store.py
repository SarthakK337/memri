"""NetworkX-backed graph store for memory nodes and edges."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import networkx as nx

from .types import MemoryNode, Edge, NodeType, EdgeType


class GraphStore:
    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()
        self.nodes: Dict[str, MemoryNode] = {}
        self._undirected_cache: Optional[nx.Graph] = None  # invalidated on add_edge

    def _get_undirected(self) -> nx.Graph:
        if self._undirected_cache is None:
            self._undirected_cache = self.graph.to_undirected()
        return self._undirected_cache

    def add_node(self, node: MemoryNode) -> None:
        self.nodes[node.id] = node
        self.graph.add_node(node.id, node_type=node.node_type.value)

    def add_edge(self, edge: Edge) -> None:
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            edge_type=edge.edge_type.value,
            weight=edge.weight,
        )
        self._undirected_cache = None  # invalidate cache

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        return self.nodes.get(node_id)

    def find_or_create_entity(self, name: str) -> MemoryNode:
        """Case-insensitive search for existing entity, or create new."""
        for node in self.nodes.values():
            if node.node_type == NodeType.ENTITY and node.content.lower() == name.lower():
                return node
        entity = MemoryNode(node_type=NodeType.ENTITY, content=name)
        self.add_node(entity)
        return entity

    def get_nodes_by_type(self, node_type: NodeType) -> List[MemoryNode]:
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[EdgeType] = None,
        max_hops: int = 1,
    ) -> Set[str]:
        """Get all nodes within max_hops of node_id (both directions)."""
        if node_id not in self.graph:
            return set()

        if max_hops == 1:
            neighbors = set()
            for _, target, data in self.graph.edges(node_id, data=True):
                if edge_type is None or data.get("edge_type") == edge_type.value:
                    neighbors.add(target)
            for source, _, data in self.graph.in_edges(node_id, data=True):
                if edge_type is None or data.get("edge_type") == edge_type.value:
                    neighbors.add(source)
            return neighbors
        else:
            # BFS up to max_hops
            visited: Set[str] = set()
            queue = [(node_id, 0)]
            while queue:
                current, depth = queue.pop(0)
                if depth > max_hops or current in visited:
                    continue
                visited.add(current)
                for neighbor in self.graph.predecessors(current):
                    queue.append((neighbor, depth + 1))
                for neighbor in self.graph.successors(current):
                    queue.append((neighbor, depth + 1))
            visited.discard(node_id)
            return visited

    def get_neighborhood(self, node_id: str, max_hops: int = 2) -> Set[str]:
        """Get all nodes within max_hops. Used for subgraph extraction."""
        return self.get_neighbors(node_id, max_hops=max_hops)

    def shortest_path_length(self, source: str, target: str) -> Optional[int]:
        try:
            return nx.shortest_path_length(self._get_undirected(), source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def has_path(self, source: str, target: str) -> bool:
        try:
            return nx.has_path(self._get_undirected(), source, target)
        except nx.NodeNotFound:
            return False

    # ── v1 traversal methods ──────────────────────────────────────────────────

    def get_temporal_chain(
        self,
        seed_id: str,
        window_days: Optional[int] = None,
    ) -> List[MemoryNode]:
        """Walk HAPPENED_BEFORE/AFTER/DURING edges from seed, return facts sorted by temporal_date."""
        temporal_edges = {
            EdgeType.HAPPENED_BEFORE.value,
            EdgeType.HAPPENED_AFTER.value,
            EdgeType.HAPPENED_DURING.value,
        }
        visited: Set[str] = set()
        queue = [seed_id]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for _, target, data in self.graph.edges(current, data=True):
                if data.get("edge_type") in temporal_edges and target not in visited:
                    queue.append(target)
            for source, _, data in self.graph.in_edges(current, data=True):
                if data.get("edge_type") in temporal_edges and source not in visited:
                    queue.append(source)

        facts = [
            self.nodes[nid] for nid in visited
            if nid in self.nodes and self.nodes[nid].node_type == NodeType.FACT
        ]

        if window_days is not None:
            cutoff = datetime.utcnow() - timedelta(days=window_days)
            # Keep facts with no date (can't filter) or within window
            facts = [
                f for f in facts
                if not f.temporal_date or f.created_at >= cutoff
            ]

        facts.sort(key=lambda f: (f.temporal_date or "", f.session_index or 0))
        return facts

    def get_causal_chain(self, seed_id: str, max_depth: int = 3) -> List[MemoryNode]:
        """Walk CAUSED/COPING_WITH edges bidirectionally, return facts cause→effect ordered."""
        causal_edges = {EdgeType.CAUSED.value, EdgeType.COPING_WITH.value}
        visited: Set[str] = set()
        queue = [(seed_id, 0)]
        while queue:
            current, depth = queue.pop(0)
            if current in visited or depth > max_depth:
                continue
            visited.add(current)
            for _, target, data in self.graph.edges(current, data=True):
                if data.get("edge_type") in causal_edges and target not in visited:
                    queue.append((target, depth + 1))
            for source, _, data in self.graph.in_edges(current, data=True):
                if data.get("edge_type") in causal_edges and source not in visited:
                    queue.append((source, depth + 1))

        facts = [
            self.nodes[nid] for nid in visited
            if nid in self.nodes and self.nodes[nid].node_type == NodeType.FACT
        ]
        facts.sort(key=lambda f: (f.session_index or 0, f.created_at))
        return facts

    def get_entity_neighborhood(self, entity_id: str, max_hops: int = 2) -> Set[str]:
        """BFS from entity node along BELONGS_TO + RELATED_TO, returning fact node IDs."""
        semantic_edges = {EdgeType.BELONGS_TO.value, EdgeType.RELATED_TO.value}
        visited: Set[str] = set()
        queue = [(entity_id, 0)]
        fact_ids: Set[str] = set()
        while queue:
            current, depth = queue.pop(0)
            if current in visited or depth > max_hops:
                continue
            visited.add(current)
            node = self.nodes.get(current)
            if node and node.node_type == NodeType.FACT:
                fact_ids.add(current)
            for _, target, data in self.graph.edges(current, data=True):
                if data.get("edge_type") in semantic_edges and target not in visited:
                    queue.append((target, depth + 1))
            for source, _, data in self.graph.in_edges(current, data=True):
                if data.get("edge_type") in semantic_edges and source not in visited:
                    queue.append((source, depth + 1))
        return fact_ids

    def get_connected_facts(
        self,
        seed_ids: List[str],
        edge_types: List[EdgeType],
        max_hops: int = 1,
    ) -> List[MemoryNode]:
        """Generic BFS from seed_ids filtering by edge_types. Returns deduplicated fact nodes."""
        allowed = {et.value for et in edge_types}
        visited: Set[str] = set()
        queue = [(nid, 0) for nid in seed_ids]
        fact_ids: Set[str] = set()
        while queue:
            current, depth = queue.pop(0)
            if current in visited or depth > max_hops:
                continue
            visited.add(current)
            node = self.nodes.get(current)
            if node and node.node_type == NodeType.FACT:
                fact_ids.add(current)
            for _, target, data in self.graph.edges(current, data=True):
                if data.get("edge_type") in allowed and target not in visited:
                    queue.append((target, depth + 1))
            for source, _, data in self.graph.in_edges(current, data=True):
                if data.get("edge_type") in allowed and source not in visited:
                    queue.append((source, depth + 1))

        facts = [self.nodes[fid] for fid in fact_ids if fid in self.nodes]
        facts.sort(key=lambda f: (f.session_index or 0, f.created_at))
        return facts
