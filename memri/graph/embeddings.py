"""ChromaDB-backed embedding store. Uses local sentence-transformers — no API keys."""

try:
    import chromadb
except ImportError:
    raise ImportError(
        "chromadb is required for graph-based memory. "
        "Install with: pip install 'memri[graph]'"
    )

from typing import List, Tuple, Optional


class EmbeddingStore:
    def __init__(self, path: str):
        self.client = chromadb.PersistentClient(path=path)
        # Use cosine metric so 1 - distance = cosine similarity
        self.collection = self.client.get_or_create_collection(
            "memri_memories",
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, node_id: str, text: str, metadata: Optional[dict] = None) -> None:
        """Add or update a node's embedding."""
        self.collection.upsert(
            ids=[node_id],
            documents=[text],
            metadatas=[metadata or {}],
        )

    def similarity(self, query: str, node_id: str) -> float:
        """Cosine similarity between query embedding and a specific node."""
        count = self.collection.count()
        if count == 0:
            return 0.0
        n = min(count, 50)
        try:
            results = self.collection.query(query_texts=[query], n_results=n)
            for i, rid in enumerate(results["ids"][0]):
                if rid == node_id:
                    return 1.0 - results["distances"][0][i]
        except Exception:
            pass
        return 0.0

    def search(self, query: str, n_results: int = 20) -> List[Tuple[str, float]]:
        """Vector search → list of (node_id, similarity_score)."""
        count = self.collection.count()
        if count == 0:
            return []
        n = min(count, n_results)
        try:
            results = self.collection.query(query_texts=[query], n_results=n)
            return [
                (rid, 1.0 - dist)
                for rid, dist in zip(results["ids"][0], results["distances"][0])
            ]
        except Exception:
            return []

    def delete(self, node_id: str) -> None:
        try:
            self.collection.delete(ids=[node_id])
        except Exception:
            pass
