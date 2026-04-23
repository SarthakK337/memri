"""Local semantic embeddings using sentence-transformers (optional dependency)."""

from __future__ import annotations

import struct
from typing import Optional

import numpy as np


class Embedder:
    """
    Wraps sentence-transformers for local embedding.
    Falls back to None if the package isn't installed.
    Call `available` before using.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast, good quality
    DIM = 384

    def __init__(self):
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            self._model = SentenceTransformer(self.MODEL_NAME)
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return self._model is not None

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        """Return list of unit-normalised float32 vectors."""
        if not self.available:
            raise RuntimeError("sentence-transformers not installed. Run: pip install memri[embeddings]")
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.astype(np.float32) for v in vecs]

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @staticmethod
    def to_blob(vec: np.ndarray) -> bytes:
        return vec.astype(np.float32).tobytes()

    @staticmethod
    def from_blob(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two unit-normalised vectors."""
        return float(np.dot(a, b))
