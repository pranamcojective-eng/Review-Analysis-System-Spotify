import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """Simple FAISS-backed vector store for review embeddings."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self._index = None
        self._ids: list[str] = []
        self._texts: list[str] = []

    def _ensure_index(self):
        if self._index is None:
            import faiss
            self._index = faiss.IndexFlatIP(self.dimension)
            logger.info("FAISS index created (dim=%d)", self.dimension)

    def add(self, ids: list[str], embeddings: np.ndarray, texts: list[str]) -> None:
        self._ensure_index()
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Expected dim {self.dimension}, got {embeddings.shape[1]}")
        self._index.add(embeddings.astype(np.float32))
        self._ids.extend(ids)
        self._texts.extend(texts)
        logger.info("Added %d vectors to FAISS (total=%d)", len(ids), len(self._ids))

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[dict]:
        self._ensure_index()
        if len(self._ids) == 0:
            return []
        k = min(k, len(self._ids))
        query = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(query, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({
                "id": self._ids[idx],
                "text": self._texts[idx],
                "score": float(score),
            })
        return results

    @property
    def size(self) -> int:
        return len(self._ids)

    def save_metadata(self, path: str) -> None:
        import json
        from pathlib import Path
        data = {"ids": self._ids, "texts": self._texts, "dimension": self.dimension}
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Vector store metadata saved to %s", path)
