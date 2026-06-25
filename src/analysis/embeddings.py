import logging
from typing import Optional

import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate text embeddings using sentence-transformers."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embedding_model
        self._model = None

    def _load_model(self):
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded")

    def encode(self, texts: list[str]) -> np.ndarray:
        self._load_model()
        logger.info("Encoding %d texts", len(texts))
        embeddings = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return np.array(embeddings, dtype=np.float32)

    @property
    def dimension(self) -> int:
        self._load_model()
        return self._model.get_sentence_embedding_dimension()
