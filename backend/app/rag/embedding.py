from __future__ import annotations

import threading
from functools import lru_cache
from typing import Sequence

from app.core.config import settings


class MiniLMEmbeddingProvider:
    """Lazy CPU-only Sentence Transformers embedding provider."""

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self._model = None
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return self._model_id

    def _get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self._model_id, device="cpu")
        return self._model

    @property
    def tokenizer(self):
        return self._get_model().tokenizer

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._get_model().encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()


@lru_cache(maxsize=1)
def get_embedding_provider() -> MiniLMEmbeddingProvider:
    return MiniLMEmbeddingProvider(settings.EMBEDDING_MODEL_NAME)
