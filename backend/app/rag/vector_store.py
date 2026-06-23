from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

from app.core.config import settings
from app.rag.interfaces import VectorRecord, VectorSearchFilters, VectorSearchResult


class ChromaVectorStore:
    """Persistent Chroma adapter containing embeddings and minimal metadata only."""

    def __init__(self, path: str, collection_name: str) -> None:
        self.path = path
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            import chromadb

            Path(self.path).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.path)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def replace_paper(self, paper_id: str, records: Sequence[VectorRecord]) -> None:
        collection = self._get_collection()
        collection.delete(where={"paper_id": paper_id})
        if not records:
            return
        collection.upsert(
            ids=[record.chunk_id for record in records],
            embeddings=[record.embedding for record in records],
            metadatas=[record.metadata for record in records],
        )

    def search(
        self,
        filters: VectorSearchFilters,
        query_embedding: Sequence[float],
        top_k: int,
    ) -> list[VectorSearchResult]:
        conditions: list[dict] = [
            {"paper_id": filters.paper_id},
            {"index_version": filters.index_version},
        ]
        result = self._get_collection().query(
            query_embeddings=[list(query_embedding)],
            n_results=top_k,
            where={"$and": conditions},
            include=["distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [
            VectorSearchResult(chunk_id=chunk_id, score=1.0 - float(distance))
            for chunk_id, distance in zip(ids, distances)
        ]


@lru_cache(maxsize=1)
def get_vector_store() -> ChromaVectorStore:
    return ChromaVectorStore(settings.CHROMA_PATH, settings.CHROMA_COLLECTION)
