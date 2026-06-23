from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


class EmbeddingProvider(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def tokenizer(self): ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class VectorRecord:
    chunk_id: str
    embedding: list[float]
    metadata: dict[str, str | int]


@dataclass(frozen=True)
class VectorSearchResult:
    chunk_id: str
    score: float


@dataclass(frozen=True)
class VectorSearchFilters:
    paper_id: str
    index_version: str


class VectorStore(Protocol):
    def replace_paper(self, paper_id: str, records: Sequence[VectorRecord]) -> None: ...

    def search(
        self,
        filters: VectorSearchFilters,
        query_embedding: Sequence[float],
        top_k: int,
    ) -> list[VectorSearchResult]: ...
