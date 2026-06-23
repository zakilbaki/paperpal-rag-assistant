from __future__ import annotations

import asyncio
from typing import Any

from bson import ObjectId

from app.core.config import settings
from app.rag.context import build_context
from app.rag.interfaces import EmbeddingProvider, VectorSearchFilters, VectorStore
from app.rag.reranking import rerank_chunks


class PaperNotIndexedError(RuntimeError):
    pass


async def retrieve_chunks(
    db,
    paper_id: ObjectId,
    question: str,
    top_k: int,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
) -> dict[str, Any]:
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty")
    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10")

    papers = db[settings.MONGODB_COLLECTION_PAPERS]
    paper = await papers.find_one({"_id": paper_id})
    if not paper:
        raise LookupError("Paper not found")

    index_state = paper.get("rag_index") or {}
    if index_state.get("status") != "indexed":
        raise PaperNotIndexedError("Paper has not been indexed")
    if index_state.get("embedding_model") != embedding_provider.model_id:
        raise PaperNotIndexedError("Paper index uses a different embedding model")
    if index_state.get("index_version") != settings.RAG_INDEX_VERSION:
        raise PaperNotIndexedError("Paper index version is outdated")

    vectors = await asyncio.to_thread(embedding_provider.embed, [question])
    if len(vectors) != 1:
        raise RuntimeError("Embedding provider returned an unexpected vector count")
    matches = await asyncio.to_thread(
        vector_store.search,
        VectorSearchFilters(
            paper_id=str(paper_id),
            index_version=settings.RAG_INDEX_VERSION,
        ),
        vectors[0],
        max(top_k, settings.RAG_RETRIEVAL_CANDIDATES),
    )

    chunk_ids = [match.chunk_id for match in matches]
    if not chunk_ids:
        documents = []
    else:
        cursor = db[settings.MONGODB_COLLECTION_RAG_CHUNKS].find(
            {"_id": {"$in": chunk_ids}, "paper_id": paper_id}
        )
        documents = await cursor.to_list(length=len(chunk_ids))
    documents_by_id = {str(document["_id"]): document for document in documents}

    candidates = []
    for match in matches:
        document = documents_by_id.get(match.chunk_id)
        if not document:
            continue
        candidates.append(
            {
                "chunk_id": match.chunk_id,
                "text": document["text"],
                "score": match.score,
                "page": document.get("page"),
                "section": document.get("section"),
                "start_char": document["start_char"],
                "end_char": document["end_char"],
            }
        )
    results = rerank_chunks(question, candidates, top_k)

    context = build_context(
        results,
        embedding_provider.tokenizer,
        settings.RAG_CONTEXT_MAX_TOKENS,
        settings.RAG_CONTEXT_MAX_PASSAGES,
    )

    return {
        "status": "success",
        "paper_id": str(paper_id),
        "question": question,
        "results": results,
        "context": context,
        "embedding_model": embedding_provider.model_id,
        "index_version": settings.RAG_INDEX_VERSION,
    }
