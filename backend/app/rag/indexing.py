from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
from typing import Any

from bson import ObjectId
from pymongo import ASCENDING

from app.core.config import settings
from app.rag.chunking import chunk_page
from app.rag.interfaces import EmbeddingProvider, VectorRecord, VectorStore


def _chunk_id(paper_id: str, page: int | None, start_char: int, end_char: int) -> str:
    value = f"{settings.RAG_INDEX_VERSION}:{paper_id}:{page}:{start_char}:{end_char}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


async def index_paper(
    db,
    paper_id: ObjectId,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
) -> dict[str, Any]:
    paper = await db[settings.MONGODB_COLLECTION_PAPERS].find_one({"_id": paper_id})
    if not paper:
        raise LookupError("Paper not found")

    pages = paper.get("pages") or [
        {
            "page": None,
            "text": paper.get("text", ""),
            "start_char": 0,
            "end_char": len(paper.get("text", "")),
        }
    ]
    if not any(str(page.get("text", "")).strip() for page in pages):
        raise ValueError("Paper text is empty")

    paper_id_text = str(paper_id)
    now = dt.datetime.now(dt.timezone.utc)
    papers = db[settings.MONGODB_COLLECTION_PAPERS]
    chunks_collection = db[settings.MONGODB_COLLECTION_RAG_CHUNKS]
    await papers.update_one(
        {"_id": paper_id},
        {
            "$set": {
                "rag_index.status": "indexing",
                "rag_index.index_version": settings.RAG_INDEX_VERSION,
                "rag_index.embedding_model": embedding_provider.model_id,
                "rag_index.updated_at": now,
            }
        },
    )

    try:
        tokenizer = await asyncio.to_thread(lambda: embedding_provider.tokenizer)
        chunk_documents: list[dict[str, Any]] = []
        ordinal = 0

        for page in pages:
            page_text = str(page.get("text", ""))
            if not page_text.strip():
                continue
            page_chunks, _ = chunk_page(
                page_text,
                tokenizer,
                max_tokens=settings.RAG_CHUNK_TOKENS,
                overlap=settings.RAG_CHUNK_OVERLAP,
            )
            page_start = int(page.get("start_char", 0))
            page_number = page.get("page")
            for chunk in page_chunks:
                global_start = page_start + chunk.start_char
                global_end = page_start + chunk.end_char
                chunk_documents.append(
                    {
                        "_id": _chunk_id(paper_id_text, page_number, global_start, global_end),
                        "paper_id": paper_id,
                        "text": chunk.text,
                        "page": page_number,
                        "section": chunk.section,
                        "start_char": global_start,
                        "end_char": global_end,
                        "ordinal": ordinal,
                        "index_version": settings.RAG_INDEX_VERSION,
                        "embedding_model": embedding_provider.model_id,
                        "created_at": now,
                    }
                )
                ordinal += 1

        embeddings = await asyncio.to_thread(
            embedding_provider.embed,
            [document["text"] for document in chunk_documents],
        )
        if len(embeddings) != len(chunk_documents):
            raise RuntimeError("Embedding provider returned an unexpected vector count")

        await chunks_collection.create_index(
            [("paper_id", ASCENDING), ("index_version", ASCENDING)]
        )
        await chunks_collection.delete_many({"paper_id": paper_id})
        if chunk_documents:
            await chunks_collection.insert_many(chunk_documents)

        vector_records = []
        for document, embedding in zip(chunk_documents, embeddings):
            metadata: dict[str, str | int] = {
                "paper_id": paper_id_text,
                "index_version": settings.RAG_INDEX_VERSION,
                "ordinal": document["ordinal"],
            }
            if document["page"] is not None:
                metadata["page"] = int(document["page"])
            if document["section"]:
                metadata["section"] = str(document["section"])
            vector_records.append(
                VectorRecord(
                    chunk_id=document["_id"],
                    embedding=embedding,
                    metadata=metadata,
                )
            )
        await asyncio.to_thread(vector_store.replace_paper, paper_id_text, vector_records)

        await papers.update_one(
            {"_id": paper_id},
            {
                "$set": {
                    "rag_index.status": "indexed",
                    "rag_index.chunk_count": len(chunk_documents),
                    "rag_index.updated_at": dt.datetime.now(dt.timezone.utc),
                },
                "$unset": {"rag_index.error": ""},
            },
        )
        return {
            "status": "indexed",
            "paper_id": paper_id_text,
            "chunk_count": len(chunk_documents),
            "embedding_model": embedding_provider.model_id,
            "index_version": settings.RAG_INDEX_VERSION,
        }
    except Exception as exc:
        await papers.update_one(
            {"_id": paper_id},
            {
                "$set": {
                    "rag_index.status": "failed",
                    "rag_index.error": str(exc),
                    "rag_index.updated_at": dt.datetime.now(dt.timezone.utc),
                }
            },
        )
        raise
