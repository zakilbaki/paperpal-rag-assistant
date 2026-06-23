import asyncio

import pytest
from bson import ObjectId

from app.core.config import settings
from app.rag.interfaces import VectorSearchResult
from app.rag.retrieval import PaperNotIndexedError, retrieve_chunks


class FakeEmbeddingProvider:
    model_id = "fake-minilm"
    tokenizer = None

    def embed(self, texts):
        return [[1.0, float(len(text))] for text in texts]


class WordTokenizer:
    def encode(self, text, add_special_tokens=False, **kwargs):
        return text.split()


FakeEmbeddingProvider.tokenizer = WordTokenizer()


class FakeVectorStore:
    def __init__(self, matches) -> None:
        self.matches = matches
        self.search_call = None

    def search(self, filters, query_embedding, top_k):
        self.search_call = (filters, query_embedding, top_k)
        return self.matches[:top_k]


class FakeCursor:
    def __init__(self, documents) -> None:
        self.documents = documents

    async def to_list(self, length):
        return self.documents[:length]


class FakePapersCollection:
    def __init__(self, paper) -> None:
        self.paper = paper

    async def find_one(self, query):
        return self.paper if query.get("_id") == self.paper["_id"] else None


class FakeChunksCollection:
    def __init__(self, documents) -> None:
        self.documents = documents
        self.last_query = None

    def find(self, query):
        self.last_query = query
        requested = set(query["_id"]["$in"])
        paper_id = query["paper_id"]
        matching = [
            document
            for document in self.documents
            if document["_id"] in requested and document["paper_id"] == paper_id
        ]
        return FakeCursor(matching)


class FakeDatabase:
    def __init__(self, paper, chunks) -> None:
        self.papers = FakePapersCollection(paper)
        self.chunks = FakeChunksCollection(chunks)

    def __getitem__(self, name):
        if name == settings.MONGODB_COLLECTION_PAPERS:
            return self.papers
        if name == settings.MONGODB_COLLECTION_RAG_CHUNKS:
            return self.chunks
        raise KeyError(name)


def indexed_paper(paper_id):
    return {
        "_id": paper_id,
        "rag_index": {
            "status": "indexed",
            "embedding_model": "fake-minilm",
            "index_version": settings.RAG_INDEX_VERSION,
        },
    }


def test_retrieve_chunks_hydrates_mongodb_in_vector_ranking_order() -> None:
    paper_id = ObjectId()
    database = FakeDatabase(
        indexed_paper(paper_id),
        [
            {
                "_id": "second",
                "paper_id": paper_id,
                "text": "Second result",
                "page": 4,
                "section": "results",
                "start_char": 200,
                "end_char": 260,
            },
            {
                "_id": "first",
                "paper_id": paper_id,
                "text": "First result",
                "page": 2,
                "section": "methods",
                "start_char": 100,
                "end_char": 160,
            },
        ],
    )
    vector_store = FakeVectorStore(
        [
            VectorSearchResult("first", 0.82),
            VectorSearchResult("second", 0.71),
        ]
    )

    result = asyncio.run(
        retrieve_chunks(
            database,
            paper_id,
            "  What method was used?  ",
            5,
            FakeEmbeddingProvider(),
            vector_store,
        )
    )

    assert result["question"] == "What method was used?"
    assert [item["chunk_id"] for item in result["results"]] == ["first", "second"]
    assert result["results"][0]["page"] == 2
    assert vector_store.search_call[0].paper_id == str(paper_id)
    assert vector_store.search_call[0].index_version == settings.RAG_INDEX_VERSION
    assert vector_store.search_call[2] == 5
    assert database.chunks.last_query["paper_id"] == paper_id
    assert result["context"]["evidence_available"] is True
    assert [item["citation"] for item in result["context"]["evidence"]] == ["C1", "C2"]
    assert result["context"]["evidence"][0]["source_chunk_ids"] == ["first"]


def test_retrieve_chunks_rejects_unindexed_paper() -> None:
    paper_id = ObjectId()
    database = FakeDatabase({"_id": paper_id}, [])

    with pytest.raises(PaperNotIndexedError, match="has not been indexed"):
        asyncio.run(
            retrieve_chunks(
                database,
                paper_id,
                "Question",
                5,
                FakeEmbeddingProvider(),
                FakeVectorStore([]),
            )
        )


@pytest.mark.parametrize("question, top_k", [("   ", 5), ("Question", 0), ("Question", 11)])
def test_retrieve_chunks_validates_request(question, top_k) -> None:
    paper_id = ObjectId()
    database = FakeDatabase(indexed_paper(paper_id), [])

    with pytest.raises(ValueError):
        asyncio.run(
            retrieve_chunks(
                database,
                paper_id,
                question,
                top_k,
                FakeEmbeddingProvider(),
                FakeVectorStore([]),
            )
        )
