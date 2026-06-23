import asyncio
from types import SimpleNamespace

from bson import ObjectId

from app.core.config import settings
from app.rag.indexing import index_paper


class WordTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()


class FakeEmbeddingProvider:
    model_id = "fake-minilm"
    tokenizer = WordTokenizer()

    def embed(self, texts):
        return [[float(len(text.split())), 1.0] for text in texts]


class FakeVectorStore:
    def __init__(self) -> None:
        self.paper_id = None
        self.records = []

    def replace_paper(self, paper_id, records) -> None:
        self.paper_id = paper_id
        self.records = list(records)


class FakePapersCollection:
    def __init__(self, paper) -> None:
        self.paper = paper
        self.updates = []

    async def find_one(self, query):
        return self.paper if query.get("_id") == self.paper["_id"] else None

    async def update_one(self, query, update):
        self.updates.append(update)
        return SimpleNamespace(modified_count=1)


class FakeChunksCollection:
    def __init__(self) -> None:
        self.documents = []
        self.indexes = []

    async def create_index(self, fields):
        self.indexes.append(fields)

    async def delete_many(self, query):
        self.documents = []

    async def insert_many(self, documents):
        self.documents = list(documents)
        return SimpleNamespace(inserted_ids=[document["_id"] for document in documents])


class FakeDatabase:
    def __init__(self, paper) -> None:
        self.papers = FakePapersCollection(paper)
        self.chunks = FakeChunksCollection()

    def __getitem__(self, name):
        if name == settings.MONGODB_COLLECTION_PAPERS:
            return self.papers
        if name == settings.MONGODB_COLLECTION_RAG_CHUNKS:
            return self.chunks
        raise KeyError(name)


def test_index_paper_is_repeatable_and_keeps_full_text_in_mongodb() -> None:
    paper_id = ObjectId()
    database = FakeDatabase(
        {
            "_id": paper_id,
            "text": "First page evidence.\nSecond page evidence.",
            "pages": [
                {"page": 1, "text": "First page evidence.", "start_char": 0, "end_char": 20},
                {"page": 2, "text": "Second page evidence.", "start_char": 21, "end_char": 42},
            ],
        }
    )
    vector_store = FakeVectorStore()

    first = asyncio.run(index_paper(database, paper_id, FakeEmbeddingProvider(), vector_store))
    second = asyncio.run(index_paper(database, paper_id, FakeEmbeddingProvider(), vector_store))

    assert first["status"] == second["status"] == "indexed"
    assert first["chunk_count"] == second["chunk_count"] == 2
    assert len(database.chunks.documents) == 2
    assert len(vector_store.records) == 2
    assert vector_store.paper_id == str(paper_id)
    assert database.chunks.documents[0]["text"] == "First page evidence."
    assert database.chunks.documents[1]["page"] == 2
