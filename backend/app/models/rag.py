from pydantic import BaseModel, Field


class RagIndexResponse(BaseModel):
    status: str
    paper_id: str
    chunk_count: int
    embedding_model: str
    index_version: str


class RagRetrieveRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    top_k: int = Field(default=5, ge=1, le=10)


class RagRetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    vector_score: float | None = None
    keyword_score: float | None = None
    vector_rank: int | None = None
    page: int | None = None
    section: str | None = None
    start_char: int
    end_char: int


class RagContextEvidence(BaseModel):
    citation: str
    source_chunk_ids: list[str]
    text: str
    score: float
    page: int | None = None
    section: str | None = None
    start_char: int
    end_char: int


class RagContextPacket(BaseModel):
    evidence_available: bool
    token_count: int
    context_text: str
    evidence: list[RagContextEvidence]


class RagRetrieveResponse(BaseModel):
    status: str
    paper_id: str
    question: str
    results: list[RagRetrievedChunk]
    context: RagContextPacket
    embedding_model: str
    index_version: str
