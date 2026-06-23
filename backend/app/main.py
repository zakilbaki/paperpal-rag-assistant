from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import compare, health, papers, rag, summarize, upload
from app.core.config import settings
from app.db.mongo import close_client


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await close_client()


app = FastAPI(
    title="PaperPal API",
    version="2.0.0",
    description=(
        "Scientific document intelligence API for PDF ingestion, summarization, "
        "keyword extraction, and document comparison."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1/health", tags=["Health"])
app.include_router(papers.router, prefix="/api/v1/papers", tags=["Papers"])
app.include_router(summarize.router, prefix="/api/v1/papers", tags=["Summarization"])
app.include_router(compare.router, prefix="/api/v1/papers", tags=["Comparison"])
app.include_router(upload.router, prefix="/api/v1/papers", tags=["Upload"])
app.include_router(rag.router, prefix="/api/v1/papers", tags=["RAG indexing"])


@app.get("/", tags=["Meta"])
async def root() -> dict[str, str]:
    return {"service": "PaperPal API", "docs": "/docs"}
