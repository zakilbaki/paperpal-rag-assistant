from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, HTTPException

from app.db.mongo import get_database
from app.models.rag import RagIndexResponse, RagRetrieveRequest, RagRetrieveResponse
from app.rag.embedding import get_embedding_provider
from app.rag.indexing import index_paper
from app.rag.retrieval import PaperNotIndexedError, retrieve_chunks
from app.rag.vector_store import get_vector_store


router = APIRouter(tags=["RAG indexing"])


@router.post("/{paper_id}/rag/index", response_model=RagIndexResponse)
async def index_paper_endpoint(
    paper_id: str,
    db=Depends(get_database),
    embedding_provider=Depends(get_embedding_provider),
    vector_store=Depends(get_vector_store),
):
    """Create or replace the retrieval index for one stored paper."""
    try:
        paper_oid = ObjectId(paper_id)
    except InvalidId as exc:
        raise HTTPException(status_code=400, detail="Invalid paper ID format") from exc

    try:
        result = await index_paper(db, paper_oid, embedding_provider, vector_store)
        return RagIndexResponse(**result)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="RAG indexing failed") from exc


@router.post("/{paper_id}/rag/retrieve", response_model=RagRetrieveResponse)
async def retrieve_paper_chunks_endpoint(
    paper_id: str,
    payload: RagRetrieveRequest,
    db=Depends(get_database),
    embedding_provider=Depends(get_embedding_provider),
    vector_store=Depends(get_vector_store),
):
    """Retrieve ranked, cited passages from one indexed paper."""
    try:
        paper_oid = ObjectId(paper_id)
    except InvalidId as exc:
        raise HTTPException(status_code=400, detail="Invalid paper ID format") from exc

    try:
        result = await retrieve_chunks(
            db,
            paper_oid,
            payload.question,
            payload.top_k,
            embedding_provider,
            vector_store,
        )
        return RagRetrieveResponse(**result)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PaperNotIndexedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="RAG retrieval failed") from exc
