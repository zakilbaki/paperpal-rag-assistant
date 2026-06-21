from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal
from bson import ObjectId
from bson.errors import InvalidId

from ...services.summarize_llm import summarize_text
from ...services.db_summary import get_summary, save_summary
from ...db.mongo import get_database

router = APIRouter(tags=["Summarization"])

# -------------------------------------------------------
# 🧩 Request / Response Models
# -------------------------------------------------------
class SummarizeIn(BaseModel):
    paper_id: str = Field(..., description="MongoDB ObjectId of the paper")
    summary_type: Literal["short", "medium", "detailed"] = "medium"
    max_tokens: Optional[int] = Field(None, description="Override max tokens (optional)")
    strategy: str = "map_reduce"
    use_cache: bool = Field(False, description="If True, reuse cached summary instead of regenerating")


class SummarizeOut(BaseModel):
    paper_id: str
    summary_type: str
    summary: str
    chunks: int
    duration_ms: int
    cached: bool


# Async jobs (for background summarization)
class SummarizeAsyncRequest(BaseModel):
    paper_id: str
    summary_type: Literal["short", "medium", "detailed"] = "medium"
    use_cache: bool = True


class SummarizeAsyncResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "error"] = "queued"


class JobResult(BaseModel):
    status: Literal["queued", "running", "done", "error"]
    summary: Optional[str] = None
    chunks: Optional[int] = None
    successful_chunks: Optional[int] = None
    duration_ms: Optional[int] = None
    model_name: Optional[str] = None
    cached: Optional[bool] = None


# -------------------------------------------------------
# ✅ Keep the old summarization endpoint working
# -------------------------------------------------------
@router.post("/summarize", response_model=SummarizeOut)
async def summarize_paper(payload: SummarizeIn, db=Depends(get_database)):
    """
    Summarize a paper’s text. Regenerates on click unless use_cache=True.
    Stores results per summary_type under summaries.<type>.
    """
    summary_type = payload.summary_type.lower().strip()
    print(f"[SUMMARY] Received summarization request for {payload.paper_id} ({summary_type})")

    # ✅ Validate ID
    try:
        paper_oid = ObjectId(payload.paper_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid paper ID format")

    # 1️⃣ Check cache
    if payload.use_cache:
        cached_summary = await get_summary(db, payload.paper_id, summary_type)
        if cached_summary:
            print(f"[SUMMARY] Returning cached {summary_type} summary.")
            return SummarizeOut(
                paper_id=payload.paper_id,
                summary_type=summary_type,
                summary=cached_summary.get("text"),
                chunks=cached_summary.get("chunks", 0),
                duration_ms=cached_summary.get("duration_ms", 0),
                cached=True,
            )

    # 2️⃣ Retrieve paper text
    paper = await db.papers.find_one({"_id": paper_oid})
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found in MongoDB")

    full_text = paper.get("text") or paper.get("content") or ""
    if not full_text.strip():
        raise HTTPException(status_code=400, detail="Paper text is empty")

    # 3️⃣ Run summarizer
    res = await summarize_text(text=full_text, summary_type=summary_type)
    if not res or "summary" not in res:
        raise RuntimeError("Summarization returned no result")

    summary_data = {
        "text": res["summary"],
        "chunks": res["chunks"],
        "duration_ms": res["duration_ms"],
        "model": {"id": res.get("model_name", "facebook/bart-base")},
    }

    # 4️⃣ Save result
    await save_summary(db, payload.paper_id, summary_data, summary_type)

    return SummarizeOut(
        paper_id=payload.paper_id,
        summary_type=summary_type,
        summary=summary_data["text"],
        chunks=summary_data["chunks"],
        duration_ms=summary_data["duration_ms"],
        cached=False,
    )


# -------------------------------------------------------
# 🩺 Health check
# -------------------------------------------------------
@router.get("/summarize")
async def summarize_ping():
    return {"message": "Summarization endpoint active"}
