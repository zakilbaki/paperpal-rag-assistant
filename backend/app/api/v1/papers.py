from fastapi import APIRouter, Depends, HTTPException

from app.db.mongo import get_database
from app.models.schemas import KeywordsRequest, KeywordsResponse
from app.services.keywords import KeywordService


router = APIRouter(tags=["papers"])


@router.post("/keywords", response_model=KeywordsResponse)
async def extract_keywords(payload: KeywordsRequest, db=Depends(get_database)):
    """Extract and optionally cache YAKE keywords for one stored paper."""
    try:
        result = await KeywordService(db).extract(
            paper_id=payload.paper_id,
            top_k=payload.top_k,
            use_cache=payload.use_cache,
        )
        return KeywordsResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Keyword extraction failed") from exc
