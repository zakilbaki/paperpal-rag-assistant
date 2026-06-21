from __future__ import annotations

import logging
import time
from pathlib import Path

import fitz
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.db.mongo import get_database


logger = logging.getLogger(__name__)
router = APIRouter(tags=["papers"])
MAX_UPLOAD_BYTES = 1 * 1024 * 1024


@router.post("/upload")
async def upload_paper(file: UploadFile = File(...), db=Depends(get_database)):
    """Extract and persist text from a PDF no larger than 1 MB."""
    filename = file.filename or "document.pdf"
    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=415, detail="Only PDF files are supported")

    started_at = time.perf_counter()
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large; maximum size is 1 MB")

    try:
        with fitz.open(stream=contents, filetype="pdf") as document:
            text = "\n".join(page.get_text("text") for page in document)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid or unreadable PDF") from exc

    if not text.strip():
        raise HTTPException(status_code=400, detail="The PDF contains no extractable text")

    try:
        result = await db.papers.insert_one(
            {
                "filename": filename,
                "text": text,
                "created_at": time.time(),
                "size_bytes": len(contents),
            }
        )
    except Exception as exc:
        logger.exception("Could not persist uploaded paper")
        raise HTTPException(status_code=503, detail="Document storage is unavailable") from exc

    return {
        "status": "success",
        "paper_id": str(result.inserted_id),
        "duration_ms": int((time.perf_counter() - started_at) * 1_000),
    }
