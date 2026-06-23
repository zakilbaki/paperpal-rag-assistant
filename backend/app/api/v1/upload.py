from __future__ import annotations

import logging
import time
from pathlib import Path

import fitz
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.db.mongo import get_database


logger = logging.getLogger(__name__)
router = APIRouter(tags=["papers"])
MAX_UPLOAD_BYTES = 3 * 1024 * 1024


def extract_pdf_pages(contents: bytes) -> tuple[str, list[dict]]:
    """Extract text while retaining page numbers and global character offsets."""
    pages: list[dict] = []
    parts: list[str] = []
    cursor = 0

    with fitz.open(stream=contents, filetype="pdf") as document:
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text("text")
            if not page_text.strip():
                continue
            if parts:
                parts.append("\n")
                cursor += 1
            start_char = cursor
            parts.append(page_text)
            cursor += len(page_text)
            pages.append(
                {
                    "page": page_number,
                    "text": page_text,
                    "start_char": start_char,
                    "end_char": cursor,
                }
            )
    return "".join(parts), pages


@router.post("/upload")
async def upload_paper(file: UploadFile = File(...), db=Depends(get_database)):
    """Extract and persist text from a PDF no larger than 3 MB."""
    filename = file.filename or "document.pdf"
    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=415, detail="Only PDF files are supported")

    started_at = time.perf_counter()
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large; maximum size is 3 MB")

    try:
        text, pages = extract_pdf_pages(contents)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid or unreadable PDF") from exc

    if not text.strip():
        raise HTTPException(status_code=400, detail="The PDF contains no extractable text")

    try:
        result = await db.papers.insert_one(
            {
                "filename": filename,
                "text": text,
                "pages": pages,
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
