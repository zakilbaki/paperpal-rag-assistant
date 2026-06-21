from typing import Any

from fastapi import APIRouter

from app.core.config import settings
from app.db.mongo import ping


router = APIRouter()


@router.get("/", summary="Health check")
async def health_check() -> dict[str, Any]:
    return {
        "service": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "mongo_ok": await ping(),
    }
