from __future__ import annotations

import asyncio
import logging
from typing import Optional

import certifi
from pymongo import AsyncMongoClient
from pymongo.errors import ServerSelectionTimeoutError

from app.core.config import settings


logger = logging.getLogger(__name__)
_client: Optional[AsyncMongoClient] = None
_db = None


def get_client() -> AsyncMongoClient:
    """Return one Mongo client for the application process."""
    global _client
    if _client is None:
        options = {
            "uuidRepresentation": "standard",
            "serverSelectionTimeoutMS": 5_000,
        }
        if settings.MONGODB_URI.startswith("mongodb+srv://"):
            options["tlsCAFile"] = certifi.where()
        _client = AsyncMongoClient(settings.MONGODB_URI, **options)
    return _client


def get_db():
    global _db
    if _db is None:
        _db = get_client()[settings.MONGODB_DB]
    return _db


async def get_database():
    return get_db()


async def close_client() -> None:
    global _client, _db
    if _client is not None:
        await _client.close()
    _client = None
    _db = None


async def ping(max_retries: int = 3, delay_s: float = 1.0) -> bool:
    client = get_client()
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.admin.command("ping")
            if response.get("ok", 0) == 1:
                return True
        except ServerSelectionTimeoutError:
            logger.warning("MongoDB ping timed out (%s/%s)", attempt, max_retries)
        except Exception:
            logger.exception("MongoDB ping failed (%s/%s)", attempt, max_retries)
        await asyncio.sleep(delay_s)
    return False
