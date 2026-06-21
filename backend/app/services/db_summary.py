from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional

from bson import ObjectId


async def get_summary(
    db,
    paper_id: str,
    summary_type: str = "medium",
) -> Optional[Dict[str, Any]]:
    projection = {f"summaries.{summary_type}": 1, "summary": 1}
    document = await db.papers.find_one({"_id": ObjectId(paper_id)}, projection)
    if not document:
        return None

    summaries = document.get("summaries", {})
    if summary_type in summaries:
        return summaries[summary_type]
    return document.get("summary")


async def save_summary(
    db,
    paper_id: str,
    summary_data: Dict[str, Any],
    summary_type: str = "medium",
) -> None:
    await db.papers.update_one(
        {"_id": ObjectId(paper_id)},
        {
            "$set": {
                f"summaries.{summary_type}": summary_data,
                "summaries.updated_at": dt.datetime.now(dt.timezone.utc),
            }
        },
    )
