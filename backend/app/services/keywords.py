from __future__ import annotations
from typing import Dict, Any, Optional
import time
from bson import ObjectId
from yake import KeywordExtractor


class KeywordService:
    """YAKE keyword extraction with Mongo caching and dynamic top_k."""

    def __init__(self, db):
        # db should already be a Database instance (not a client)
        self.db = db

    async def extract(
        self,
        paper_id: str,
        top_k: int = 15,
        use_cache: bool = False
    ) -> Dict[str, Any]:
        """
        Extract keywords using YAKE.
        - Respects top_k.
        - Uses cache only if explicitly requested.
        """
        start = time.time()
        coll = self.db["papers"]

        # 1️⃣ Optional cache usage
        doc = await coll.find_one({"_id": ObjectId(paper_id)}, {"keywords": 1})
        if use_cache and doc and "keywords" in doc and doc["keywords"]:
            print("[KEYWORDS] Returning cached keywords.")
            return {
                "paper_id": paper_id,
                "keywords": doc["keywords"],
                "cached": True,
                "duration_ms": int((time.time() - start) * 1000),
            }

        # 2️⃣ Load text from Mongo
        doc = await coll.find_one({"_id": ObjectId(paper_id)})
        if not doc:
            raise ValueError("Paper not found")

        source_text: Optional[str] = None
        for key in ("full_text", "text", "content_text"):
            if isinstance(doc.get(key), str) and doc[key].strip():
                source_text = doc[key]
                break

        # fallback to summary
        if not source_text:
            summary = (doc.get("summary") or {}).get("text")
            if isinstance(summary, str) and summary.strip():
                source_text = summary

        if not source_text:
            raise ValueError("No text available for keyword extraction")

        # 3️⃣ Extract fresh keywords
        print(f"[KEYWORDS] Extracting {top_k} keywords with YAKE...")
        extractor = KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=top_k)
        keywords = [
            {"text": k, "score": float(s)}
            for k, s in extractor.extract_keywords(source_text)
        ]

        # 4️⃣ Save new results
        await coll.update_one(
            {"_id": ObjectId(paper_id)},
            {"$set": {"keywords": keywords}},
            upsert=False,
        )
        print(f"[KEYWORDS] ✅ Saved {len(keywords)} keywords to MongoDB.")

        return {
            "paper_id": paper_id,
            "keywords": keywords,
            "cached": False,
            "duration_ms": int((time.time() - start) * 1000),
        }
