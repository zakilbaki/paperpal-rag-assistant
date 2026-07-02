from __future__ import annotations

import re
from typing import Any


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9-]*")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "paper",
    "study",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
_VECTOR_WEIGHT = 0.60
_KEYWORD_WEIGHT = 0.40


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _query_terms(question: str) -> list[str]:
    return [token for token in _tokens(question) if token not in _STOP_WORDS]


def keyword_score(question: str, text: str) -> float:
    """Score how directly a candidate passage matches the question words.

    This is a lightweight reranking signal, not a truth score. It rewards candidate
    chunks that mention more of the meaningful question terms and gives a small bonus
    when those terms appear as a phrase.
    """
    terms = _query_terms(question)
    if not terms:
        return 0.0

    unique_terms = set(terms)
    text_tokens = _tokens(text)
    if not text_tokens:
        return 0.0

    text_token_set = set(text_tokens)
    coverage = len(unique_terms & text_token_set) / len(unique_terms)
    phrase = " ".join(terms)
    phrase_bonus = 1.0 if phrase and phrase in " ".join(text_tokens) else 0.0
    return min(1.0, (0.85 * coverage) + (0.15 * phrase_bonus))


def _clamp_score(score: float) -> float:
    return max(0.0, min(1.0, score))


def rerank_chunks(question: str, candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Rerank vector-search candidates and return the strongest passages.

    The current reranker deliberately avoids a second ML model. It combines the Chroma
    semantic score with a transparent lexical signal so broad candidate retrieval can
    rescue chunks that vector search placed slightly too low.
    """
    scored = []
    for vector_rank, candidate in enumerate(candidates):
        vector_score = float(candidate["score"])
        lexical_score = keyword_score(question, candidate["text"])
        rerank_score = (_VECTOR_WEIGHT * _clamp_score(vector_score)) + (
            _KEYWORD_WEIGHT * lexical_score
        )
        scored.append(
            {
                **candidate,
                "score": rerank_score,
                "vector_score": vector_score,
                "keyword_score": lexical_score,
                "vector_rank": vector_rank + 1,
            }
        )

    return sorted(
        scored,
        key=lambda item: (-item["score"], item["vector_rank"], item["chunk_id"]),
    )[:limit]
