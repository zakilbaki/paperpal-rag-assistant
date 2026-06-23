import pytest

from app.rag.reranking import keyword_score, rerank_chunks


def candidate(chunk_id: str, text: str, score: float) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "score": score,
        "page": 1,
        "section": None,
        "start_char": 0,
        "end_char": len(text),
    }


def test_keyword_score_rewards_meaningful_question_terms() -> None:
    score = keyword_score(
        "What method was used?",
        "The method used transformer attention layers.",
    )

    assert score > keyword_score("What method was used?", "References and bibliography.")


def test_rerank_chunks_can_promote_direct_answer_over_higher_vector_score() -> None:
    results = rerank_chunks(
        "What method was used?",
        [
            candidate("references", "References and bibliography.", 0.90),
            candidate("method", "The method used transformer attention layers.", 0.70),
        ],
        limit=2,
    )

    assert [item["chunk_id"] for item in results] == ["method", "references"]
    assert results[0]["vector_score"] == 0.70
    assert results[0]["keyword_score"] > results[1]["keyword_score"]
    assert results[0]["vector_rank"] == 2
    assert results[0]["score"] == pytest.approx(0.82)


def test_rerank_chunks_limits_final_results() -> None:
    results = rerank_chunks(
        "What is the abstract?",
        [
            candidate("a", "Abstract\nThis paper proposes attention.", 0.60),
            candidate("b", "Unrelated appendix.", 0.59),
            candidate("c", "Unrelated references.", 0.58),
        ],
        limit=1,
    )

    assert [item["chunk_id"] for item in results] == ["a"]
