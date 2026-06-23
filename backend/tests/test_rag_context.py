from app.rag.context import build_context


class WordTokenizer:
    def encode(self, text, add_special_tokens=False, **kwargs):
        return text.split()


def chunk(chunk_id, text, start, end, page=1, section="methods", score=0.8):
    return {
        "chunk_id": chunk_id,
        "text": text,
        "score": score,
        "page": page,
        "section": section,
        "start_char": start,
        "end_char": end,
    }


def test_merges_overlapping_chunks_and_preserves_provenance() -> None:
    packet = build_context(
        [
            chunk("a", "alpha beta gamma", 0, 16, score=0.9),
            chunk("b", "gamma delta", 11, 22, score=0.7),
        ],
        WordTokenizer(),
        max_tokens=100,
        max_passages=5,
    )

    assert packet["evidence_available"] is True
    assert len(packet["evidence"]) == 1
    assert packet["evidence"][0]["text"] == "alpha beta gamma delta"
    assert packet["evidence"][0]["source_chunk_ids"] == ["a", "b"]
    assert packet["evidence"][0]["citation"] == "C1"
    assert packet["evidence"][0]["score"] == 0.9


def test_does_not_merge_across_pages() -> None:
    packet = build_context(
        [chunk("a", "shared text", 0, 11, page=1), chunk("b", "shared text", 0, 11, page=2)],
        WordTokenizer(),
        max_tokens=100,
        max_passages=5,
    )

    assert len(packet["evidence"]) == 2


def test_bridge_chunk_merges_all_overlapping_groups() -> None:
    packet = build_context(
        [
            chunk("a", "alpha", 0, 5),
            chunk("b", "gamma", 10, 15),
            chunk("bridge", "a-----g", 4, 11),
        ],
        WordTokenizer(),
        max_tokens=100,
        max_passages=5,
    )

    assert len(packet["evidence"]) == 1
    assert set(packet["evidence"][0]["source_chunk_ids"]) == {"a", "b", "bridge"}


def test_budget_skips_lower_ranked_passage() -> None:
    packet = build_context(
        [chunk("a", "one two", 0, 7), chunk("b", "three four", 20, 30)],
        WordTokenizer(),
        max_tokens=13,
        max_passages=5,
    )

    assert [item["source_chunk_ids"] for item in packet["evidence"]] == [["a"]]
    assert packet["token_count"] <= 13


def test_empty_results_produce_empty_packet() -> None:
    packet = build_context([], WordTokenizer(), max_tokens=100, max_passages=5)

    assert packet == {
        "evidence_available": False,
        "token_count": 0,
        "context_text": "",
        "evidence": [],
    }
