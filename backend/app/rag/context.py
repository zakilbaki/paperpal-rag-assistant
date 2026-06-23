from __future__ import annotations

from typing import Any


def _tokens(tokenizer, text: str) -> list[int]:
    try:
        return tokenizer.encode(text, add_special_tokens=False, verbose=False)
    except TypeError:
        return tokenizer.encode(text, add_special_tokens=False)


def _can_merge(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("page") == right.get("page")
        and left.get("section") == right.get("section")
        and left["start_char"] < right["end_char"]
        and right["start_char"] < left["end_char"]
    )


def _merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    start = min(left["start_char"], right["start_char"])
    end = max(left["end_char"], right["end_char"])
    chars = [""] * (end - start)
    preferred, other = sorted((left, right), key=lambda item: item["rank"])
    for item in (other, preferred):
        offset = item["start_char"] - start
        chars[offset : offset + len(item["text"])] = item["text"]
    source_ids = list(dict.fromkeys(preferred["source_chunk_ids"] + other["source_chunk_ids"]))
    return {
        **preferred,
        "text": "".join(chars),
        "score": max(left["score"], right["score"]),
        "start_char": start,
        "end_char": end,
        "source_chunk_ids": source_ids,
        "rank": preferred["rank"],
    }


def _block(citation: str, evidence: dict[str, Any]) -> str:
    metadata = [citation]
    if evidence.get("page") is not None:
        metadata.append(f"page {evidence['page']}")
    if evidence.get("section"):
        metadata.append(f"section {evidence['section']}")
    metadata.append(f"score {evidence['score']:.3f}")
    return f"[{' | '.join(metadata)}]\n{evidence['text']}"


def build_context(
    results: list[dict[str, Any]],
    tokenizer,
    max_tokens: int,
    max_passages: int,
) -> dict[str, Any]:
    merged: list[dict[str, Any]] = []
    for rank, result in enumerate(results):
        candidate = {**result, "source_chunk_ids": [result["chunk_id"]], "rank": rank}
        remaining = []
        for item in merged:
            if _can_merge(item, candidate):
                candidate = _merge(candidate, item)
            else:
                remaining.append(item)
        merged = remaining + [candidate]

    selected = []
    blocks = []
    for item in sorted(merged, key=lambda value: value["rank"]):
        if len(selected) >= max_passages:
            break
        citation = f"C{len(selected) + 1}"
        block = _block(citation, item)
        proposed = "\n\n".join(blocks + [block])
        if len(_tokens(tokenizer, proposed)) > max_tokens:
            continue
        selected.append(
            {
                "citation": citation,
                "source_chunk_ids": item["source_chunk_ids"],
                "text": item["text"],
                "score": item["score"],
                "page": item.get("page"),
                "section": item.get("section"),
                "start_char": item["start_char"],
                "end_char": item["end_char"],
            }
        )
        blocks.append(block)

    context_text = "\n\n".join(blocks)
    return {
        "evidence_available": bool(selected),
        "token_count": len(_tokens(tokenizer, context_text)),
        "context_text": context_text,
        "evidence": selected,
    }
