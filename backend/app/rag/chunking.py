from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class ChunkSpan:
    text: str
    start_char: int
    end_char: int
    section: str | None


def _trimmed_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return None if start == end else (start, end)


def _sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = start
    for boundary in _SENTENCE_BOUNDARY.finditer(text, start, end):
        span = _trimmed_span(text, cursor, boundary.start())
        if span:
            spans.append(span)
        cursor = boundary.end()
    final_span = _trimmed_span(text, cursor, end)
    if final_span:
        spans.append(final_span)
    return spans


def _token_count(tokenizer: Any, text: str) -> int:
    try:
        return len(tokenizer.encode(text, add_special_tokens=False, verbose=False))
    except TypeError:
        return len(tokenizer.encode(text, add_special_tokens=False))


def _long_sentence_chunks(
    text: str,
    start: int,
    end: int,
    tokenizer: Any,
    max_tokens: int,
    overlap: int,
    section: str | None,
) -> list[ChunkSpan]:
    options = {
        "add_special_tokens": False,
        "return_offsets_mapping": True,
        "truncation": False,
    }
    try:
        encoded = tokenizer(text[start:end], verbose=False, **options)
    except TypeError:
        encoded = tokenizer(text[start:end], **options)
    offsets = encoded["offset_mapping"]
    step = max_tokens - overlap
    chunks: list[ChunkSpan] = []
    for token_start in range(0, len(offsets), step):
        token_end = min(token_start + max_tokens, len(offsets))
        char_start = start + offsets[token_start][0]
        char_end = start + offsets[token_end - 1][1]
        chunks.append(ChunkSpan(text[char_start:char_end], char_start, char_end, section))
        if token_end == len(offsets):
            break
    return chunks


def chunk_page(
    text: str,
    tokenizer: Any,
    max_tokens: int,
    overlap: int,
) -> tuple[list[ChunkSpan], str | None]:
    """Create sentence-aware, token-bounded chunks with exact page-local offsets.

    The current baseline does not infer sections from headings because that heuristic
    proved too fragile across scientific domains.
    """
    if max_tokens <= 0 or overlap < 0 or overlap >= max_tokens:
        raise ValueError("Chunk limits require max_tokens > overlap >= 0")

    chunks: list[ChunkSpan] = []
    current: list[tuple[int, int, int]] = []
    current_tokens = 0
    section: str | None = None

    for start, end in _sentence_spans(text, 0, len(text)):
        token_count = _token_count(tokenizer, text[start:end])
        if token_count > max_tokens:
            if current:
                chunks.append(ChunkSpan(text[current[0][0]:current[-1][1]], current[0][0], current[-1][1], section))
                current = []
                current_tokens = 0
            chunks.extend(
                _long_sentence_chunks(text, start, end, tokenizer, max_tokens, overlap, section)
            )
            continue

        if current and current_tokens + token_count > max_tokens:
            chunks.append(ChunkSpan(text[current[0][0]:current[-1][1]], current[0][0], current[-1][1], section))
            tail: list[tuple[int, int, int]] = []
            tail_tokens = 0
            if overlap > 0:
                for item in reversed(current):
                    if tail and tail_tokens + item[2] > overlap:
                        break
                    tail.append(item)
                    tail_tokens += item[2]
            current = list(reversed(tail))
            current_tokens = tail_tokens
            while current and current_tokens + token_count > max_tokens:
                current_tokens -= current.pop(0)[2]

        current.append((start, end, token_count))
        current_tokens += token_count

    if current:
        chunks.append(ChunkSpan(text[current[0][0]:current[-1][1]], current[0][0], current[-1][1], section))
    return chunks, None
