import re

from app.rag.chunking import chunk_page


class WordTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = True,
        truncation: bool = False,
    ) -> dict:
        return {"offset_mapping": [match.span() for match in re.finditer(r"\S+", text)]}


def test_chunk_page_preserves_overlap_and_offsets() -> None:
    text = "One two three. Four five six. Seven eight nine."

    chunks, final_section = chunk_page(
        text,
        WordTokenizer(),
        max_tokens=6,
        overlap=3,
    )

    assert [chunk.text for chunk in chunks] == [
        "One two three. Four five six.",
        "Four five six. Seven eight nine.",
    ]
    assert all(chunk.section is None for chunk in chunks)
    assert all(text[chunk.start_char:chunk.end_char] == chunk.text for chunk in chunks)
    assert final_section is None


def test_chunk_page_splits_oversized_sentence_with_token_overlap() -> None:
    text = "one two three four five six seven"

    chunks, _ = chunk_page(text, WordTokenizer(), max_tokens=4, overlap=1)

    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "four five six seven",
    ]


def test_chunk_page_allows_zero_overlap() -> None:
    text = "One two three. Four five six. Seven eight nine."

    chunks, _ = chunk_page(text, WordTokenizer(), max_tokens=6, overlap=0)

    assert [chunk.text for chunk in chunks] == [
        "One two three. Four five six.",
        "Seven eight nine.",
    ]


def test_chunk_page_does_not_infer_sections_from_headings() -> None:
    text = "Methodology\nWe sampled the data.\n\nFindings\nAccuracy improved."

    chunks, final_section = chunk_page(text, WordTokenizer(), max_tokens=20, overlap=0)

    assert [chunk.section for chunk in chunks] == [None]
    assert "Methodology" in chunks[0].text
    assert "Findings" in chunks[0].text
    assert final_section is None
