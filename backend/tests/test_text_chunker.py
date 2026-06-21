from app.services.text_chunker import chunk_by_tokens, split_sentences


class WordTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str], skip_special_tokens: bool = True) -> str:
        return " ".join(tokens)


def test_split_sentences_normalizes_whitespace() -> None:
    assert split_sentences("First sentence.\n\nSecond sentence!") == [
        "First sentence.",
        "Second sentence!",
    ]


def test_chunk_by_tokens_respects_maximum_for_long_sentence() -> None:
    chunks = chunk_by_tokens(
        "one two three four five six seven eight.",
        WordTokenizer(),
        max_tokens=3,
        overlap=0,
    )

    assert chunks == ["one two three", "four five six", "seven eight."]
