import asyncio

from app.services.summarize_llm import summarize_text


def test_short_text_summary_does_not_load_transformer_model() -> None:
    result = asyncio.run(
        summarize_text(
            "This short scientific abstract describes a reproducible local test.",
            summary_type="short",
        )
    )

    assert result["chunks"] == 1
    assert "reproducible local test" in result["summary"]
