from app.services.pdf_parser import segment_sections


def test_segment_sections_recognizes_scientific_headers() -> None:
    sections = segment_sections(
        "A useful paper title\n\nAbstract\nA concise abstract.\n\nMethods\nA clear method."
    )

    assert [section["name"] for section in sections] == ["title", "abstract", "methods"]
    assert sections[1]["text"] == "A concise abstract."


def test_segment_sections_falls_back_to_body() -> None:
    sections = segment_sections("A title\nThis document has no explicit section headers.")

    assert sections[0]["name"] == "title"
    assert sections[-1]["name"] == "body"
