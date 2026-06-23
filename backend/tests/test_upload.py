from types import SimpleNamespace

import fitz
from bson import ObjectId
from fastapi.testclient import TestClient

from app.db.mongo import get_database
from app.api.v1.upload import MAX_UPLOAD_BYTES
from app.main import app


class FakePapersCollection:
    def __init__(self) -> None:
        self.inserted_document = None

    async def insert_one(self, document):
        self.inserted_document = document
        return SimpleNamespace(inserted_id=ObjectId())


class FakeDatabase:
    def __init__(self) -> None:
        self.papers = FakePapersCollection()


def create_pdf() -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "A local PaperPal integration test document.")
    contents = document.tobytes()
    document.close()
    return contents


def test_upload_extracts_and_persists_pdf_text() -> None:
    database = FakeDatabase()
    app.dependency_overrides[get_database] = lambda: database

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/papers/upload",
                files={"file": ("paper.pdf", create_pdf(), "application/pdf")},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "PaperPal integration test" in database.papers.inserted_document["text"]
    assert database.papers.inserted_document["pages"][0]["page"] == 1
    assert database.papers.inserted_document["pages"][0]["start_char"] == 0
    assert database.papers.inserted_document["pages"][0]["end_char"] > 0


def test_upload_rejects_pdf_larger_than_three_megabytes() -> None:
    database = FakeDatabase()
    app.dependency_overrides[get_database] = lambda: database

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/papers/upload",
                files={"file": ("large.pdf", b"0" * (MAX_UPLOAD_BYTES + 1), "application/pdf")},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 413
    assert response.json()["detail"] == "File too large; maximum size is 3 MB"
    assert database.papers.inserted_document is None
