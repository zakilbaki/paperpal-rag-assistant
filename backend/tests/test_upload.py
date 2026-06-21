from types import SimpleNamespace

import fitz
from bson import ObjectId
from fastapi.testclient import TestClient

from app.db.mongo import get_database
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
