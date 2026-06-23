from fastapi.testclient import TestClient

from app.main import app


def test_root_exposes_api_documentation() -> None:
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"service": "PaperPal API", "docs": "/docs"}


def test_openapi_exposes_rag_index_and_retrieval_routes() -> None:
    with TestClient(app) as client:
        paths = client.get("/openapi.json").json()["paths"]

    assert "/api/v1/papers/{paper_id}/rag/index" in paths
    assert "/api/v1/papers/{paper_id}/rag/retrieve" in paths
