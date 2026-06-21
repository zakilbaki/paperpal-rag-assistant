from fastapi.testclient import TestClient

from app.main import app


def test_root_exposes_api_documentation() -> None:
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"service": "PaperPal API", "docs": "/docs"}
