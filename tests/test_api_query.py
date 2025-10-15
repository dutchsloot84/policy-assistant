import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REDACT_PII", "false")
    from src.api import app as app_module

    importlib.reload(app_module)

    # Patch embeddings to deterministic vectors
    app_module.embedding_service.embed_query = lambda _: [1.0, 0.0]
    app_module.embedding_service.embed_documents = lambda texts: [[1.0, 0.0] for _ in texts]

    # Patch OpenAI chat to deterministic answer
    app_module.openai_client.chat = lambda query, context_blocks: "Answer with citations"

    # Pre-populate store with a single chunk
    app_module.vector_store.add(
        embeddings=[[1.0, 0.0]],
        metadatas=[
            app_module.Metadata(
                document_id="doc.pdf",
                chunk_id="chunk-1",
                text="Policy content",
                source="doc.pdf",
            )
        ],
    )
    return TestClient(app_module.app)


def test_query_endpoint(client):
    response = client.post(
        "/query", json={"query": "What is the policy?", "top_k": 1, "redact": False}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Answer with citations"
    assert payload["sources"][0]["chunk_id"] == "chunk-1"
