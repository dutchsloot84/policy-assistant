import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def structured_client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REDACT_PII", "false")
    monkeypatch.setenv("HIST_LEDGER", str(tmp_path / "ledger.jsonl"))

    from src.api import app as app_module

    importlib.reload(app_module)

    app_module.vector_store = app_module.FaissVectorStore(
        index_path=tmp_path / "index.faiss",
        meta_path=tmp_path / "meta.pkl",
    )
    app_module.retriever = app_module.Retriever(app_module.vector_store)

    captured_queries = []

    def fake_embed_query(q: str):
        captured_queries.append(q)
        return [1.0, 0.0]

    app_module.embedding_service.embed_query = fake_embed_query
    app_module.embedding_service.embed_documents = lambda texts: [[1.0, 0.0] for _ in texts]

    chat_called = {"value": False}

    def fake_chat(*args, **kwargs):  # pragma: no cover - ensure not used
        chat_called["value"] = True
        raise AssertionError("Chat should not be invoked for structured shortcut")

    app_module.openai_client.chat = fake_chat

    app_module.vector_store.add(
        embeddings=[[1.0, 0.0]],
        metadatas=[
            app_module.Metadata(
                document_id="policy.pdf",
                chunk_id="chunk-123",
                text="Estimated Total Premium\n$ 299,997.00",
                source="policy.pdf",
                page_start=2,
                page_end=2,
                fields={"estimated_total_premium": "$ 299,997.00"},
            )
        ],
    )

    client = TestClient(app_module.app)
    return client, captured_queries, chat_called


def test_structured_shortcut_returns_field(structured_client):
    client, captured_queries, chat_called = structured_client

    response = client.post(
        "/query",
        json={"query": "What is the estimated total premium?", "top_k": 1, "redact": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "$ 299,997.00" in payload["answer"]
    assert "policy.pdf" in payload["answer"]
    assert "Page 2" in payload["answer"]
    assert payload["sources"][0]["chunk_id"] == "chunk-123"
    assert payload["sources"][0]["page_start"] == 2
    assert not chat_called["value"]
    assert captured_queries[0].startswith("What is the estimated total premium?")
    assert "premium overall" in captured_queries[0].lower()
