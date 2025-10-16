import importlib

import pytest
from fastapi.testclient import TestClient

from src.core.cost_guard import estimate_tokens


def _make_pdf_bytes(text: str) -> bytes:
    header = b"%PDF-1.4\n"
    objects = [
        b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n",
        b"2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n",
        (
            b"3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>endobj\n"
        ),
    ]
    stream = f"BT /F1 12 Tf 72 72 Td ({text}) Tj ET".encode("utf-8")
    objects.append(
        b"4 0 obj<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>stream\n"
        + stream
        + b"\nendstream\nendobj\n"
    )
    objects.append(b"5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n")

    body = b""
    offsets = []
    current = len(header)
    for obj in objects:
        offsets.append(current)
        body += obj
        current += len(obj)

    xref = b"xref\n0 " + str(len(objects) + 1).encode("ascii") + b"\n"
    xref += b"0000000000 65535 f \n"
    for offset in offsets:
        xref += f"{offset:010d} 00000 n \n".encode("ascii")
    trailer = b"trailer<< /Root 1 0 R /Size " + str(len(objects) + 1).encode("ascii") + b" >>\n"
    startxref = len(header) + len(body)
    eof = b"startxref\n" + str(startxref).encode("ascii") + b"\n%%EOF"
    return header + body + xref + trailer + eof


@pytest.fixture
def ingest_client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REDACT_PII", "false")
    monkeypatch.setenv("EMBED_CACHE_PATH", str(tmp_path / "emb.pkl"))
    monkeypatch.setenv("MAX_TOKENS", "300")
    monkeypatch.setenv("MAX_EMBED_TOKENS", "300")
    monkeypatch.setenv("HIST_LEDGER", str(tmp_path / "ledger.jsonl"))

    from src.api import app as app_module

    importlib.reload(app_module)

    app_module.vector_store = app_module.FaissVectorStore(
        index_path=tmp_path / "index.faiss",
        meta_path=tmp_path / "meta.pkl",
    )
    app_module.retriever = app_module.Retriever(app_module.vector_store)

    def fake_embed(texts):
        return [[float(i + 1)] for i, _text in enumerate(texts)]

    app_module.embedding_service.client.embed_texts = fake_embed
    return TestClient(app_module.app)


def test_ingest_allows_large_batches_with_small_chunks(ingest_client):
    sentence = "This policy statement ensures compliance with regulatory requirements."
    text = " ".join([sentence] * 40)
    pdf_bytes = _make_pdf_bytes(text)

    response = ingest_client.post(
        "/ingest",
        files={"file": ("policy.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["chunks"] >= 2


def test_ingest_many_small_chunks_exceeding_chat_budget(monkeypatch, ingest_client):
    from src.api import app as app_module
    from src.core import chunk as chunk_module

    def small_chunker(text: str):
        return chunk_module.chunk_text(text, max_chars=80, overlap=0)

    monkeypatch.setattr(app_module, "chunk_text", small_chunker)

    sentence = "Policy reminder: follow procedure A before proceeding to step B."
    text = " ".join([sentence] * 60)
    pdf_bytes = _make_pdf_bytes(text)

    chunks = small_chunker(text)
    assert len(chunks) > 40  # plenty of small chunks
    assert max(len(chunk.text) for chunk in chunks) <= 80
    assert sum(estimate_tokens(chunk.text) for chunk in chunks) > 300

    response = ingest_client.post(
        "/ingest",
        files={"file": ("policy.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["chunks"] == len(chunks)
