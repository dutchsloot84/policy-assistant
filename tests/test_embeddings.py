"""Tests for embedding service safeguards."""

from __future__ import annotations

from src.core.embeddings import EmbeddingService


class DummyClient:
    def embed_texts(self, texts):  # type: ignore[override]
        return [[float(idx), float(idx)] for idx, _ in enumerate(texts)]


def test_embedding_batches_allow_large_total_under_per_chunk_limit(tmp_path, monkeypatch):
    """Ensure batching is not blocked by aggregate token estimation."""

    monkeypatch.setenv("MAX_TOKENS", "300")
    monkeypatch.delenv("MAX_EMBED_TOKENS", raising=False)

    cache_file = tmp_path / "emb_cache.pkl"
    service = EmbeddingService(client=DummyClient(), cache_file=cache_file)

    # Ten chunks of ~200 characters each (≈50 tokens) would previously exceed the
    # shared 300-token budget when concatenated, triggering a ValueError.
    chunks = [f"chunk-{i}-" + ("a" * 200) for i in range(10)]

    vectors = service.embed_documents(chunks)

    assert len(vectors) == len(chunks)
