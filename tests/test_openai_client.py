"""Tests for the thin OpenAI client wrapper."""

from __future__ import annotations

import pytest
from openai import OpenAIError

from src.llm.openai_client import OpenAIClient


class DummyOpenAIError(OpenAIError):
    """Concrete error type for testing."""


def test_embed_texts_retries_and_reraises(monkeypatch):
    """Non-API errors should be retried and ultimately re-raised."""

    error = DummyOpenAIError("transient failure")
    call_count = 0

    class DummyEmbeddings:
        def create(self, *args, **kwargs):  # pragma: no cover - exercised via client
            nonlocal call_count
            call_count += 1
            raise error

    class DummyClient:
        def __init__(self, *_, **__):
            self.embeddings = DummyEmbeddings()

    monkeypatch.setattr("src.llm.openai_client.OpenAI", DummyClient)
    monkeypatch.setattr("src.llm.openai_client.time.sleep", lambda _timeout: None)

    client = OpenAIClient(api_key="test")

    with pytest.raises(DummyOpenAIError) as excinfo:
        client.embed_texts(["hello"])

    assert call_count == 5
    assert "transient failure" in str(excinfo.value)
