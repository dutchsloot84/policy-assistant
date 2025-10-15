from __future__ import annotations

import json

from src.historian.schema import Marker, QueryEvent, RetrievalHit


def test_query_event_serialization() -> None:
    hits = [
        RetrievalHit(source="doc1.pdf", chunk_id="111aaa", score=0.75, preview="alpha"),
        RetrievalHit(source="doc2.pdf", chunk_id="222bbb", score=0.55, preview="beta"),
    ]
    event = QueryEvent(
        query="What is the deductible?",
        top_k=len(hits),
        hits=hits,
        model="gpt-test",
        max_tokens=256,
        temperature=0.3,
        latency_ms=987,
        answer_chars=1234,
        est_input_tokens=1500,
        est_output_tokens=250,
        est_usd=0.12,
        markers=[Marker(type="Decision", text="Policy response")],
    )

    payload = json.loads(event.model_dump_json())
    assert payload["kind"] == "query"
    assert isinstance(payload["hits"], list)
    assert len(payload["hits"]) == 2
    for hit in payload["hits"]:
        assert set(hit) >= {"source", "chunk_id", "score", "preview"}
        assert isinstance(hit["score"], float)
    assert payload["markers"][0]["type"] == "Decision"
    assert payload["latency_ms"] == 987
    assert payload["est_input_tokens"] == 1500
    assert payload["est_usd"] == 0.12
