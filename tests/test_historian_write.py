from __future__ import annotations

import json

from src.historian import Ledger
from src.historian.export import load_ledger, summarize
from src.historian.schema import IngestEvent, LedgerConfig, Marker, QueryEvent, RetrievalHit


def test_ledger_appends(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    cfg = LedgerConfig(path=path, rotate_mb=100)
    ledger = Ledger(cfg)

    ingest = IngestEvent(
        filename="example.pdf",
        chunks=4,
        embed_batches=1,
        duration_ms=123,
        markers=[Marker(type="Note", text="test")],
    )
    hit = RetrievalHit(source="example.pdf", chunk_id="abc123", score=0.9, preview="text")
    query = QueryEvent(
        query="What is coverage?",
        top_k=1,
        hits=[hit],
        model="gpt-test",
        max_tokens=300,
        temperature=0.2,
        latency_ms=456,
        answer_chars=12,
    )

    ledger.append(ingest.model_dump())
    ledger.append(query.model_dump())

    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        data = json.loads(line)
        assert "kind" in data
        assert data["kind"] in {"ingest", "query"}

    events = list(load_ledger(path))
    assert len(events) == 2
    summary = summarize(path)
    assert summary["ingest_events"] == 1
    assert summary["query_events"] == 1
    assert summary["files"] == ["example.pdf"]
    assert summary["sample_queries"][0]["query"] == "What is coverage?"


def test_ledger_rotation(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    cfg = LedgerConfig(path=path, rotate_mb=1)
    ledger = Ledger(cfg)

    large_payload = "x" * (2 * 1024 * 1024)
    first = {
        "kind": "ingest",
        "ts": "2025-01-01T00:00:00",
        "run": "r1",
        "filename": "big.pdf",
        "chunks": 1,
        "embed_batches": 1,
        "duration_ms": 1,
        "markers": [],
        "blob": large_payload,
    }
    ledger.append(first)
    second = IngestEvent(
        filename="small.pdf",
        chunks=1,
        embed_batches=1,
        duration_ms=2,
    ).model_dump()
    ledger.append(second)

    rotated = path.with_name("ledger.r1.jsonl")
    assert rotated.exists()
    assert path.exists()
    assert len(rotated.read_text(encoding="utf-8").strip().splitlines()) == 1
