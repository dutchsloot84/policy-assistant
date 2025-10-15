"""Helpers for reading and summarizing the historian ledger."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List


def load_ledger(path: str | Path) -> Iterable[dict]:
    """Yield parsed JSON objects from a ledger file."""
    ledger_path = Path(path)
    if not ledger_path.exists():
        return []

    def _generator() -> Iterator[dict]:
        with ledger_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    return _generator()


def summarize(path: str | Path) -> Dict[str, object]:
    """Summarize the ledger contents for quick inspection."""
    ingest_events = 0
    query_events = 0
    files = set()
    sample_queries: List[dict] = []

    for entry in load_ledger(path):
        kind = entry.get("kind")
        if kind == "ingest":
            ingest_events += 1
            filename = entry.get("filename")
            if isinstance(filename, str):
                files.add(filename)
        elif kind == "query":
            query_events += 1
            if len(sample_queries) < 10:
                sample_queries.append(
                    {
                        "query": entry.get("query"),
                        "ts": entry.get("ts"),
                        "top_k": entry.get("top_k"),
                        "hits": len(entry.get("hits") or []),
                    }
                )

    return {
        "ingest_events": ingest_events,
        "files": sorted(files),
        "query_events": query_events,
        "sample_queries": sample_queries,
    }


__all__ = ["load_ledger", "summarize"]
