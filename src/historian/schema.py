"""Schema definitions for the historian ledger."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

_PHX_ZONE = ZoneInfo("America/Phoenix")


def run_id() -> str:
    """Return a new opaque run identifier."""
    return uuid4().hex


def tz_now() -> str:
    """Return the current timestamp in the Phoenix timezone."""
    return datetime.now(_PHX_ZONE).isoformat()


class Marker(BaseModel):
    """Marker attached to an event for additional context."""

    type: str
    text: str


class RetrievalHit(BaseModel):
    """Details about a retrieval hit from the vector store."""

    source: str
    chunk_id: str
    score: float
    preview: str
    page_start: int | None = None
    page_end: int | None = None


class IngestEvent(BaseModel):
    """Event recorded after a successful ingest."""

    kind: str = Field(default="ingest", frozen=True)
    ts: str = Field(default_factory=tz_now)
    run: str = Field(default_factory=run_id)
    filename: str
    chunks: int
    embed_batches: int
    duration_ms: int
    markers: list[Marker] = Field(default_factory=list)


class QueryEvent(BaseModel):
    """Event recorded after a successful query."""

    kind: str = Field(default="query", frozen=True)
    ts: str = Field(default_factory=tz_now)
    run: str = Field(default_factory=run_id)
    query: str
    top_k: int
    hits: list[RetrievalHit]
    model: str
    max_tokens: int
    temperature: float
    latency_ms: int
    answer_chars: int
    est_input_tokens: Optional[int] = None
    est_output_tokens: Optional[int] = None
    est_usd: Optional[float] = None
    markers: list[Marker] = Field(default_factory=list)


def _ledger_path_from_env() -> Path:
    value = os.getenv("HIST_LEDGER", "data/historian/ledger.jsonl")
    return Path(value)


def _rotate_mb_from_env() -> int:
    value = os.getenv("HIST_ROTATE_MB", "10")
    try:
        return int(value)
    except ValueError:  # pragma: no cover - defensive guard
        return 10


class LedgerConfig(BaseModel):
    """Runtime configuration for the ledger."""

    path: Path = Field(default_factory=_ledger_path_from_env)
    rotate_mb: int = Field(default_factory=_rotate_mb_from_env)


__all__ = [
    "IngestEvent",
    "LedgerConfig",
    "Marker",
    "QueryEvent",
    "RetrievalHit",
    "run_id",
    "tz_now",
]
