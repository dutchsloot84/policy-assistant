"""Retrieval utilities using FAISS vector store."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List

from dotenv import load_dotenv

from ..core.redact import redact_text
from ..store.faiss_store import FaissVectorStore, Metadata

load_dotenv()

DEFAULT_TOP_K = int(os.getenv("TOP_K", "3"))


@dataclass
class RetrievedChunk:
    score: float
    chunk_id: str
    text: str
    source: str
    metadata: Metadata


class Retriever:
    def __init__(self, store: FaissVectorStore) -> None:
        self.store = store

    def search(
        self,
        query_embedding: Iterable[float],
        *,
        top_k: int | None = None,
        redact: bool | None = None,
    ) -> List[RetrievedChunk]:
        k = top_k or DEFAULT_TOP_K
        results = self.store.search(query_embedding, k)
        chunks: List[RetrievedChunk] = []
        for score, meta in results:
            text = redact_text(meta.text, enabled=redact)
            chunks.append(
                RetrievedChunk(
                    score=score,
                    chunk_id=meta.chunk_id,
                    text=text,
                    source=meta.source,
                    metadata=meta,
                )
            )
        return chunks

    def build_context(self, chunks: Iterable[RetrievedChunk]) -> List[str]:
        context_blocks = []
        for chunk in chunks:
            context_blocks.append(
                "Source: "
                + f"{chunk.source} | Chunk: {chunk.chunk_id}"
                + f"\nScore: {chunk.score:.4f}\n{chunk.text}"
            )
        return context_blocks
