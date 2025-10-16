"""FastAPI application exposing ingestion and query endpoints."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..core.chunk import chunk_text, format_page_label, map_offsets_to_page_range
from ..core.embeddings import EmbeddingService
from ..core.field_extract import extract_fields
from ..core.parse_pdf import extract_text_from_pdf
from ..core.query_rewrite import expand_query
from ..core.retrieval import Retriever
from ..historian import Ledger
from ..historian.schema import IngestEvent, Marker, QueryEvent, RetrievalHit
from ..llm.openai_client import OpenAIClient
from ..store.faiss_store import FaissVectorStore, Metadata

load_dotenv()

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Policy Assistant POC", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_store = FaissVectorStore()
embedding_service = EmbeddingService()
retriever = Retriever(vector_store)
openai_client = OpenAIClient()
ledger = Ledger()


def get_retriever() -> Retriever:
    return retriever


def get_embedding_service() -> EmbeddingService:
    return embedding_service


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "index_size": vector_store.size()}


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),  # noqa: B008
    embeddings: EmbeddingService = Depends(get_embedding_service),  # noqa: B008
) -> JSONResponse:
    start = time.monotonic()
    content = await file.read()
    text, page_breaks = extract_text_from_pdf(content, filename=file.filename)
    if not text:
        raise HTTPException(status_code=400, detail="No text could be extracted from PDF")

    fields = extract_fields(text)

    chunks = chunk_text(text)
    documents = [chunk.text for chunk in chunks]
    vectors = embeddings.embed_documents(documents)

    metadata_items: List[Metadata] = []
    for chunk, _vector in zip(chunks, vectors, strict=True):
        page_start, page_end = map_offsets_to_page_range(chunk, page_breaks)
        metadata_items.append(
            Metadata(
                document_id=file.filename or "uploaded.pdf",
                chunk_id=chunk.id,
                text=chunk.text,
                source=file.filename or "uploaded.pdf",
                page_start=page_start,
                page_end=page_end,
                fields=dict(fields),
            )
        )

    vector_store.add(vectors, metadata_items)
    elapsed = time.monotonic() - start
    duration_ms = int(elapsed * 1000)
    chunk_count = len(chunks)
    embed_batches = max(1, chunk_count // 64 + (1 if chunk_count % 64 else 0))
    markers = [Marker(type="Note", text="POC ingest")]
    if fields:
        summary_parts = []
        for key, value in fields.items():
            truncated_value = value[:60]
            if len(value) > 60:
                truncated_value += "…"
            summary_parts.append(f"{key}={truncated_value}")
        summary = ", ".join(summary_parts)
        markers.append(Marker(type="Note", text=f"Extracted fields: {summary}"))

    ledger.append(
        IngestEvent(
            filename=file.filename or "uploaded.pdf",
            chunks=chunk_count,
            embed_batches=embed_batches,
            duration_ms=duration_ms,
            markers=markers,
        ).model_dump()
    )
    return JSONResponse(
        {
            "chunks": len(chunks),
            "vectors": len(vectors),
            "elapsed_sec": round(elapsed, 2),
        }
    )


@app.post("/query")
async def query(
    payload: Dict[str, Any],
    embeddings: EmbeddingService = Depends(get_embedding_service),  # noqa: B008
    store_retriever: Retriever = Depends(get_retriever),  # noqa: B008
) -> JSONResponse:
    start = time.monotonic()

    query_text = payload.get("query")
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text is required")

    top_k_value = payload.get("top_k")
    if isinstance(top_k_value, int):
        top_k = top_k_value
    elif isinstance(top_k_value, str):
        top_k = int(top_k_value)
    else:
        top_k = int(os.getenv("TOP_K", "3"))
    redact = payload.get("redact")

    expanded_query = expand_query(query_text)
    query_vector = embeddings.embed_query(expanded_query)
    results = store_retriever.search(query_vector, top_k=top_k, redact=redact)

    if not results:
        return JSONResponse({"answer": "I don't know.", "sources": []})

    lower_query = query_text.lower()
    structured_map = {
        "estimated total premium": "estimated_total_premium",
        "total premium": "estimated_total_premium",
        "policy number": "policy_number",
        "premium at inception": "premium_at_inception",
    }
    matched_phrase = None
    requested_field = None
    for phrase in sorted(structured_map.keys(), key=len, reverse=True):
        if phrase in lower_query:
            matched_phrase = phrase
            requested_field = structured_map[phrase]
            break

    field_metadata = None
    field_value = None
    field_from_results = False
    if requested_field:
        for chunk in results:
            value = chunk.metadata.fields.get(requested_field)
            if value:
                field_metadata = chunk.metadata
                field_value = value
                field_from_results = True
                break
        if field_metadata is None:
            for meta in store_retriever.store.metadata:
                value = meta.fields.get(requested_field)
                if value:
                    field_metadata = meta
                    field_value = value
                    break

    snippets = [chunk.text[:500] for chunk in results]
    sources = [
        {
            "source": chunk.source,
            "chunk_id": chunk.chunk_id,
            "score": chunk.score,
            "page_start": chunk.metadata.page_start,
            "page_end": chunk.metadata.page_end,
        }
        for chunk in results
    ]

    markers = [Marker(type="Decision", text="Answer grounded in retrieved context")]

    if requested_field and field_metadata and field_value:
        if matched_phrase == "estimated total premium":
            display_label = "Estimated total premium"
        elif matched_phrase == "total premium":
            display_label = "Total premium"
        elif matched_phrase == "premium at inception":
            display_label = "Premium at inception"
        else:
            display_label = "Policy number"

        page_label = format_page_label(field_metadata.page_start, field_metadata.page_end)
        page_note = f" | {page_label}" if page_label else ""
        answer = (
            f"{display_label}: {field_value} "
            f"(Source: {field_metadata.source}{page_note} | Chunk: {field_metadata.chunk_id})"
        )
        markers.append(Marker(type="Note", text=f"Structured field shortcut: {requested_field}"))
        if not field_from_results:
            if not any(item["chunk_id"] == field_metadata.chunk_id for item in sources):
                sources.append(
                    {
                        "source": field_metadata.source,
                        "chunk_id": field_metadata.chunk_id,
                        "score": 0.0,
                        "page_start": field_metadata.page_start,
                        "page_end": field_metadata.page_end,
                    }
                )
    else:
        context_blocks = store_retriever.build_context(results)
        if not context_blocks:
            return JSONResponse({"answer": "I don't know.", "sources": []})
        answer = openai_client.chat(query=query_text, context_blocks=context_blocks)

    duration_ms = int((time.monotonic() - start) * 1000)
    rhits: List[RetrievalHit] = []
    for chunk in results:
        hasher = hashlib.sha1()
        hasher.update(chunk.text.encode("utf-8"))
        rhits.append(
            RetrievalHit(
                source=chunk.source,
                chunk_id=hasher.hexdigest()[:12],
                score=chunk.score,
                preview=chunk.text[:300],
                page_start=chunk.metadata.page_start,
                page_end=chunk.metadata.page_end,
            )
        )

    ledger.append(
        QueryEvent(
            query=query_text,
            top_k=len(results),
            hits=rhits,
            model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
            max_tokens=int(os.getenv("MAX_TOKENS", "300")),
            temperature=float(os.getenv("TEMPERATURE", "0.2")),
            latency_ms=duration_ms,
            answer_chars=len(answer),
            markers=markers,
        ).model_dump()
    )
    return JSONResponse({"answer": answer, "snippets": snippets, "sources": sources})
