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

from ..core.chunk import chunk_text
from ..core.embeddings import EmbeddingService
from ..core.parse_pdf import extract_text_from_pdf
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
    text = extract_text_from_pdf(content, filename=file.filename)
    if not text:
        raise HTTPException(status_code=400, detail="No text could be extracted from PDF")

    chunks = chunk_text(text)
    documents = [chunk.text for chunk in chunks]
    vectors = embeddings.embed_documents(documents)

    metadata_items: List[Metadata] = []
    for chunk, _vector in zip(chunks, vectors, strict=True):
        metadata_items.append(
            Metadata(
                document_id=file.filename or "uploaded.pdf",
                chunk_id=chunk.id,
                text=chunk.text,
                source=file.filename or "uploaded.pdf",
            )
        )

    vector_store.add(vectors, metadata_items)
    elapsed = time.monotonic() - start
    duration_ms = int(elapsed * 1000)
    chunk_count = len(chunks)
    embed_batches = max(1, chunk_count // 64 + (1 if chunk_count % 64 else 0))
    ledger.append(
        IngestEvent(
            filename=file.filename or "uploaded.pdf",
            chunks=chunk_count,
            embed_batches=embed_batches,
            duration_ms=duration_ms,
            markers=[Marker(type="Note", text="POC ingest")],
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

    query_vector = embeddings.embed_query(query_text)
    results = store_retriever.search(query_vector, top_k=top_k, redact=redact)
    context_blocks = store_retriever.build_context(results)

    if not context_blocks:
        return JSONResponse({"answer": "I don't know.", "sources": []})

    answer = openai_client.chat(query=query_text, context_blocks=context_blocks)
    snippets = [chunk.text[:500] for chunk in results]
    sources = [
        {
            "source": chunk.source,
            "chunk_id": chunk.chunk_id,
            "score": chunk.score,
        }
        for chunk in results
    ]
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
            markers=[
                Marker(type="Decision", text="Answer grounded in retrieved context"),
            ],
        ).model_dump()
    )
    return JSONResponse({"answer": answer, "snippets": snippets, "sources": sources})
