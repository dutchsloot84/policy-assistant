"""Embedding utilities with local caching and deduplication."""

from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from dotenv import load_dotenv

from ..llm.openai_client import OpenAIClient
from .cost_guard import estimate_tokens

load_dotenv()

CACHE_FILE = Path(os.getenv("EMBED_CACHE_PATH", "data/emb_cache.pkl"))
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
MAX_EMBED_TOKENS = int(os.getenv("MAX_EMBED_TOKENS", "6000"))


class EmbeddingService:
    def __init__(
        self, *, client: OpenAIClient | None = None, cache_file: Path | None = None
    ) -> None:
        self.client = client or OpenAIClient()
        self.cache_file = cache_file or CACHE_FILE
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, List[float]] = self._load_cache()
        self.embed_max_tokens = MAX_EMBED_TOKENS

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

    def _embed(self, texts: Sequence[str]) -> List[List[float]]:
        deduped: Dict[str, str] = {}
        order: List[str] = []
        for text in texts:
            key = self._hash_text(text)
            deduped.setdefault(key, text)
            order.append(key)

        embeddings: Dict[str, List[float]] = {}
        missing = [key for key in deduped if key not in self._cache]

        if missing:
            batches = [missing[i : i + BATCH_SIZE] for i in range(0, len(missing), BATCH_SIZE)]
            for batch_keys in batches:
                batch_texts = [deduped[key] for key in batch_keys]
                vectors = self._request_embeddings(batch_texts)
                for key, vector in zip(batch_keys, vectors, strict=True):
                    self._cache[key] = vector
            self._persist_cache()

        for key in order:
            embeddings[key] = self._cache[key]
        return [embeddings[key] for key in order]

    def _request_embeddings(self, texts: Iterable[str]) -> List[List[float]]:
        payload = list(texts)
        max_tokens = self.embed_max_tokens
        if max_tokens > 0:
            for item in payload:
                if estimate_tokens(item) > max_tokens:
                    raise ValueError("Estimated tokens for embedding input exceed MAX_EMBED_TOKENS")
        return self.client.embed_texts(payload)

    def _load_cache(self) -> Dict[str, List[float]]:
        if not self.cache_file.exists():
            return {}
        try:
            with self.cache_file.open("rb") as fh:
                data = pickle.load(fh)
            return data if isinstance(data, dict) else {}
        except Exception:  # pragma: no cover - defensive
            return {}

    def _persist_cache(self) -> None:
        with self.cache_file.open("wb") as fh:
            pickle.dump(self._cache, fh)

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
