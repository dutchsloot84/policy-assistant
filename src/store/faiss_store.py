"""Local FAISS index management."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import faiss  # type: ignore
import numpy as np
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
INDEX_PATH = DATA_DIR / "index.faiss"
META_PATH = DATA_DIR / "meta.pkl"


@dataclass
class Metadata:
    document_id: str
    chunk_id: str
    text: str
    source: str
    fields: Dict[str, str] = field(default_factory=dict)


class FaissVectorStore:
    def __init__(self, *, index_path: Path | None = None, meta_path: Path | None = None) -> None:
        self.index_path = index_path or INDEX_PATH
        self.meta_path = meta_path or META_PATH
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.index: faiss.Index | None = None
        self.metadata: List[Metadata] = []
        self._load()

    def _load(self) -> None:
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        if self.meta_path.exists():
            with self.meta_path.open("rb") as fh:
                data = pickle.load(fh)
                self.metadata = data if isinstance(data, list) else []
        if self.index is None and self.metadata:
            self.metadata = []

    def _persist(self) -> None:
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        with self.meta_path.open("wb") as fh:
            pickle.dump(self.metadata, fh)

    def add(self, embeddings: Sequence[Sequence[float]], metadatas: Sequence[Metadata]) -> None:
        if not embeddings:
            return
        vectors = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(vectors)
        if self.index is None:
            dimension = vectors.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
        self.index.add(vectors)
        self.metadata.extend(metadatas)
        self._persist()

    def search(
        self, embedding: Sequence[float] | Iterable[float], k: int
    ) -> List[tuple[float, Metadata]]:
        if self.index is None:
            return []
        vector = np.array([embedding], dtype="float32")
        faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, k)
        results: List[tuple[float, Metadata]] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx == -1:
                continue
            if idx < len(self.metadata):
                results.append((float(score), self.metadata[idx]))
        return results

    def size(self) -> int:
        return len(self.metadata)
