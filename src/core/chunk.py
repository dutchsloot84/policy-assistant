"""Chunking utilities for policy document text."""

from __future__ import annotations

import logging
import math
import os
import re
import uuid
from dataclasses import dataclass
from typing import Iterable, List

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from nltk.tokenize import sent_tokenize  # type: ignore
except Exception:  # pragma: no cover
    sent_tokenize = None


def _get_env_int(name: str, default: int) -> int:
    """Parse an integer environment variable with validation."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError:
        LOGGER.warning("Invalid value for %s=%s; using default %s", name, raw_value, default)
        return default

    if value < 0:
        LOGGER.warning("Negative value for %s=%s; using default %s", name, raw_value, default)
        return default

    return value


@dataclass
class Chunk:
    """Represents a chunk of text."""

    id: str
    text: str
    start: int
    end: int


DEFAULT_MAX_CHARS = _get_env_int("CHUNK_MAX_CHARS", 1200)
DEFAULT_OVERLAP = _get_env_int("CHUNK_OVERLAP", 150)


def chunk_text(
    text: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> List[Chunk]:
    """Split text into overlapping chunks, sentence-aware if possible."""
    cleaned = text.strip()
    if not cleaned:
        return []

    sentences = _split_into_sentences(cleaned)
    chunks: List[Chunk] = []
    buffer: list[str] = []
    start_index = 0

    def flush_buffer() -> None:
        nonlocal buffer, start_index
        if not buffer:
            return
        chunk_text_value = " ".join(buffer).strip()
        if not chunk_text_value:
            buffer = []
            return
        chunk_id = str(uuid.uuid4())
        end_index = start_index + len(chunk_text_value)
        chunks.append(Chunk(id=chunk_id, text=chunk_text_value, start=start_index, end=end_index))
        # prepare overlap
        if overlap > 0:
            overlap_text = chunk_text_value[-overlap:].lstrip()
            buffer = [overlap_text] if overlap_text else []
        else:
            buffer = []
        start_index = max(0, end_index - overlap)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_chars:
            flush_buffer()
            for piece in _hard_split(sentence, max_chars):
                chunk_id = str(uuid.uuid4())
                end_index = start_index + len(piece)
                chunks.append(Chunk(id=chunk_id, text=piece, start=start_index, end=end_index))
                start_index = max(0, end_index - overlap)
            continue

        prospective = len(" ".join(buffer + [sentence]).strip()) if buffer else len(sentence)
        if prospective > max_chars:
            flush_buffer()
            # after flush, buffer may contain overlap text; re-evaluate length
            prospective = len(" ".join(buffer + [sentence]).strip()) if buffer else len(sentence)
            while prospective > max_chars:
                # sentence itself still too long with overlap; split sentence and continue
                for piece in _hard_split(sentence, max_chars):
                    chunk_id = str(uuid.uuid4())
                    end_index = start_index + len(piece)
                    chunks.append(Chunk(id=chunk_id, text=piece, start=start_index, end=end_index))
                    start_index = max(0, end_index - overlap)
                sentence = ""
                break
            if not sentence:
                continue
        buffer.append(sentence)

    flush_buffer()
    return chunks


def _split_into_sentences(text: str) -> Iterable[str]:
    if sent_tokenize is None:
        return re.split(r"(?<=[.!?])\s+", text)
    try:  # pragma: no cover
        return sent_tokenize(text)
    except LookupError:  # pragma: no cover
        LOGGER.warning("NLTK punkt model missing; using regex fallback")
        return re.split(r"(?<=[.!?])\s+", text)


def _hard_split(sentence: str, max_chars: int) -> Iterable[str]:
    total = len(sentence)
    parts = int(math.ceil(total / max_chars))
    for i in range(parts):
        start = i * max_chars
        end = min(start + max_chars, total)
        yield sentence[start:end]
