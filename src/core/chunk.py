"""Chunking utilities for policy document text."""

from __future__ import annotations

import logging
import math
import os
import re
import uuid
from bisect import bisect_right
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

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


DEFAULT_MAX_CHARS = _get_env_int("CHUNK_MAX_CHARS", 550)
DEFAULT_OVERLAP = _get_env_int("CHUNK_OVERLAP", 90)


def chunk_text(
    text: str,
    *,
    max_chars: int | None = None,
    overlap: int | None = None,
) -> List[Chunk]:
    """Split text into overlapping chunks, sentence-aware if possible."""
    resolved_max_chars = (
        max_chars if max_chars is not None else _get_env_int("CHUNK_MAX_CHARS", DEFAULT_MAX_CHARS)
    )
    resolved_overlap = (
        overlap if overlap is not None else _get_env_int("CHUNK_OVERLAP", DEFAULT_OVERLAP)
    )

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
        if resolved_overlap > 0:
            overlap_text = chunk_text_value[-resolved_overlap:].lstrip()
            buffer = [overlap_text] if overlap_text else []
        else:
            buffer = []
        start_index = max(0, end_index - resolved_overlap)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > resolved_max_chars:
            flush_buffer()
            for piece in _hard_split(sentence, resolved_max_chars):
                chunk_id = str(uuid.uuid4())
                end_index = start_index + len(piece)
                chunks.append(Chunk(id=chunk_id, text=piece, start=start_index, end=end_index))
                start_index = max(0, end_index - resolved_overlap)
            continue

        prospective = len(" ".join(buffer + [sentence]).strip()) if buffer else len(sentence)
        if prospective > resolved_max_chars:
            flush_buffer()
            # after flush, buffer may contain overlap text; re-evaluate length
            prospective = len(" ".join(buffer + [sentence]).strip()) if buffer else len(sentence)
            while prospective > resolved_max_chars:
                # sentence itself still too long with overlap; split sentence and continue
                for piece in _hard_split(sentence, resolved_max_chars):
                    chunk_id = str(uuid.uuid4())
                    end_index = start_index + len(piece)
                    chunks.append(Chunk(id=chunk_id, text=piece, start=start_index, end=end_index))
                    start_index = max(0, end_index - resolved_overlap)
                sentence = ""
                break
            if not sentence:
                continue
        buffer.append(sentence)

    flush_buffer()
    return chunks


def map_offsets_to_page_range(chunk: Chunk, page_breaks: Sequence[int]) -> Tuple[int, int]:
    """Return the 1-indexed page range that best matches the chunk offsets."""

    if not page_breaks:
        return (1, 1)

    start_page_index = bisect_right(page_breaks, chunk.start) - 1
    start_page_index = max(0, start_page_index)

    effective_end = max(chunk.start, chunk.end - 1)
    end_page_index = bisect_right(page_breaks, effective_end) - 1
    end_page_index = max(0, end_page_index)

    last_index = len(page_breaks) - 1
    start_page_index = min(start_page_index, last_index)
    end_page_index = min(end_page_index, last_index)

    return (start_page_index + 1, end_page_index + 1)


def format_page_label(page_start: int | None, page_end: int | None) -> str:
    """Return a human-readable label for a page range."""

    if page_start is None:
        return ""
    if page_end is None or page_end == page_start:
        return f"Page {page_start}"
    return f"Pages {page_start}–{page_end}"


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
