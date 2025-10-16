"""Utilities for extracting normalized text from PDF documents."""

from __future__ import annotations

import logging
import re
import tempfile
from io import BytesIO
from typing import Callable, List, Optional, Sequence

from pypdf import PdfReader

PdfMinerExtractor = Callable[[BytesIO], str]

try:  # pragma: no cover - optional dependency path
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:  # pragma: no cover - guard against optional import errors
    pdfminer_func: Optional[PdfMinerExtractor] = None
else:  # pragma: no cover - optional dependency available
    pdfminer_func = pdfminer_extract_text  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)


def normalize_for_chunking(text: str) -> str:
    """Normalize PDF text while keeping table-friendly spacing."""

    normalized = text.replace("\r", "\n").replace("\t", " ")
    normalized = re.sub(r"[ ]{3,}", "  ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


PAGE_BREAK_SENTINEL = "\f"


def extract_text_from_pdf(
    file_bytes: bytes, *, filename: Optional[str] = None
) -> tuple[str, List[int]]:
    """Extract normalized text and per-page offsets from PDF bytes."""

    if not file_bytes:
        return "", []

    page_texts = _extract_with_pypdf(file_bytes)
    if page_texts:
        return _normalize_with_page_breaks(page_texts)

    LOGGER.warning(
        "Primary PDF extraction failed; attempting pdfminer fallback for %s",
        filename or "unknown file",
    )
    pdfminer_text = _extract_with_pdfminer(file_bytes)
    if pdfminer_text:
        return _normalize_with_page_breaks([pdfminer_text])

    LOGGER.warning(
        "Both primary PDF extractors failed; attempting OCR fallback for %s",
        filename or "unknown file",
    )
    ocr_pages = _extract_with_ocr(file_bytes, filename=filename)
    if ocr_pages:
        return _normalize_with_page_breaks(ocr_pages)

    return "", []


def _extract_with_pypdf(file_bytes: bytes) -> List[str]:
    try:
        reader = PdfReader(BytesIO(file_bytes))
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.error("Failed to read PDF with pypdf", exc_info=exc)
        return []

    texts = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.warning("Failed to extract page with pypdf", exc_info=exc)
            page_text = ""
        texts.append(page_text)

    return texts


def _extract_with_pdfminer(file_bytes: bytes) -> str:
    if pdfminer_func is None:
        return ""

    try:
        return pdfminer_func(BytesIO(file_bytes)) or ""
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.error("Failed to extract PDF with pdfminer", exc_info=exc)
        return ""


def _extract_with_ocr(file_bytes: bytes, *, filename: Optional[str] = None) -> List[str]:
    """Attempt OCR-based extraction using optional dependencies."""

    try:  # pragma: no cover - optional dependency path
        import ocrmypdf  # type: ignore[import-untyped]
    except Exception:  # pragma: no cover - guard against optional import errors
        LOGGER.warning(
            (
                "OCR dependencies unavailable (missing ocrmypdf); "
                "install optional extras to enable OCR for %s"
            ),
            filename or "unknown file",
        )
        return []

    try:  # pragma: no cover - optional dependency path
        import pytesseract  # type: ignore[import-untyped]
    except Exception:  # pragma: no cover - guard against optional import errors
        LOGGER.warning(
            (
                "OCR dependencies unavailable (missing pytesseract); "
                "install optional extras to enable OCR for %s"
            ),
            filename or "unknown file",
        )
        return []
    _ = pytesseract  # ensure the import is referenced for linters

    try:
        with (
            tempfile.NamedTemporaryFile(suffix=".pdf") as input_tmp,
            tempfile.NamedTemporaryFile(suffix=".pdf") as output_tmp,
        ):
            input_tmp.write(file_bytes)
            input_tmp.flush()

            ocrmypdf.ocr(  # type: ignore[attr-defined]
                input_tmp.name,
                output_tmp.name,
                progress_bar=False,
                skip_text=True,
                force_ocr=True,
            )

            output_tmp.seek(0)
            ocr_bytes = output_tmp.read()
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.error("OCR processing failed for %s", filename or "unknown file", exc_info=exc)
        return []

    if not ocr_bytes:
        LOGGER.warning("OCR processing produced no output for %s", filename or "unknown file")
        return []

    # Try the extractors again on the OCR-processed PDF bytes.
    page_texts = _extract_with_pypdf(ocr_bytes)
    if page_texts:
        return page_texts

    fallback_text = _extract_with_pdfminer(ocr_bytes)
    return [fallback_text] if fallback_text else []


def _normalize_with_page_breaks(pages: Sequence[str]) -> tuple[str, List[int]]:
    if not pages:
        return "", []

    combined = PAGE_BREAK_SENTINEL.join(pages)
    normalized = normalize_for_chunking(combined)

    segments = normalized.split(PAGE_BREAK_SENTINEL)
    parts: list[str] = []
    page_breaks: list[int] = []
    offset = 0
    for index, segment in enumerate(segments):
        if index > 0:
            parts.append("\n\n")
            offset += 2
        page_breaks.append(offset)
        parts.append(segment)
        offset += len(segment)

    raw_text = "".join(parts)
    trimmed_text = raw_text.strip()
    if not trimmed_text:
        return "", []

    leading_trim = len(raw_text) - len(raw_text.lstrip())
    if leading_trim:
        page_breaks = [max(0, break_pos - leading_trim) for break_pos in page_breaks]

    return trimmed_text, page_breaks
