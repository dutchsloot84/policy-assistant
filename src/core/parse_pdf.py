"""Utilities for extracting normalized text from PDF documents."""

from __future__ import annotations

import logging
import re
import tempfile
from io import BytesIO
from typing import Callable, Optional

from pypdf import PdfReader

PdfMinerExtractor = Callable[[BytesIO], str]

try:  # pragma: no cover - optional dependency path
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:  # pragma: no cover - guard against optional import errors
    pdfminer_func: Optional[PdfMinerExtractor] = None
else:  # pragma: no cover - optional dependency available
    pdfminer_func = pdfminer_extract_text  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)


def _normalize_whitespace(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value)
    return normalized.strip()


def extract_text_from_pdf(file_bytes: bytes, *, filename: Optional[str] = None) -> str:
    """Extract text from PDF bytes using pypdf with pdfminer fallback."""
    if not file_bytes:
        return ""

    pypdf_text = _extract_with_pypdf(file_bytes)
    if pypdf_text:
        return _normalize_whitespace(pypdf_text)

    LOGGER.warning(
        "Primary PDF extraction failed; attempting pdfminer fallback for %s",
        filename or "unknown file",
    )
    pdfminer_text = _extract_with_pdfminer(file_bytes)
    if pdfminer_text:
        return _normalize_whitespace(pdfminer_text)

    LOGGER.warning(
        "Both primary PDF extractors failed; attempting OCR fallback for %s",
        filename or "unknown file",
    )
    ocr_text = _extract_with_ocr(file_bytes, filename=filename)
    return _normalize_whitespace(ocr_text) if ocr_text else ""


def _extract_with_pypdf(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(file_bytes))
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.error("Failed to read PDF with pypdf", exc_info=exc)
        return ""

    texts = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.warning("Failed to extract page with pypdf", exc_info=exc)
            page_text = ""
        texts.append(page_text)

    combined = " ".join(filter(None, texts))
    return combined


def _extract_with_pdfminer(file_bytes: bytes) -> str:
    if pdfminer_func is None:
        return ""

    try:
        return pdfminer_func(BytesIO(file_bytes)) or ""
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.error("Failed to extract PDF with pdfminer", exc_info=exc)
        return ""


def _extract_with_ocr(file_bytes: bytes, *, filename: Optional[str] = None) -> str:
    """Attempt OCR-based extraction using optional dependencies."""

    try:  # pragma: no cover - optional dependency path
        import ocrmypdf  # type: ignore[import-untyped]
    except Exception:  # pragma: no cover - guard against optional import errors
        LOGGER.warning(
            "OCR dependencies unavailable (missing ocrmypdf); install optional extras to enable OCR for %s",
            filename or "unknown file",
        )
        return ""

    try:  # pragma: no cover - optional dependency path
        import pytesseract  # type: ignore[import-untyped]
    except Exception:  # pragma: no cover - guard against optional import errors
        LOGGER.warning(
            "OCR dependencies unavailable (missing pytesseract); install optional extras to enable OCR for %s",
            filename or "unknown file",
        )
        return ""
    _ = pytesseract  # ensure the import is referenced for linters

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as input_tmp, tempfile.NamedTemporaryFile(
            suffix=".pdf"
        ) as output_tmp:
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
        LOGGER.error(
            "OCR processing failed for %s", filename or "unknown file", exc_info=exc
        )
        return ""

    if not ocr_bytes:
        LOGGER.warning(
            "OCR processing produced no output for %s", filename or "unknown file"
        )
        return ""

    # Try the extractors again on the OCR-processed PDF bytes.
    text = _extract_with_pypdf(ocr_bytes) or _extract_with_pdfminer(ocr_bytes)
    return text or ""
