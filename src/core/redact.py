"""Redaction helpers to mask common PII patterns."""

from __future__ import annotations

import os
import re
from typing import Pattern

REDACTION_PATTERNS: dict[str, Pattern[str]] = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", re.IGNORECASE),
    "phone": re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "policy": re.compile(r"\b\d{4}-\d{4}\b"),
    "address": re.compile(
        r"\b\d{1,5}\s+[A-Za-z0-9'.\-]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b",
        re.IGNORECASE,
    ),
}

REDACTION_TOKENS = {
    "email": "[REDACTED_EMAIL]",
    "phone": "[REDACTED_PHONE]",
    "policy": "[REDACTED_POLICY_ID]",
    "address": "[REDACTED_ADDRESS]",
}


def redact_text(value: str, *, enabled: bool | None = None) -> str:
    """Redact PII from text if enabled."""
    if enabled is None:
        enabled = os.getenv("REDACT_PII", "true").lower() in {"1", "true", "yes"}

    if not enabled or not value:
        return value

    redacted = value
    for key, pattern in REDACTION_PATTERNS.items():
        token = REDACTION_TOKENS[key]
        redacted = pattern.sub(token, redacted)
    return redacted
