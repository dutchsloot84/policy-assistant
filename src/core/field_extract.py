"""Lightweight regex-based field extraction from policy documents."""

from __future__ import annotations

import re
from typing import Dict

POLICY_NUMBER_RE = re.compile(r"\bPOLICY\s*NUMBER\s*[:#]?\s*([A-Z0-9\-]+)", re.I)
_AMOUNT_PATTERN = r"([$€£]?\s*[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)"
ESTIMATED_TOTAL_PREMIUM_RE = re.compile(
    rf"(?:ESTIMATED\s+TOTAL\s+PREMIUM|TOTAL\s+PREMIUM)[^0-9$€£]*{_AMOUNT_PATTERN}",
    re.I | re.S,
)
PREMIUM_AT_INCEPTION_RE = re.compile(
    rf"PREMIUM\s+SHOWN\s+IS\s+PAYABLE\s+AT\s+INCEPTION[^0-9$€£]*{_AMOUNT_PATTERN}",
    re.I | re.S,
)


def extract_fields(page_or_doc_text: str) -> Dict[str, str]:
    """Extract structured field values from normalized policy text."""

    if not page_or_doc_text:
        return {}

    fields: Dict[str, str] = {}

    policy_match = POLICY_NUMBER_RE.search(page_or_doc_text)
    if policy_match:
        fields["policy_number"] = policy_match.group(1).strip()

    premium_match = ESTIMATED_TOTAL_PREMIUM_RE.search(page_or_doc_text)
    if premium_match:
        fields["estimated_total_premium"] = premium_match.group(1).strip()

    inception_match = PREMIUM_AT_INCEPTION_RE.search(page_or_doc_text)
    if inception_match:
        fields["premium_at_inception"] = inception_match.group(1).strip()

    return fields
