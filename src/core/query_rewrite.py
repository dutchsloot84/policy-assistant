"""Query rewriting helpers for synonym expansion."""

from __future__ import annotations

from typing import List, Set


def _append_synonyms(existing_terms: List[str], query_lower: str, synonyms: List[str]) -> List[str]:
    additions: List[str] = []
    existing_lower: Set[str] = {term.lower() for term in existing_terms}
    for synonym in synonyms:
        synonym_lower = synonym.lower()
        if synonym_lower in query_lower:
            continue
        if synonym_lower in existing_lower:
            continue
        if synonym_lower in {term.lower() for term in additions}:
            continue
        additions.append(synonym)
    return additions


def expand_query(query: str) -> str:
    """Append common synonyms to improve recall for structured fields."""

    lowered = query.lower()
    appended: List[str] = []

    if "policy number" in lowered:
        appended.extend(_append_synonyms(appended, lowered, ["policy #", "policy no", "policy id"]))
    if "estimated total premium" in lowered:
        appended.extend(
            _append_synonyms(
                appended,
                lowered,
                ["total premium", "premium total", "premium overall"],
            )
        )
    elif "total premium" in lowered:
        appended.extend(
            _append_synonyms(
                appended,
                lowered,
                ["estimated total premium", "premium total", "premium overall"],
            )
        )
    if "premium at inception" in lowered:
        appended.extend(
            _append_synonyms(
                appended, lowered, ["payable at inception premium", "inception premium"]
            )
        )

    if not appended:
        return query

    synonyms_text = " ".join(f'"{syn}"' if " " in syn else syn for syn in appended)
    if synonyms_text:
        return f"{query} {synonyms_text}".strip()
    return query
