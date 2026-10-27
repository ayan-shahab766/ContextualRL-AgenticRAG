"""Evidence packaging and transparent lexical grounding diagnostics."""
from __future__ import annotations

import re

_STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or", "that", "the", "to", "with"}


def _terms(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9_]+", text.lower()) if len(word) > 2 and word not in _STOPWORDS}


def build_evidence(answer: str, sources: list[dict]) -> dict:
    """Return stable evidence IDs and a clearly-labeled lexical support estimate.

    This is intentionally a retrieval diagnostic, not a claim of semantic factuality.
    """
    evidence: list[dict] = []
    answer_terms = _terms(answer)
    overlaps: list[float] = []
    for index, source in enumerate(sources, start=1):
        metadata = source.get("metadata", {}) or {}
        content = str(source.get("content", ""))
        source_terms = _terms(content)
        overlap = len(answer_terms & source_terms) / max(1, len(answer_terms))
        overlaps.append(overlap)
        locator = metadata.get("page")
        if locator is None and "row_start" in metadata:
            locator = f"row {metadata['row_start']}"
        evidence.append(
            {
                "evidence_id": f"E{index}",
                "source": metadata.get("source_filename") or metadata.get("source") or "indexed_document",
                "locator": locator,
                "excerpt": content[:300],
                "lexical_support": round(overlap, 3),
            }
        )

    coverage = sum(overlaps) / len(overlaps) if overlaps else 0.0
    return {
        "items": evidence,
        "grounding": {
            "status": "supported" if evidence and coverage >= 0.12 else "needs_review",
            "lexical_evidence_coverage": round(coverage, 3),
            "method": "retrieval_lexical_overlap",
            "note": "Coverage measures answer/source term overlap and should be reviewed for high-stakes decisions.",
        },
    }
