"""Classify questions into categories WITHOUT LLM calls (fast, free, deterministic).

Categories: factual, analytical, relational, procedural, vague, visual.
The bandit router learns the best agent per category.
"""
from __future__ import annotations

import re

CATEGORY_PATTERNS: dict[str, list[str]] = {
    "factual": [
        r"\b(what is|who is|when did|where is|define|what does|what are)\b",
        r"\b(name of|tell me the|which one)\b",
    ],
    "analytical": [
        r"\b(compare|analyze|how many|average|total|sum|count|trend|percentage|ratio)\b",
        r"\b(statistics|data|numbers|grouped by|breakdown)\b",
    ],
    "relational": [
        r"\b(relat|connect|between|linked|associated|works with|reports to)\b",
        r"\b(hierarchy|network|chain|depend)\b",
    ],
    "procedural": [
        r"\b(how to|how do|steps to|process of|explain how|guide|tutorial)\b",
        r"\b(implement|build|create|set up|configure)\b",
    ],
    "visual": [
        r"\b(image|picture|diagram|chart|screenshot|photo|figure|graph|plot)\b",
    ],
}


def classify_question(question: str) -> str:
    """Classify a question into a category using pattern matching."""
    q = question.lower().strip()

    scores: dict[str, int] = {}
    for category, patterns in CATEGORY_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, q))
        if score > 0:
            scores[category] = score

    if not scores:
        return "vague" if len(q.split()) <= 4 else "factual"
    return max(scores, key=scores.get)
