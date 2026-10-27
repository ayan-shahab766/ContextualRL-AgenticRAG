"""Deterministic query signals used by contextual_rl_agentic_rag's contextual routing policy."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from contextual_rl_agentic_rag.rl.question_classifier import classify_question


@dataclass(frozen=True)
class QuerySignals:
    """Compact, explainable context for a routing decision."""

    category: str
    domain: str
    modality: str
    complexity: str
    intent: str
    token_count: int

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def context_key(self) -> str:
        return "|".join((self.category, self.domain, self.modality, self.complexity))


_DOMAIN_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("technical", (r"\b(api|sdk|code|python|java|typescript|database|sql|docker|kubernetes|model|algorithm)\b",)),
    ("financial", (r"\b(revenue|profit|loss|budget|invoice|market|portfolio|cash flow)\b",)),
    ("legal", (r"\b(contract|clause|law|regulation|compliance|liability|court)\b",)),
    ("research", (r"\b(study|paper|methodology|hypothesis|experiment|dataset|citation)\b",)),
)


def derive_query_signals(question: str) -> QuerySignals:
    """Derive routing context without adding an LLM call or external dependency."""
    q = question.lower().strip()
    token_count = len(re.findall(r"\w+", q))
    category = classify_question(q)

    domain = "general"
    for candidate, patterns in _DOMAIN_PATTERNS:
        if any(re.search(pattern, q) for pattern in patterns):
            domain = candidate
            break

    modality = "visual" if category == "visual" else "tabular" if category == "analytical" and re.search(
        r"\b(table|csv|spreadsheet|column|row|metric)\b", q
    ) else "text"
    reasoning_terms = len(re.findall(r"\b(compare|analyze|why|how|relationship|trade-?off|step|evaluate)\b", q))
    complexity = "high" if token_count > 22 or reasoning_terms >= 2 else "medium" if token_count > 8 or reasoning_terms else "low"
    intent = "analysis" if category == "analytical" else "relationship" if category == "relational" else "procedure" if category == "procedural" else "lookup"

    return QuerySignals(category, domain, modality, complexity, intent, token_count)
