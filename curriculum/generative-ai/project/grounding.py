"""Transparent citation-grounding metrics for RAG evaluation fixtures."""

from __future__ import annotations


def citation_precision(claim_citations: list[set[str]], supporting: set[tuple[int, str]]) -> float:
    """Fraction of supplied claim/citation pairs marked as supporting."""
    pairs = [(index, citation) for index, citations in enumerate(claim_citations) for citation in citations]
    if not pairs:
        return 0.0
    return sum(pair in supporting for pair in pairs) / len(pairs)


def citation_coverage(claim_citations: list[set[str]], required_claims: set[int]) -> float:
    if not required_claims:
        raise ValueError("required_claims cannot be empty")
    if any(index < 0 or index >= len(claim_citations) for index in required_claims):
        raise ValueError("required claim index is out of range")
    return sum(bool(claim_citations[index]) for index in required_claims) / len(required_claims)


def retrieval_recall(relevant: set[str], retrieved: list[str]) -> float:
    if not relevant:
        raise ValueError("relevant documents cannot be empty")
    return len(relevant.intersection(retrieved)) / len(relevant)
