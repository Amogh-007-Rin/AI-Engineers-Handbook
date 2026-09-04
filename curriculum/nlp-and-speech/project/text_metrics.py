"""Small dependency-free text metrics for learning and test fixtures."""

from __future__ import annotations

from collections import Counter


def token_f1(reference: list[str], prediction: list[str]) -> tuple[float, float, float]:
    overlap = sum((Counter(reference) & Counter(prediction)).values())
    precision = overlap / len(prediction) if prediction else 0.0
    recall = overlap / len(reference) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def recall_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    if not relevant:
        raise ValueError("relevant set cannot be empty")
    if k <= 0:
        raise ValueError("k must be positive")
    return len(relevant.intersection(ranked[:k])) / len(relevant)
