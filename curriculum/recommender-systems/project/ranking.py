"""Dependency-free top-k recommendation metrics."""

import math


def precision_recall_at_k(ranked, relevant, k):
    if k < 1 or len(ranked) != len(set(ranked)):
        raise ValueError("positive cutoff and unique ranked items required")
    relevant = set(relevant)
    if not relevant:
        raise ValueError("at least one relevant item required")
    hits = sum(item in relevant for item in ranked[:k])
    return hits / k, hits / len(relevant)


def ndcg_at_k(ranked, relevant, k):
    relevant = set(relevant)
    if not relevant or k < 1:
        raise ValueError("relevance and positive cutoff required")
    dcg = sum((item in relevant) / math.log2(index + 2) for index, item in enumerate(ranked[:k]))
    ideal = sum(1 / math.log2(index + 2) for index in range(min(k, len(relevant))))
    return dcg / ideal
