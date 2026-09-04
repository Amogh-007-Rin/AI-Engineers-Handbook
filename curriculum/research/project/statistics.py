"""Dependency-free summaries for repeated experimental measurements."""

from __future__ import annotations

import math


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("values cannot be empty")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("values must be finite")
    return sum(values) / len(values)


def sample_standard_deviation(values: list[float]) -> float:
    if len(values) < 2:
        raise ValueError("at least two values are required")
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def paired_differences(baseline: list[float], candidate: list[float]) -> list[float]:
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("paired samples must have equal nonzero length")
    return [new - old for old, new in zip(baseline, candidate, strict=True)]
