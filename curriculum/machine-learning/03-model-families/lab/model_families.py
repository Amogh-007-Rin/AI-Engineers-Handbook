"""Transparent representatives of classical model-family inductive biases."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import asdict, dataclass
import json
import math
import random


def finite(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} must be non-empty")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite values")
    return result


@dataclass(frozen=True)
class Line:
    intercept: float
    slope: float

    def predict(self, value: float) -> float:
        return self.intercept + self.slope * value


def fit_line(features: Sequence[float], targets: Sequence[float]) -> Line:
    x, y = finite(features, "features"), finite(targets, "targets")
    if len(x) != len(y):
        raise ValueError("features and targets must have equal length")
    x_mean, y_mean = math.fsum(x) / len(x), math.fsum(y) / len(y)
    denominator = math.fsum((value - x_mean) ** 2 for value in x)
    if denominator == 0:
        raise ValueError("linear slope is undefined for constant features")
    slope = math.fsum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / denominator
    return Line(y_mean - slope * x_mean, slope)


def knn_classify(features: Sequence[float], labels: Sequence[Hashable], query: float, k: int) -> Hashable:
    x = finite(features, "features")
    if len(x) != len(labels) or not labels:
        raise ValueError("features and labels must have equal non-zero length")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(x):
        raise ValueError("k must be an integer between one and training size")
    if not math.isfinite(query):
        raise ValueError("query must be finite")
    nearest = sorted(((abs(value - query), index, labels[index]) for index, value in enumerate(x)))[:k]
    counts = Counter(label for _, _, label in nearest)
    return min(counts, key=lambda label: (-counts[label], repr(label)))


@dataclass(frozen=True)
class Stump:
    threshold: float
    left_label: int
    right_label: int
    errors: int

    def predict(self, value: float) -> int:
        return self.left_label if value <= self.threshold else self.right_label


def majority(labels: Sequence[int]) -> int:
    counts = Counter(labels)
    return min(counts, key=lambda label: (-counts[label], label))


def fit_stump(features: Sequence[float], labels: Sequence[int]) -> Stump:
    x = finite(features, "features")
    if len(x) != len(labels) or not labels:
        raise ValueError("features and labels must have equal non-zero length")
    if any(label not in {0, 1} or isinstance(label, bool) for label in labels):
        raise ValueError("stump labels must be integer zero or one")
    unique = sorted(set(x))
    if len(unique) < 2:
        raise ValueError("stump requires at least two unique feature values")
    candidates = []
    for left_value, right_value in zip(unique, unique[1:]):
        threshold = (left_value + right_value) / 2
        left = [label for value, label in zip(x, labels) if value <= threshold]
        right = [label for value, label in zip(x, labels) if value > threshold]
        left_label, right_label = majority(left), majority(right)
        errors = sum(label != (left_label if value <= threshold else right_label)
                     for value, label in zip(x, labels))
        candidates.append(Stump(threshold, left_label, right_label, errors))
    return min(candidates, key=lambda item: (item.errors, item.threshold, item.left_label, item.right_label))


def kmeans_1d(values: Sequence[float], clusters: int, seed: int = 0,
              max_iterations: int = 100, tolerance: float = 1e-9) -> tuple[float, ...]:
    data = finite(values, "values")
    unique = sorted(set(data))
    if isinstance(clusters, bool) or not isinstance(clusters, int) or not 1 <= clusters <= len(unique):
        raise ValueError("clusters must be between one and the number of unique values")
    if max_iterations < 1 or tolerance < 0:
        raise ValueError("iteration limit must be positive and tolerance non-negative")
    centers = [float(value) for value in random.Random(seed).sample(unique, clusters)]
    for _ in range(max_iterations):
        groups = [[] for _ in centers]
        for value in data:
            index = min(range(len(centers)), key=lambda i: (abs(value - centers[i]), i))
            groups[index].append(value)
        updated = [math.fsum(group) / len(group) if group else centers[index]
                   for index, group in enumerate(groups)]
        if max(abs(before - after) for before, after in zip(centers, updated)) <= tolerance:
            centers = updated
            break
        centers = updated
    return tuple(sorted(centers))


def example() -> dict[str, object]:
    line = fit_line([1, 2, 3], [2, 4, 6])
    stump = fit_stump([0, 1, 2, 3], [0, 0, 1, 1])
    return {
        "line": asdict(line),
        "prediction_at_4": line.predict(4),
        "neighbor_class": knn_classify([0, 1, 9, 10], ["cold", "cold", "warm", "warm"], 8, 3),
        "stump": asdict(stump),
        "centroids": kmeans_1d([0, 1, 9, 10], 2, seed=3),
    }


def main() -> int:
    print(json.dumps(example(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
