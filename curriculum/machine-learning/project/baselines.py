"""Dependency-free baselines and metrics for the stage project."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
import random


Label = Hashable


def validate_labels(actual: Sequence[Label], predicted: Sequence[Label]) -> None:
    if not actual:
        raise ValueError("evaluation requires at least one observation")
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted lengths differ")


@dataclass
class MajorityClassifier:
    """Predict the most frequent training label; ties use stable representation."""

    label_: Label | None = None

    def fit(self, labels: Iterable[Label]) -> "MajorityClassifier":
        counts = Counter(labels)
        if not counts:
            raise ValueError("fit requires at least one label")
        self.label_ = min(counts, key=lambda label: (-counts[label], repr(label)))
        return self

    def predict(self, count: int) -> list[Label]:
        if self.label_ is None:
            raise RuntimeError("fit must be called before predict")
        if count < 0:
            raise ValueError("count cannot be negative")
        return [self.label_] * count


def accuracy(actual: Sequence[Label], predicted: Sequence[Label]) -> float:
    validate_labels(actual, predicted)
    return sum(a == p for a, p in zip(actual, predicted, strict=True)) / len(actual)


def precision_recall(actual: Sequence[Label], predicted: Sequence[Label], positive: Label) -> tuple[float, float]:
    validate_labels(actual, predicted)
    true_positive = sum(a == positive and p == positive for a, p in zip(actual, predicted, strict=True))
    false_positive = sum(a != positive and p == positive for a, p in zip(actual, predicted, strict=True))
    false_negative = sum(a == positive and p != positive for a, p in zip(actual, predicted, strict=True))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return precision, recall


def grouped_split(groups: Sequence[Label], test_fraction: float = 0.2, seed: int = 0) -> tuple[list[int], list[int]]:
    """Split indices while keeping every group wholly in train or test."""
    if not groups:
        raise ValueError("split requires at least one group")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    unique = sorted(set(groups), key=repr)
    if len(unique) < 2:
        raise ValueError("grouped split requires at least two groups")
    random.Random(seed).shuffle(unique)
    test_count = max(1, min(len(unique) - 1, round(len(unique) * test_fraction)))
    test_groups = set(unique[:test_count])
    train = [index for index, group in enumerate(groups) if group not in test_groups]
    test = [index for index, group in enumerate(groups) if group in test_groups]
    return train, test
