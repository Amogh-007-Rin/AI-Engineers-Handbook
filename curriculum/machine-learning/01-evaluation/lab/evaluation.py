"""Dependency-free binary evaluation contracts for the classical ML lesson."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
import json
import math
import random


@dataclass(frozen=True)
class Confusion:
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int


def validate(actual: Sequence[int], probabilities: Sequence[float]) -> None:
    if not actual:
        raise ValueError("evaluation requires at least one observation")
    if len(actual) != len(probabilities):
        raise ValueError("labels and probabilities must have equal length")
    if any(label not in {0, 1} or isinstance(label, bool) for label in actual):
        raise ValueError("labels must be integers zero or one")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           or not math.isfinite(float(value)) or not 0 <= float(value) <= 1
           for value in probabilities):
        raise ValueError("probabilities must be finite numbers in [0, 1]")


def predictions(probabilities: Sequence[float], threshold: float) -> tuple[int, ...]:
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    return tuple(int(value >= threshold) for value in probabilities)


def confusion(actual: Sequence[int], probabilities: Sequence[float], threshold: float) -> Confusion:
    validate(actual, probabilities)
    predicted = predictions(probabilities, threshold)
    return Confusion(
        true_positive=sum(y == 1 and p == 1 for y, p in zip(actual, predicted)),
        false_positive=sum(y == 0 and p == 1 for y, p in zip(actual, predicted)),
        true_negative=sum(y == 0 and p == 0 for y, p in zip(actual, predicted)),
        false_negative=sum(y == 1 and p == 0 for y, p in zip(actual, predicted)),
    )


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def metrics(counts: Confusion) -> dict[str, float | int | None]:
    total = sum(asdict(counts).values())
    return {
        **asdict(counts),
        "support": total,
        "accuracy": safe_ratio(counts.true_positive + counts.true_negative, total),
        "precision": safe_ratio(counts.true_positive, counts.true_positive + counts.false_positive),
        "recall": safe_ratio(counts.true_positive, counts.true_positive + counts.false_negative),
        "specificity": safe_ratio(counts.true_negative, counts.true_negative + counts.false_positive),
    }


def brier_score(actual: Sequence[int], probabilities: Sequence[float]) -> float:
    validate(actual, probabilities)
    return math.fsum((float(score) - label) ** 2 for label, score in zip(actual, probabilities)) / len(actual)


def threshold_cost(actual: Sequence[int], probabilities: Sequence[float], threshold: float,
                   false_positive_cost: float, false_negative_cost: float) -> float:
    if false_positive_cost < 0 or false_negative_cost < 0:
        raise ValueError("error costs must be non-negative")
    counts = confusion(actual, probabilities, threshold)
    return counts.false_positive * false_positive_cost + counts.false_negative * false_negative_cost


def choose_threshold(actual: Sequence[int], probabilities: Sequence[float], thresholds: Iterable[float],
                     false_positive_cost: float, false_negative_cost: float) -> dict[str, float]:
    candidates = tuple(float(value) for value in thresholds)
    if not candidates:
        raise ValueError("at least one threshold is required")
    rows = [(threshold_cost(actual, probabilities, value, false_positive_cost, false_negative_cost), value)
            for value in candidates]
    cost, threshold = min(rows, key=lambda row: (row[0], row[1]))
    return {"threshold": threshold, "cost": cost}


def bootstrap_accuracy(actual: Sequence[int], probabilities: Sequence[float], threshold: float,
                       repetitions: int = 500, seed: int = 0) -> tuple[float, float]:
    validate(actual, probabilities)
    if repetitions < 20:
        raise ValueError("bootstrap requires at least 20 repetitions")
    predicted = predictions(probabilities, threshold)
    rng = random.Random(seed)
    values = []
    for _ in range(repetitions):
        indices = [rng.randrange(len(actual)) for _ in actual]
        values.append(sum(actual[i] == predicted[i] for i in indices) / len(indices))
    values.sort()
    return values[int(0.025 * repetitions)], values[min(repetitions - 1, int(0.975 * repetitions))]


def slice_report(actual: Sequence[int], probabilities: Sequence[float], slices: Sequence[str],
                 threshold: float) -> dict[str, dict[str, float | int | None]]:
    validate(actual, probabilities)
    if len(slices) != len(actual):
        raise ValueError("slice labels must match evaluation length")
    result = {}
    for name in sorted(set(slices)):
        indices = [index for index, value in enumerate(slices) if value == name]
        result[name] = metrics(confusion([actual[i] for i in indices],
                                         [probabilities[i] for i in indices], threshold))
    return result


def example() -> dict[str, object]:
    actual = [1, 1, 0, 0]
    probabilities = [0.9, 0.65, 0.55, 0.1]
    selected = choose_threshold(actual, probabilities, [0.4, 0.6, 0.8], 2, 3)
    threshold = selected["threshold"]
    return {
        "selected": selected,
        "overall": metrics(confusion(actual, probabilities, threshold)),
        "brier_score": brier_score(actual, probabilities),
        "accuracy_interval": bootstrap_accuracy(actual, probabilities, threshold, seed=7),
        "slices": slice_report(actual, probabilities, ["new", "returning", "new", "returning"], threshold),
    }


def main() -> int:
    print(json.dumps(example(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
