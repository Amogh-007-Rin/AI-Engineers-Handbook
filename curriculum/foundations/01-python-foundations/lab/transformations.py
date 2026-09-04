"""Typed, dependency-free transformations for the Python foundations lab."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
import json
import math
from typing import Any


@dataclass(frozen=True)
class Observation:
    subject_id: str
    group: str
    score: float


def parse_observation(raw: Mapping[str, Any]) -> Observation:
    """Validate one untrusted record and preserve useful failure context."""
    subject_id = str(raw.get("subject_id", "")).strip()
    group = str(raw.get("group", "")).strip().lower()
    if not subject_id:
        raise ValueError("subject_id must be non-empty")
    if group not in {"control", "treatment"}:
        raise ValueError("group must be 'control' or 'treatment'")
    try:
        score = float(raw["score"])
    except KeyError as error:
        raise ValueError("score is required") from error
    except (TypeError, ValueError) as error:
        raise ValueError(f"score must be numeric, received {raw.get('score')!r}") from error
    if not math.isfinite(score):
        raise ValueError("score must be finite")
    return Observation(subject_id=subject_id, group=group, score=score)


def iter_observations(records: Iterable[Mapping[str, Any]]) -> Iterator[Observation]:
    """Lazily validate records, rejecting duplicate subject identifiers."""
    seen: set[str] = set()
    for raw in records:
        observation = parse_observation(raw)
        if observation.subject_id in seen:
            raise ValueError(f"duplicate subject_id: {observation.subject_id!r}")
        seen.add(observation.subject_id)
        yield observation


def summarize(observations: Iterable[Observation]) -> dict[str, dict[str, float | int]]:
    """Return deterministic count and mean by group without mutating inputs."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for observation in observations:
        totals[observation.group] = totals.get(observation.group, 0.0) + observation.score
        counts[observation.group] = counts.get(observation.group, 0) + 1
    if not counts:
        raise ValueError("at least one observation is required")
    return {
        group: {"count": counts[group], "mean": totals[group] / counts[group]}
        for group in sorted(counts)
    }


def example() -> dict[str, object]:
    records = [
        {"subject_id": "A-01", "group": "Control", "score": "0.50"},
        {"subject_id": "B-02", "group": "treatment", "score": 0.75},
        {"subject_id": "C-03", "group": "treatment", "score": 0.95},
    ]
    observations = tuple(iter_observations(records))
    return {
        "observations": [asdict(item) for item in observations],
        "summary": summarize(observations),
    }


def main() -> int:
    print(json.dumps(example(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
