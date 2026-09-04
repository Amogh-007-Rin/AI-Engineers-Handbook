"""Dependency-free event quality, split leakage, and join-cardinality contracts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import asdict, dataclass
import json
import math
from typing import Any

FIELDS = {"event_id", "entity_id", "observed_at", "prediction_time", "score", "status"}
STATUSES = {"pending", "complete"}


@dataclass(frozen=True)
class Event:
    event_id: str
    entity_id: str
    observed_at: int
    prediction_time: int
    score: float | None
    status: str


def inspect_event(raw: Mapping[str, Any]) -> tuple[Event | None, tuple[str, ...]]:
    """Return a normalized event or named issues; never silently coerce a bad row."""
    issues: list[str] = []
    if set(raw) != FIELDS:
        return None, ("schema",)
    event_id = raw["event_id"] if isinstance(raw["event_id"], str) else ""
    entity_id = raw["entity_id"] if isinstance(raw["entity_id"], str) else ""
    if not event_id.strip() or not entity_id.strip():
        issues.append("identity")
    observed_at, prediction_time = raw["observed_at"], raw["prediction_time"]
    if (isinstance(observed_at, bool) or not isinstance(observed_at, int)
            or isinstance(prediction_time, bool) or not isinstance(prediction_time, int)):
        issues.append("timestamp_type")
    elif observed_at > prediction_time:
        issues.append("temporal_availability")
    status = raw["status"]
    if status not in STATUSES:
        issues.append("status")
    score = raw["score"]
    if score is not None:
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            issues.append("score_type")
        elif not math.isfinite(float(score)):
            issues.append("nonfinite")
        elif not 0 <= float(score) <= 1:
            issues.append("range")
    if status == "complete" and score is None:
        issues.append("cross_field_completeness")
    if issues:
        return None, tuple(issues)
    return Event(event_id.strip(), entity_id.strip(), observed_at, prediction_time,
                 None if score is None else float(score), status), ()


def audit_events(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = tuple(rows)
    if not materialized:
        raise ValueError("event dataset must be non-empty")
    accepted: list[Event] = []
    quarantine: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(materialized):
        event, issues = inspect_event(raw)
        if event is not None and event.event_id in seen:
            event, issues = None, ("duplicate_event",)
        if event is None:
            quarantine.append({"row": index, "issues": list(issues)})
        else:
            seen.add(event.event_id)
            accepted.append(event)
    issue_counts = Counter(issue for row in quarantine for issue in row["issues"])
    return {
        "received": len(materialized),
        "accepted": len(accepted),
        "quarantined": len(quarantine),
        "issue_counts": dict(sorted(issue_counts.items())),
        "quarantine": quarantine,
        "events": [asdict(event) for event in accepted],
    }


def split_overlap(train_entities: Iterable[Hashable], test_entities: Iterable[Hashable]) -> set[Hashable]:
    """Return identities shared by train and test, a direct group-leakage signal."""
    return set(train_entities) & set(test_entities)


def validate_join(left_keys: Iterable[Hashable], right_keys: Iterable[Hashable], relation: str) -> int:
    """Validate declared key cardinality and return the expected inner-join row count."""
    allowed = {"one-to-one", "one-to-many", "many-to-one", "many-to-many"}
    if relation not in allowed:
        raise ValueError(f"relation must be one of {sorted(allowed)}")
    left, right = Counter(left_keys), Counter(right_keys)
    if relation in {"one-to-one", "one-to-many"} and any(count > 1 for count in left.values()):
        raise ValueError("left keys violate declared cardinality")
    if relation in {"one-to-one", "many-to-one"} and any(count > 1 for count in right.values()):
        raise ValueError("right keys violate declared cardinality")
    return sum(left[key] * right[key] for key in left.keys() & right.keys())


def example() -> dict[str, Any]:
    rows = [
        {"event_id": "E1", "entity_id": "A", "observed_at": 10, "prediction_time": 10, "score": 0.7, "status": "complete"},
        {"event_id": "E2", "entity_id": "B", "observed_at": 11, "prediction_time": 12, "score": None, "status": "pending"},
        {"event_id": "E3", "entity_id": "C", "observed_at": 15, "prediction_time": 14, "score": 1.2, "status": "complete"},
        {"event_id": "E4", "entity_id": "D", "observed_at": 12, "prediction_time": 12, "score": None, "status": "complete"},
    ]
    return audit_events(rows)


def main() -> int:
    print(json.dumps(example(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
