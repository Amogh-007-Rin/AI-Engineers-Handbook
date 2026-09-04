"""Dependency-free fitted-state feature pipeline for leakage-safe preprocessing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from statistics import median
from typing import Any

FIELDS = {"age", "plan"}
ARTIFACT_VERSION = 1
OTHER = "__other__"


@dataclass(frozen=True)
class PipelineState:
    version: int
    age_median: float
    age_scale: float
    plans: tuple[str, ...]


def numeric(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("age must be a number or null")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("age must be finite")
    return result


def validate_schema(record: dict[str, Any]) -> None:
    if set(record) != FIELDS:
        missing, extra = sorted(FIELDS - set(record)), sorted(set(record) - FIELDS)
        raise ValueError(f"schema mismatch: missing={missing}, extra={extra}")


def fit(records: list[dict[str, Any]]) -> PipelineState:
    if not records:
        raise ValueError("fit requires training records")
    for record in records:
        validate_schema(record)
    ages = [numeric(record["age"]) for record in records if record["age"] is not None]
    if not ages:
        raise ValueError("age cannot be all-missing during fit")
    center = float(median(ages))
    scale = max(abs(value - center) for value in ages)
    if scale == 0:
        scale = 1.0
    plans = []
    for record in records:
        plan = record["plan"]
        if not isinstance(plan, str) or not plan.strip():
            raise ValueError("plan must be a non-empty string")
        normalized = plan.strip().lower()
        if normalized == OTHER:
            raise ValueError(f"plan value {OTHER!r} is reserved")
        plans.append(normalized)
    return PipelineState(ARTIFACT_VERSION, center, scale, tuple(sorted(set(plans))))


def feature_names(state: PipelineState) -> tuple[str, ...]:
    return ("age_scaled", "age_missing", *(f"plan={plan}" for plan in state.plans), "plan=other")


def transform_one(record: dict[str, Any], state: PipelineState) -> tuple[float, ...]:
    if state.version != ARTIFACT_VERSION:
        raise ValueError("pipeline artifact version is incompatible")
    validate_schema(record)
    if record["age"] is None:
        age, missing = state.age_median, 1.0
    else:
        age, missing = numeric(record["age"]), 0.0
    plan = record["plan"]
    if not isinstance(plan, str) or not plan.strip():
        raise ValueError("plan must be a non-empty string")
    normalized = plan.strip().lower()
    known = tuple(float(normalized == value) for value in state.plans)
    other = float(normalized not in state.plans)
    return ((age - state.age_median) / state.age_scale, missing, *known, other)


def transform(records: list[dict[str, Any]], state: PipelineState) -> list[tuple[float, ...]]:
    return [transform_one(record, state) for record in records]


def dumps(state: PipelineState) -> str:
    payload = asdict(state)
    payload["plans"] = list(state.plans)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def loads(payload: str) -> PipelineState:
    try:
        raw = json.loads(payload)
        if set(raw) != {"version", "age_median", "age_scale", "plans"}:
            raise ValueError("artifact schema mismatch")
        state = PipelineState(int(raw["version"]), float(raw["age_median"]),
                              float(raw["age_scale"]), tuple(raw["plans"]))
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise ValueError("invalid pipeline artifact") from error
    if state.version != ARTIFACT_VERSION or state.age_scale <= 0 or not math.isfinite(state.age_scale):
        raise ValueError("pipeline artifact version or scale is incompatible")
    return state


def example() -> dict[str, object]:
    training = [{"age": 20, "plan": "basic"}, {"age": 30, "plan": "pro"},
                {"age": 40, "plan": "basic"}]
    state = fit(training)
    values = transform([{"age": None, "plan": "enterprise"}, {"age": 40, "plan": "pro"}], state)
    return {"state": asdict(state), "feature_names": feature_names(state), "values": values,
            "artifact": dumps(state)}


def main() -> int:
    print(json.dumps(example(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
