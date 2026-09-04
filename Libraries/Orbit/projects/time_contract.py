"""Dependency-free time contract used before fitting any Orbit model."""

from __future__ import annotations
from datetime import datetime, timedelta


def validate_daily_dates(values: list[datetime]) -> None:
    if len(values) < 3 or values != sorted(values) or len(set(values)) != len(values):
        raise ValueError("dates must be unique sorted and contain at least three observations")
    if any(right - left != timedelta(days=1) for left, right in zip(values, values[1:])):
        raise ValueError("dates must have daily frequency without gaps")


def rolling_origins(length: int, initial: int, horizon: int) -> list[tuple[range, range]]:
    if initial < 2 or horizon <= 0 or length < initial + horizon:
        raise ValueError("invalid rolling-origin dimensions")
    return [(range(0, origin), range(origin, origin + horizon)) for origin in range(initial, length - horizon + 1)]
