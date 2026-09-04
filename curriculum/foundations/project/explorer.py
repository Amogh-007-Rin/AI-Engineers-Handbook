"""Dependency-free tabular profile used by the foundation project."""

import math


def profile(rows):
    rows = list(rows)
    if not rows:
        raise ValueError("at least one row required")
    columns = tuple(rows[0])
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("column names must be non-empty and unique")
    if any(tuple(row) != columns for row in rows):
        raise ValueError("all rows must share the same ordered schema")
    missing = {column: sum(row[column] in (None, "") for row in rows) for column in columns}
    numeric = {}
    for column in columns:
        raw_values = [row[column] for row in rows if row[column] not in (None, "")]
        try:
            values = [float(value) for value in raw_values]
        except (TypeError, ValueError):
            continue
        if values:
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"numeric column {column!r} contains non-finite values")
            numeric[column] = {"minimum": min(values), "maximum": max(values), "mean": sum(values) / len(values)}
    return {"rows": len(rows), "columns": columns, "missing": missing, "numeric": numeric}
