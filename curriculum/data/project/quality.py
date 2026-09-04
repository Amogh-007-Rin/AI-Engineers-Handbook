"""Dependency-free row schema and quality checks."""

import math


def audit(rows, schema, unique_key):
    rows = list(rows)
    if not rows:
        raise ValueError("dataset must be non-empty")
    if unique_key not in schema:
        raise ValueError("unique key must be declared in schema")
    issues, seen = [], set()
    for index, row in enumerate(rows):
        missing = set(schema) - set(row)
        extra = set(row) - set(schema)
        if missing or extra:
            issues.append({"row": index, "kind": "schema", "missing": sorted(missing), "extra": sorted(extra)})
            continue
        identity = row[unique_key]
        if identity in seen:
            issues.append({"row": index, "kind": "duplicate", "value": identity})
        seen.add(identity)
        for field, expected in schema.items():
            value = row[field]
            if value is not None and not isinstance(value, expected):
                issues.append({"row": index, "kind": "type", "field": field})
            if isinstance(value, float) and not math.isfinite(value):
                issues.append({"row": index, "kind": "nonfinite", "field": field})
    return {"rows": len(rows), "issues": issues, "valid": not issues}
