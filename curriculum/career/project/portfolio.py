"""Portfolio evidence manifest validator."""


REQUIRED = {"title", "problem", "stakeholder", "baseline", "data", "tests", "metrics",
            "decision", "limitations", "responsible_ai", "operations", "reproduce"}


def validate_entry(entry):
    missing = REQUIRED - set(entry)
    if missing:
        raise ValueError("portfolio evidence missing: " + ", ".join(sorted(missing)))
    if any(not entry[field] for field in REQUIRED):
        raise ValueError("portfolio evidence fields must be non-empty")
    claims = entry.get("claims", [])
    if any(not claim.get("evidence") for claim in claims):
        raise ValueError("every public claim requires evidence")
    return True
