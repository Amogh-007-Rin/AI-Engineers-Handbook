"""Dependency-free Bento service specification validator."""


def validate_service(spec):
    for key in ("name", "model_ref", "input_schema", "output_schema", "timeout_ms", "max_batch"):
        if key not in spec:
            raise ValueError(f"missing service field: {key}")
    if "@" not in spec["model_ref"] or spec["timeout_ms"] <= 0 or spec["max_batch"] < 1:
        raise ValueError("model must be versioned and limits positive")
    if not spec.get("readiness") or not spec.get("metrics"):
        raise ValueError("readiness and metrics are required")
    return True
