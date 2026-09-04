"""Dependency-free Spark pipeline specification validator."""


def validate_pipeline(spec):
    if not spec.get("input_schema") or not spec.get("output_schema"):
        raise ValueError("input and output schemas required")
    if spec.get("join_keys") and not spec.get("null_policy"):
        raise ValueError("joins require explicit null policy")
    if spec.get("writes") and not spec.get("idempotent"):
        raise ValueError("distributed writes must be idempotent")
    if spec.get("driver_collect", False):
        raise ValueError("unbounded driver collection is forbidden")
    return True
