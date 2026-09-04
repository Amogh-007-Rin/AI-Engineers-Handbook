"""Dependency-free Ray job specification validator."""


def validate_job(job):
    if not job.get("name") or job.get("retries", -1) < 0:
        raise ValueError("job name and non-negative retry budget required")
    resources = job.get("resources", {})
    if not resources or any(value <= 0 for value in resources.values()):
        raise ValueError("positive resource reservations required")
    if job.get("side_effects") and not job.get("idempotency_key"):
        raise ValueError("side effects require idempotency")
    return True
