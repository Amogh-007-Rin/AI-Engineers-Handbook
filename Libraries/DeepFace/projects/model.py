"""Biometric verification threshold and governance contract."""

import math


def verify(distance, threshold):
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (distance, threshold)):
        raise ValueError("finite numeric distance and threshold required")
    if distance < 0 or threshold <= 0:
        raise ValueError("distance must be non-negative and threshold positive")
    return {"verified": distance <= threshold, "distance": distance, "threshold": threshold}


def validate_governance(policy):
    required = ("consent", "purpose", "retention_days", "deletion", "encryption", "human_review")
    if any(not policy.get(key) for key in required) or policy["retention_days"] <= 0:
        raise ValueError("complete biometric governance policy required")
    return True
