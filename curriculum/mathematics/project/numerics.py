"""Small transparent numerical contracts for mathematical verification."""

import math


def dot(left, right):
    if len(left) != len(right) or not left:
        raise ValueError("equal non-empty vector dimensions required")
    result = sum(a * b for a, b in zip(left, right))
    if not math.isfinite(result):
        raise ValueError("dot product is non-finite")
    return result


def central_difference(function, x, step=1e-5):
    if step <= 0 or not math.isfinite(step):
        raise ValueError("finite positive step required")
    return (function(x + step) - function(x - step)) / (2 * step)
