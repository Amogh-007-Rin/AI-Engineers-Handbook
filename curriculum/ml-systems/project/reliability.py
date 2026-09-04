"""Small reliability calculations for production-readiness exercises."""

def availability(successful: int, total: int) -> float:
    if total <= 0 or not 0 <= successful <= total:
        raise ValueError("counts must satisfy 0 <= successful <= total and total > 0")
    return successful / total


def error_budget(total: int, target_availability: float) -> float:
    if total < 0 or not 0 < target_availability <= 1:
        raise ValueError("invalid total or availability target")
    return total * (1 - target_availability)


def burn_rate(observed_errors: int, total: int, target_availability: float) -> float:
    budget = error_budget(total, target_availability)
    if budget == 0:
        return float("inf") if observed_errors else 0.0
    return observed_errors / budget
