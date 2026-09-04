"""Dependency-free rolling-origin baseline."""


def rolling_origins(values, minimum_train, horizon=1):
    if minimum_train < 1 or horizon < 1 or len(values) < minimum_train + horizon:
        raise ValueError("insufficient history or invalid window")
    return [(values[:origin], values[origin:origin + horizon])
            for origin in range(minimum_train, len(values) - horizon + 1)]


def last_value_forecast(history, horizon):
    if not history or horizon < 1:
        raise ValueError("history and positive horizon required")
    return [history[-1]] * horizon


def mae(actual, predicted):
    if len(actual) != len(predicted) or not actual:
        raise ValueError("aligned non-empty arrays required")
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)
