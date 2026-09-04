"""Rolling-origin ARIMA evaluation against a last-value baseline."""

from __future__ import annotations
import math
from statsmodels.tsa.arima.model import ARIMA


def mae(actual: list[float], predicted: list[float]) -> float:
    if not actual or len(actual) != len(predicted):
        raise ValueError("actual and predicted must have equal nonzero length")
    return sum(abs(a - p) for a, p in zip(actual, predicted, strict=True)) / len(actual)


def rolling_forecasts(values: list[float], initial: int) -> tuple[list[float], list[float], list[float]]:
    if len(values) < initial + 2 or initial < 8 or not all(math.isfinite(v) for v in values):
        raise ValueError("finite series and at least two forecast origins required")
    actual, naive, arima = [], [], []
    for origin in range(initial, len(values)):
        history = values[:origin]
        actual.append(values[origin])
        naive.append(history[-1])
        fitted = ARIMA(history, order=(0, 1, 0), trend="t").fit()
        arima.append(float(fitted.forecast(1)[0]))
    return actual, naive, arima
