"""Validated SciPy optimization and integration examples."""

from __future__ import annotations
import math
from scipy.integrate import quad
from scipy.optimize import minimize, root_scalar


def positive_root(value: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError("value must be finite and nonnegative")
    if value == 0:
        return 0.0
    result = root_scalar(lambda x: x * x - value, bracket=(0.0, max(1.0, value)), xtol=1e-12)
    if not result.converged or abs(result.root**2 - value) > 1e-9 * max(1.0, value):
        raise RuntimeError("root did not meet residual contract")
    return float(result.root)


def constrained_minimum() -> tuple[float, float]:
    result = minimize(lambda x: (x[0] - 3) ** 2, x0=[0.0], bounds=[(0.0, 2.0)], method="L-BFGS-B")
    if not result.success or not 0 <= result.x[0] <= 2:
        raise RuntimeError(result.message)
    return float(result.x[0]), float(result.fun)


def normal_mass() -> tuple[float, float]:
    density = lambda x: math.exp(-(x * x) / 2) / math.sqrt(2 * math.pi)
    value, error = quad(density, -1, 1, epsabs=1e-12)
    return value, error
