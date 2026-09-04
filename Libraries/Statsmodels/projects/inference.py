"""Deterministic OLS fixture with explicit intercept and diagnostics."""

from __future__ import annotations
import numpy as np
import statsmodels.api as sm


def fit_line(x: list[float], y: list[float]):
    if len(x) != len(y) or len(x) < 3 or not np.isfinite(x + y).all():
        raise ValueError("finite paired data with at least three rows required")
    design = sm.add_constant(np.asarray(x, dtype=float), has_constant="add")
    return sm.OLS(np.asarray(y, dtype=float), design).fit()


def prediction_at(result, x: float) -> tuple[float, float, float]:
    frame = result.get_prediction([1.0, x]).summary_frame(alpha=0.05).iloc[0]
    return float(frame["mean"]), float(frame["mean_ci_lower"]), float(frame["mean_ci_upper"])
