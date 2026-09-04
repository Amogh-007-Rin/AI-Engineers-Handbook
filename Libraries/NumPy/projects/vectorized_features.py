"""Reference-quality vectorized features with explicit fitted-state contracts."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatMatrix = NDArray[np.float64]


def matrix(values: ArrayLike) -> FloatMatrix:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError("expected a nonempty 2D matrix")
    if not np.isfinite(result).all():
        raise ValueError("matrix contains nonfinite values")
    return result


@dataclass
class Standardizer:
    mean_: FloatMatrix | None = None
    scale_: FloatMatrix | None = None

    def fit(self, values: ArrayLike) -> "Standardizer":
        data = matrix(values)
        self.mean_ = data.mean(axis=0, keepdims=True)
        scale = data.std(axis=0, keepdims=True)
        self.scale_ = np.where(scale == 0, 1.0, scale)
        return self

    def transform(self, values: ArrayLike) -> FloatMatrix:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("fit must be called before transform")
        data = matrix(values)
        if data.shape[1] != self.mean_.shape[1]:
            raise ValueError("feature count differs from fitted data")
        return (data - self.mean_) / self.scale_

    def fit_transform(self, values: ArrayLike) -> FloatMatrix:
        return self.fit(values).transform(values)


def pairwise_interactions(values: ArrayLike) -> FloatMatrix:
    data = matrix(values)
    left, right = np.triu_indices(data.shape[1], k=1)
    return data[:, left] * data[:, right]


def cosine_similarity(values: ArrayLike) -> FloatMatrix:
    data = matrix(values)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    normalized = data / safe
    result = normalized @ normalized.T
    zero_rows = norms[:, 0] == 0
    result[zero_rows, :] = 0
    result[:, zero_rows] = 0
    return result
