"""Transparent linear-algebra operations for the foundation mathematics lab."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import json
import math

Vector = tuple[float, ...]
Matrix = tuple[Vector, ...]


def vector(values: Iterable[float]) -> Vector:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError("vector must be non-empty")
    if not all(math.isfinite(value) for value in result):
        raise ValueError("vector values must be finite")
    return result


def matrix(rows: Iterable[Iterable[float]]) -> Matrix:
    result = tuple(vector(row) for row in rows)
    if not result:
        raise ValueError("matrix must have at least one row")
    width = len(result[0])
    if any(len(row) != width for row in result):
        raise ValueError("matrix rows must have equal non-zero length")
    return result


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    x, y = vector(left), vector(right)
    if len(x) != len(y):
        raise ValueError("dot product requires equal dimensions")
    result = math.fsum(a * b for a, b in zip(x, y))
    if not math.isfinite(result):
        raise ValueError("dot product result must be finite")
    return result


def transpose(value: Iterable[Iterable[float]]) -> Matrix:
    source = matrix(value)
    return tuple(tuple(source[row][column] for row in range(len(source)))
                 for column in range(len(source[0])))


def matvec(value: Iterable[Iterable[float]], weights: Sequence[float]) -> Vector:
    source, w = matrix(value), vector(weights)
    if len(source[0]) != len(w):
        raise ValueError("matrix columns must equal vector dimension")
    return tuple(dot(row, w) for row in source)


def matmul(left: Iterable[Iterable[float]], right: Iterable[Iterable[float]]) -> Matrix:
    a, b = matrix(left), matrix(right)
    if len(a[0]) != len(b):
        raise ValueError("left columns must equal right rows")
    columns = transpose(b)
    return tuple(tuple(dot(row, column) for column in columns) for row in a)


def norm(value: Sequence[float]) -> float:
    x = vector(value)
    return math.sqrt(dot(x, x))


def project(value: Sequence[float], direction: Sequence[float]) -> Vector:
    x, u = vector(value), vector(direction)
    if len(x) != len(u):
        raise ValueError("projection requires equal dimensions")
    denominator = dot(u, u)
    if denominator == 0.0:
        raise ValueError("projection direction must be non-zero")
    scale = dot(x, u) / denominator
    return tuple(scale * coordinate for coordinate in u)


def determinant_2x2(value: Iterable[Iterable[float]]) -> float:
    source = matrix(value)
    if len(source) != 2 or len(source[0]) != 2:
        raise ValueError("determinant_2x2 requires shape (2, 2)")
    return source[0][0] * source[1][1] - source[0][1] * source[1][0]


def solve_2x2(value: Iterable[Iterable[float]], target: Sequence[float], tolerance: float = 1e-12) -> Vector:
    source, b = matrix(value), vector(target)
    if len(source) != 2 or len(source[0]) != 2 or len(b) != 2:
        raise ValueError("solve_2x2 requires a (2, 2) matrix and length-two target")
    determinant = determinant_2x2(source)
    scale = max(abs(item) for row in source for item in row)
    if abs(determinant) <= tolerance * max(1.0, scale * scale):
        raise ValueError("system is singular within the declared tolerance")
    a, c = source[0]
    d, e = source[1]
    return ((b[0] * e - c * b[1]) / determinant,
            (a * b[1] - b[0] * d) / determinant)


def sensitivity_example() -> float:
    source = ((1.0, 1.0), (1.0, 1.000001))
    baseline = solve_2x2(source, (2.0, 2.000001), tolerance=1e-15)
    perturbed = solve_2x2(source, (2.0, 2.0000011), tolerance=1e-15)
    output_change = norm(tuple(after - before for before, after in zip(baseline, perturbed)))
    return output_change / 1e-7


def main() -> int:
    result = {
        "predictions": matvec(((1, 2), (0, -3), (3, 0)), (0.5, 1.0)),
        "projection": project((3, 4), (1, 0)),
        "sensitivity_ratio": sensitivity_example(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
