"""Transparent box geometry used by the computer-vision project."""

from __future__ import annotations

from math import isfinite

Box = tuple[float, float, float, float]


def area(box: Box) -> float:
    x1, y1, x2, y2 = box
    if not all(isfinite(value) for value in box):
        raise ValueError("box coordinates must be finite")
    if x2 < x1 or y2 < y1:
        raise ValueError("box maximum cannot be below minimum")
    return (x2 - x1) * (y2 - y1)


def intersection_over_union(first: Box, second: Box) -> float:
    first_area, second_area = area(first), area(second)
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first_area + second_area - intersection
    if union == 0:
        raise ValueError("IoU is undefined when both boxes have zero area")
    return intersection / union
