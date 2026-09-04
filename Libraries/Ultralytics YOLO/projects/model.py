"""YOLO normalized annotation contract validator."""

import math


def validate_labels(rows, class_count):
    if class_count < 1:
        raise ValueError("class_count must be positive")
    for row in rows:
        if len(row) != 5:
            raise ValueError("label rows require class x_center y_center width height")
        cls, x, y, width, height = row
        if int(cls) != cls or not 0 <= cls < class_count:
            raise ValueError("class ID outside configured range")
        if not all(math.isfinite(v) for v in (x, y, width, height)):
            raise ValueError("coordinates must be finite")
        if width <= 0 or height <= 0 or not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError("invalid normalized box")
        if x - width / 2 < 0 or x + width / 2 > 1 or y - height / 2 < 0 or y + height / 2 > 1:
            raise ValueError("box extends beyond image")
    return True
