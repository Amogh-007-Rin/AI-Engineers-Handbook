"""Detectron2-style dataset dictionary validator; no model download required."""

import math


def validate_record(record, class_count):
    for key in ("file_name", "image_id", "height", "width", "annotations"):
        if key not in record:
            raise ValueError(f"missing dataset field: {key}")
    if record["height"] <= 0 or record["width"] <= 0 or class_count < 1:
        raise ValueError("positive dimensions and class count required")
    for annotation in record["annotations"]:
        box = annotation.get("bbox", [])
        if len(box) != 4 or not all(math.isfinite(value) for value in box):
            raise ValueError("finite four-value boxes required")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= record["width"] and 0 <= y1 < y2 <= record["height"]):
            raise ValueError("XYXY box outside image")
        category = annotation.get("category_id")
        if not isinstance(category, int) or not 0 <= category < class_count:
            raise ValueError("category IDs must be contiguous")
    return True
